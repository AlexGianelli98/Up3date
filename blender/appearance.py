"""CityJSON appearance helpers for Blender import."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import bpy

from ..models.cityjson import Appearance
from ..models.geomprimitives import GeometryPrimitive, MaterialValue, TextureTheme
from .blender_types import BlenderMaterial
from .scene import assign_properties, blender_safe_name

UVCoordinate = tuple[float, float]
FaceUVs = list[UVCoordinate] | None
MaterialCacheKey = tuple[str, int | None, int | None, str | None]


@dataclass
class FaceTexture:
    texture_index: int | None = None
    uvs: FaceUVs = None


@dataclass
class AppliedAppearance:
    materials: list[BlenderMaterial]
    material_indices: list[int | None]
    uvs_by_face: list[FaceUVs] | None


class AppearanceImporter:
    """Resolve CityJSON appearance into Blender materials and UV coordinates."""

    def __init__(self, appearance: Appearance, cityjson_path: str) -> None:
        self.appearance = appearance
        self.base_dir = Path(cityjson_path).resolve().parent
        self.image_cache: dict[Path, Any] = {}
        self.material_cache: dict[MaterialCacheKey, BlenderMaterial] = {}

    def apply_to_mesh(
        self,
        geometry: GeometryPrimitive,
        faces: list[tuple[int, ...]],
        materials: list[BlenderMaterial],
        material_indices: list[int | None],
    ) -> AppliedAppearance:
        """Return material slots, face material indices, and optional UVs."""
        face_count = len(faces)
        face_textures = self.get_face_textures(geometry, faces)
        face_materials = self.get_face_materials(geometry, face_count)
        has_textures = any(
            face_texture.uvs is not None for face_texture in face_textures
        )
        has_materials = any(
            material_index is not None for material_index in face_materials
        )

        if not has_textures and not has_materials:
            return AppliedAppearance(materials, material_indices, None)

        resolved_materials = list(materials)
        resolved_indices = self.normalized_material_indices(
            len(faces), resolved_materials, material_indices
        )
        uvs_by_face: list[FaceUVs] = [None] * face_count

        for face_index, face_texture in enumerate(face_textures):
            semantic_type = self.get_semantic_type(
                resolved_materials, resolved_indices[face_index]
            )
            appearance_material_index = face_materials[face_index]
            material = self.get_appearance_material(
                face_texture.texture_index,
                appearance_material_index,
                semantic_type,
            )
            if material is not None:
                resolved_indices[face_index] = self.material_slot(
                    resolved_materials, material
                )
            if face_texture.uvs is not None:
                uvs_by_face[face_index] = face_texture.uvs

        return AppliedAppearance(
            resolved_materials,
            resolved_indices,
            uvs_by_face if has_textures else None,
        )

    @staticmethod
    def normalized_material_indices(
        face_count: int,
        materials: list[BlenderMaterial],
        material_indices: list[int | None],
    ) -> list[int | None]:
        """Build one material index per face while preserving existing defaults."""
        if len(material_indices) == face_count:
            return list(material_indices)
        if materials:
            return [0] * face_count
        return [None] * face_count

    @staticmethod
    def material_slot(
        materials: list[BlenderMaterial], material: BlenderMaterial
    ) -> int:
        """Return the slot for material, appending it when needed."""
        if material in materials:
            return materials.index(material)
        materials.append(material)
        return len(materials) - 1

    @staticmethod
    def get_semantic_type(
        materials: list[BlenderMaterial], material_index: int | None
    ) -> str | None:
        """Return the semantic surface type represented by an existing material."""
        if material_index is None:
            return None
        if material_index < 0 or material_index >= len(materials):
            return None
        material_type = materials[material_index].get("type")
        return material_type if isinstance(material_type, str) else None

    def get_appearance_material(
        self,
        texture_index: int | None,
        material_index: int | None,
        semantic_type: str | None,
    ) -> BlenderMaterial | None:
        """Return a cached Blender material for a CityJSON texture/material pair."""
        if texture_index is None and material_index is None:
            return None
        cache_kind = "texture" if texture_index is not None else "material"
        cache_key = (cache_kind, texture_index, material_index, semantic_type)
        if cache_key in self.material_cache:
            return self.material_cache[cache_key]

        material_data = self.get_material_data(material_index)
        if texture_index is None:
            material = self.create_cityjson_material(
                material_index, material_data, semantic_type
            )
        else:
            texture_data = self.get_texture_data(texture_index)
            material = self.create_textured_material(
                texture_index,
                texture_data,
                material_data,
                semantic_type,
            )

        if material is not None:
            self.material_cache[cache_key] = material
        return material

    def get_texture_data(self, texture_index: int | None) -> dict[str, Any] | None:
        if texture_index is None:
            return None
        if texture_index < 0 or texture_index >= len(self.appearance.textures):
            print(f"CityJSON texture index {texture_index} is out of range")
            return None
        return self.appearance.textures[texture_index]

    def get_material_data(self, material_index: int | None) -> dict[str, Any] | None:
        if material_index is None:
            return None
        if material_index < 0 or material_index >= len(self.appearance.materials):
            print(f"CityJSON material index {material_index} is out of range")
            return None
        return self.appearance.materials[material_index]

    def create_textured_material(
        self,
        texture_index: int,
        texture_data: dict[str, Any] | None,
        material_data: dict[str, Any] | None,
        semantic_type: str | None,
    ) -> BlenderMaterial | None:
        """Create a Blender material with an image texture node."""
        if texture_data is None:
            return None
        image_path = texture_data.get("image")
        if not isinstance(image_path, str):
            print(f"CityJSON texture {texture_index} does not define an image")
            return None
        resolved_path = self.resolve_image_path(image_path)
        if resolved_path is None:
            print(f"Skipping remote CityJSON texture: {image_path}")
            return None
        if not resolved_path.exists():
            print(f"CityJSON texture image not found: {resolved_path}")
            return None

        image = self.load_image(resolved_path)
        if image is None:
            return None

        texture_name = Path(image_path).name or f"texture-{texture_index}"
        name = f"CityJSON {texture_index} {texture_name}"
        if semantic_type is not None:
            name = f"{semantic_type} {name}"
        material = bpy.data.materials.new(name=blender_safe_name(name))
        material["cityjson_texture_index"] = texture_index
        material["cityjson_texture_image"] = image_path
        texture_type = texture_data.get("type")
        if isinstance(texture_type, str):
            material["cityjson_texture_type"] = texture_type
        if semantic_type is not None:
            material["type"] = semantic_type
        self.apply_cityjson_material_properties(material, material_data)
        self.link_image_texture(material, image, texture_data)
        return material

    def create_cityjson_material(
        self,
        material_index: int | None,
        material_data: dict[str, Any] | None,
        semantic_type: str | None,
    ) -> BlenderMaterial | None:
        """Create a Blender material from a CityJSON Material Object."""
        if material_index is None or material_data is None:
            return None
        name = str(material_data.get("name", f"CityJSON Material {material_index}"))
        if semantic_type is not None:
            name = f"{semantic_type} {name}"
        material = bpy.data.materials.new(name=blender_safe_name(name))
        material["cityjson_material_index"] = material_index
        if semantic_type is not None:
            material["type"] = semantic_type
        assign_properties(material, material_data)
        self.apply_cityjson_material_properties(material, material_data)
        return material

    def apply_cityjson_material_properties(
        self, material: BlenderMaterial, material_data: dict[str, Any] | None
    ) -> None:
        """Map display-oriented CityJSON material values to Blender properties."""
        if material_data is None:
            return
        diffuse_color = material_data.get("diffuseColor")
        transparency = material_data.get("transparency")
        alpha = (
            1.0 - float(transparency) if isinstance(transparency, int | float) else 1.0
        )
        if isinstance(diffuse_color, list) and len(diffuse_color) == 3:
            material.diffuse_color = (
                float(diffuse_color[0]),
                float(diffuse_color[1]),
                float(diffuse_color[2]),
                alpha,
            )
        elif alpha < 1.0:
            material.diffuse_color = (
                material.diffuse_color[0],
                material.diffuse_color[1],
                material.diffuse_color[2],
                alpha,
            )
        specular_color = material_data.get("specularColor")
        if isinstance(specular_color, list) and len(specular_color) == 3:
            setattr(material, "specular_color", tuple(float(c) for c in specular_color))
        shininess = material_data.get("shininess")
        if isinstance(shininess, int | float):
            setattr(material, "roughness", max(0.0, min(1.0, 1.0 - float(shininess))))
        if alpha < 1.0:
            if hasattr(material, "surface_render_method"):
                setattr(material, "surface_render_method", "BLENDED")
            elif hasattr(material, "blend_method"):
                setattr(material, "blend_method", "BLEND")

    def link_image_texture(
        self, material: BlenderMaterial, image: Any, texture_data: dict[str, Any]
    ) -> None:
        """Attach an image texture node to a Blender material when nodes exist."""
        try:
            material.use_nodes = True
        except AttributeError:
            pass
        node_tree = getattr(material, "node_tree", None)
        if node_tree is None:
            return
        nodes = node_tree.nodes
        texture_node = nodes.new(type="ShaderNodeTexImage")
        texture_node.image = image
        texture_node.extension = self.blender_wrap_mode(texture_data.get("wrapMode"))
        principled_node = self.find_principled_node(nodes)
        if principled_node is None:
            principled_node = nodes.new(type="ShaderNodeBsdfPrincipled")
        node_tree.links.new(
            texture_node.outputs["Color"], principled_node.inputs["Base Color"]
        )

    @staticmethod
    def find_principled_node(nodes: Any) -> Any | None:
        for node in nodes:
            if getattr(node, "type", None) == "BSDF_PRINCIPLED":
                return node
            if getattr(node, "bl_idname", None) == "ShaderNodeBsdfPrincipled":
                return node
        return None

    @staticmethod
    def blender_wrap_mode(wrap_mode: object) -> str:
        if wrap_mode == "mirror":
            return "MIRROR"
        if wrap_mode == "none":
            return "CLIP"
        if wrap_mode in {"clamp", "border"}:
            return "EXTEND"
        return "REPEAT"

    def resolve_image_path(self, image_path: str) -> Path | None:
        parsed = urlparse(image_path)
        if parsed.scheme in {"http", "https"}:
            return None
        path = Path(image_path)
        if path.is_absolute():
            return path
        return self.base_dir / path

    def load_image(self, image_path: Path) -> Any | None:
        if image_path in self.image_cache:
            return self.image_cache[image_path]
        try:
            image = bpy.data.images.load(str(image_path), check_existing=True)
        except RuntimeError:
            print(f"Unable to load CityJSON texture image: {image_path}")
            return None
        self.image_cache[image_path] = image
        return image

    def get_face_textures(
        self, geometry: GeometryPrimitive, faces: list[tuple[int, ...]]
    ) -> list[FaceTexture]:
        theme = self.get_texture_theme(geometry)
        if theme is None:
            return [FaceTexture() for _face in faces]
        texture_rings = self.exterior_texture_rings(geometry.type, theme.values)
        face_textures = []
        for face, texture_ring in zip(faces, texture_rings, strict=False):
            face_textures.append(self.get_face_texture(face, texture_ring))
        while len(face_textures) < len(faces):
            face_textures.append(FaceTexture())
        return face_textures

    def get_face_texture(
        self, face: tuple[int, ...], texture_ring: object
    ) -> FaceTexture:
        if not isinstance(texture_ring, list) or texture_ring == [None]:
            return FaceTexture()
        if len(texture_ring) != len(face) + 1:
            print("CityJSON texture ring does not match face vertex count")
            return FaceTexture()
        texture_index = texture_ring[0]
        if not isinstance(texture_index, int):
            return FaceTexture()
        uvs = []
        for uv_index in texture_ring[1:]:
            if not isinstance(uv_index, int):
                print("CityJSON texture ring contains a non-integer UV index")
                return FaceTexture()
            if uv_index < 0 or uv_index >= len(self.appearance.vertices_texture):
                print(f"CityJSON UV index {uv_index} is out of range")
                return FaceTexture()
            coords = self.appearance.vertices_texture[uv_index]
            if len(coords) != 2:
                print(f"CityJSON UV coordinate {uv_index} is not two-dimensional")
                return FaceTexture()
            uvs.append((float(coords[0]), float(coords[1])))
        return FaceTexture(texture_index=texture_index, uvs=uvs)

    def get_face_materials(
        self, geometry: GeometryPrimitive, face_count: int
    ) -> list[int | None]:
        material_value = self.get_material_theme(geometry)
        if material_value is None:
            return [None] * face_count
        if material_value.value is not None:
            return [material_value.value] * face_count
        values = self.scalar_values(material_value.values)
        material_indices = [
            value if isinstance(value, int) else None for value in values
        ]
        while len(material_indices) < face_count:
            material_indices.append(None)
        return material_indices[:face_count]

    def get_texture_theme(self, geometry: GeometryPrimitive) -> TextureTheme | None:
        texture = getattr(geometry, "texture", None)
        if not texture:
            return None
        default_theme = self.appearance.default_theme_texture
        if default_theme is not None and default_theme in texture:
            return texture[default_theme]
        return next(iter(texture.values()))

    def get_material_theme(self, geometry: GeometryPrimitive) -> MaterialValue | None:
        material = getattr(geometry, "material", None)
        if not material:
            return None
        default_theme = self.appearance.default_theme_material
        if default_theme is not None and default_theme in material:
            return material[default_theme]
        return next(iter(material.values()))

    @staticmethod
    def exterior_texture_rings(geometry_type: str, values: object) -> list[object]:
        rings: list[object] = []
        if not isinstance(values, list):
            return rings
        if geometry_type in {"MultiSurface", "CompositeSurface"}:
            for surface in values:
                rings.append(AppearanceImporter.first_ring(surface))
        elif geometry_type == "Solid":
            for shell in values:
                if not isinstance(shell, list):
                    continue
                for surface in shell:
                    rings.append(AppearanceImporter.first_ring(surface))
        return rings

    @staticmethod
    def first_ring(surface: object) -> object:
        if not isinstance(surface, list) or not surface:
            return None
        return surface[0]

    @staticmethod
    def scalar_values(values: object) -> list[object]:
        if not isinstance(values, list):
            return [values]
        scalars = []
        for value in values:
            scalars.extend(AppearanceImporter.scalar_values(value))
        return scalars
