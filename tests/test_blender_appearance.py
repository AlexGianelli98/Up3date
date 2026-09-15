"""Tests for applying CityJSON appearance during Blender import."""

import bpy
import pytest

from Up3date.blender.appearance import AppearanceImporter
from Up3date.models.cityjson import Appearance
from Up3date.models.geomprimitives import (
    GeometrySemantics,
    MaterialValue,
    MultiSurface,
    Semantics,
    Solid,
    TextureTheme,
)


class FakeImage:
    def __init__(self, filepath):
        self.filepath = filepath


class ImageStore:
    def __init__(self):
        self.loaded = []

    def load(self, filepath, check_existing=False):
        self.loaded.append((filepath, check_existing))
        return FakeImage(filepath)


class FakeMaterial(dict):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.diffuse_color = (0.8, 0.8, 0.8, 1.0)
        self.node_tree = None


class MaterialStore(list):
    def new(self, name):
        material = FakeMaterial(name)
        self.append(material)
        return material


@pytest.fixture(autouse=True)
def blender_material_state():
    bpy.data.materials = MaterialStore()
    bpy.data.images = ImageStore()


def test_applies_texture_material_and_uvs(tmp_path):
    texture_path = tmp_path / "facade.jpg"
    texture_path.write_bytes(b"image")
    appearance = Appearance(
        textures=[{"type": "JPG", "image": "facade.jpg"}],
        vertices_texture=[[0, 0], [1, 0], [1, 1], [0, 1]],
        default_theme_texture="rgbTexture",
    )
    geometry = Solid(
        lod="2",
        boundaries=[[[[0, 1, 2, 3]]]],
        semantics=GeometrySemantics(
            surfaces=[Semantics(type="WallSurface")],
            values=[[0]],
        ),
        texture={"rgbTexture": TextureTheme(values=[[[[0, 0, 1, 2, 3]]]])},
    )
    fallback_material = FakeMaterial("WallSurface")
    fallback_material["type"] = "WallSurface"

    result = AppearanceImporter(appearance, str(tmp_path / "city.json")).apply_to_mesh(
        geometry,
        [(0, 1, 2, 3)],
        [fallback_material],
        [0],
    )

    assert len(result.materials) == 2
    assert result.material_indices == [1]
    assert result.uvs_by_face == [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]]
    texture_material = result.materials[1]
    assert texture_material["type"] == "WallSurface"
    assert texture_material["cityjson_texture_index"] == 0
    assert texture_material["cityjson_texture_image"] == "facade.jpg"
    assert bpy.data.images.loaded == [(str(texture_path), True)]


def test_applies_cityjson_material_without_texture(tmp_path):
    appearance = Appearance(
        materials=[
            {
                "name": "painted-wall",
                "diffuseColor": [0.1, 0.2, 0.3],
                "transparency": 0.25,
            }
        ],
        default_theme_material="wallTheme",
    )
    geometry = MultiSurface(
        lod="2",
        boundaries=[[[0, 1, 2]]],
        material={"wallTheme": MaterialValue(values=[0])},
    )

    result = AppearanceImporter(appearance, str(tmp_path / "city.json")).apply_to_mesh(
        geometry,
        [(0, 1, 2)],
        [],
        [],
    )

    assert result.material_indices == [0]
    assert result.uvs_by_face is None
    material = result.materials[0]
    assert material.name == "painted-wall"
    assert material["cityjson_material_index"] == 0
    assert material.diffuse_color == (0.1, 0.2, 0.3, 0.75)


def test_invalid_texture_data_falls_back(tmp_path):
    appearance = Appearance(
        textures=[{"type": "JPG", "image": "missing.jpg"}],
        vertices_texture=[[0, 0]],
        default_theme_texture="rgbTexture",
    )
    geometry = Solid(
        lod="2",
        boundaries=[[[[0, 1, 2]]]],
        texture={"rgbTexture": TextureTheme(values=[[[[0, 0, 1, 2]]]])},
    )
    fallback_material = FakeMaterial("fallback")

    result = AppearanceImporter(appearance, str(tmp_path / "city.json")).apply_to_mesh(
        geometry,
        [(0, 1, 2)],
        [fallback_material],
        [],
    )

    assert result.materials == [fallback_material]
    assert result.material_indices == []
    assert result.uvs_by_face is None
