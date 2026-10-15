"""Run Up3date integration tests inside Blender's background process."""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import bpy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT.parent))

coverage_session = None
if coverage_file := os.environ.get("COVERAGE_FILE"):
    import coverage

    coverage_session = coverage.Coverage(
        branch=True,
        data_file=coverage_file,
        omit=[str(PROJECT_ROOT / ".venv" / "*"), str(PROJECT_ROOT / "tests" / "*")],
        source=[str(PROJECT_ROOT)],
    )
    coverage_session.start()

import Up3date
from Up3date import addon
from Up3date.blender.exporter import CityJSONExporter
from Up3date.blender.importer import CityJSONImporter
from Up3date.blender.scene import (
    create_mesh_object,
    get_collection,
    remove_scene_objects,
)
from Up3date.models.cityjson import CityJSONDocument


class BlenderIntegrationTests(unittest.TestCase):
    """Exercise Blender data blocks, registration, and CityJSON round trips."""

    def setUp(self) -> None:
        remove_scene_objects()
        self._addon_registered = False

    def tearDown(self) -> None:
        if self._addon_registered:
            addon.unregister()
        remove_scene_objects()

    def assert_imported_attributes(self, source: dict[str, Any]) -> tuple[int, int]:
        def flatten(
            values: dict[str, Any], prefix: str, result: dict[str, Any]
        ) -> None:
            for key, value in values.items():
                full_name = f"{prefix}.{key}"
                if isinstance(value, dict):
                    flatten(value, full_name, result)
                else:
                    self.assertLessEqual(len(full_name.encode("utf-8")), 63)
                    self.assertNotIn(full_name, result, f"Ambiguous key: {full_name}")
                    result[full_name] = value

        def normalize(value: Any) -> Any:
            if hasattr(value, "to_dict"):
                value = value.to_dict()
            elif hasattr(value, "to_list"):
                value = value.to_list()
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [normalize(item) for item in value]
            return value

        expected = {}
        for object_id, city_object in source["CityObjects"].items():
            properties: dict[str, Any] = {}
            flatten(city_object.get("attributes", {}), "attributes", properties)
            expected[object_id] = properties

        mesh_count = 0
        attribute_count = 0
        mesh_ids = set()
        attributed_ids = set()
        containers = {}
        for obj in bpy.data.objects:
            object_id = obj.get("cityjson_id")
            self.assertIn(object_id, expected)
            city_object = source["CityObjects"][object_id]
            if obj.type == "EMPTY" and obj.get("type") == city_object["type"]:
                containers[object_id] = obj
            if obj.type != "MESH":
                continue
            actual = {
                key: normalize(value)
                for key, value in obj.items()
                if key.startswith("attributes.")
            }
            self.assertEqual(actual, expected[object_id], f"{object_id}: {obj.name}")
            self.assertFalse(
                any(key.startswith("attributes.") for key, _value in obj.data.items())
            )
            self.assertEqual(obj.parent.get("cityjson_id"), object_id)
            self.assertIn(
                (obj.get("type"), obj.get("lod")),
                {
                    (geometry["type"], str(geometry.get("lod")))
                    for geometry in city_object.get("geometry", [])
                },
            )
            mesh_count += 1
            attribute_count += len(actual)
            mesh_ids.add(object_id)
            if actual:
                attributed_ids.add(object_id)

        self.assertEqual(set(containers), set(expected))
        fallback_count = 0
        for object_id, container in containers.items():
            actual = {
                key: normalize(value)
                for key, value in container.items()
                if key.startswith("attributes.")
            }
            self.assertEqual(
                actual, {} if object_id in mesh_ids else expected[object_id]
            )
            if object_id not in mesh_ids and actual:
                fallback_count += 1
        print(
            f"Verified post-import attributes: {mesh_count} meshes, "
            f"{len(attributed_ids)} CityObjects with attributes, "
            f"{attribute_count} properties, {fallback_count} fallback containers",
            flush=True,
        )
        return mesh_count, attribute_count

    def test_create_mesh_object_and_collection(self) -> None:
        material = bpy.data.materials.new("WallSurface")
        mesh_object = create_mesh_object(
            "triangle",
            [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            [(0, 1, 2)],
            [material],
            [0],
        )
        collection = get_collection("LoD1")
        collection.objects.link(mesh_object)

        self.assertEqual(len(mesh_object.data.vertices), 3)
        self.assertEqual(len(mesh_object.data.polygons), 1)
        self.assertEqual(tuple(mesh_object.data.polygons[0].vertices), (0, 1, 2))
        self.assertIs(mesh_object.data.materials[0], material)
        self.assertIn(mesh_object, collection.objects.values())

    def test_import_cityjson_texture_and_uvs(self) -> None:
        cityjson = {
            "type": "CityJSON",
            "version": "2.0",
            "appearance": {
                "textures": [{"type": "PNG", "image": "textures/facade.png"}],
                "vertices-texture": [[0, 0], [1, 0], [0, 1]],
                "default-theme-texture": "rgbTexture",
            },
            "CityObjects": {
                "building-1": {
                    "type": "Building",
                    "geometry": [
                        {
                            "type": "MultiSurface",
                            "lod": "2",
                            "boundaries": [[[0, 1, 2]]],
                            "semantics": {
                                "surfaces": [{"type": "WallSurface"}],
                                "values": [0],
                            },
                            "texture": {"rgbTexture": {"values": [[[0, 0, 1, 2]]]}},
                        }
                    ],
                }
            },
            "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        }
        png_1x1 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0l"
            "EQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            texture_dir = temp_path / "textures"
            texture_dir.mkdir()
            (texture_dir / "facade.png").write_bytes(base64.b64decode(png_1x1))
            input_path = temp_path / "textured.city.json"
            input_path.write_text(json.dumps(cityjson), encoding="utf-8")

            CityJSONImporter(
                str(input_path),
                material_type="SURFACES",
                reuse_materials=False,
                clear_scene=True,
            ).execute()

        geometry = bpy.data.objects["0: [LoD2] building-1"]
        polygon = geometry.data.polygons[0]
        material = geometry.data.materials[polygon.material_index]
        uv_layer = geometry.data.uv_layers[0]

        self.assertEqual(material["type"], "WallSurface")
        self.assertEqual(material["cityjson_texture_index"], 0)
        self.assertEqual(material["cityjson_texture_image"], "textures/facade.png")
        self.assertIn("facade.png", {image.name for image in bpy.data.images})
        self.assertEqual(
            [tuple(uv.vector) for uv in uv_layer.uv],
            [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        )

    def test_addon_registration_and_operator(self) -> None:
        Up3date.register()
        self._addon_registered = True
        self.assertTrue(hasattr(bpy.types.Scene, "cityjsonfy_properties"))

        menu_calls = []
        menu = SimpleNamespace(
            layout=SimpleNamespace(
                operator=lambda operator, **kwargs: menu_calls.append(
                    (operator, kwargs)
                )
            )
        )
        addon.menu_func_import(menu, bpy.context)
        addon.menu_func_export(menu, bpy.context)
        self.assertEqual(
            menu_calls,
            [
                ("cityjson.import_file", {"text": "CityJSON (.json)"}),
                ("cityjson.export_file", {"text": "CityJSON (.json)"}),
            ],
        )

        mesh = bpy.data.meshes.new("source-mesh")
        source = bpy.data.objects.new("building-1", mesh)
        bpy.context.scene.collection.objects.link(source)
        source.select_set(True)
        bpy.context.view_layer.objects.active = source
        props = bpy.context.scene.cityjsonfy_properties
        props.lod = "2"
        props.feature_type = "Building"
        props.geometry_type = "MultiSurface"

        result = bpy.ops.cityjson.cityjsonfy()

        self.assertEqual(result, {"FINISHED"})
        self.assertEqual(source.name, "2: [LOD2] building-1")
        self.assertEqual(source["type"], "MultiSurface")
        self.assertEqual(source["lod"], "2")
        parent = bpy.data.objects["building-1"]
        self.assertEqual(parent.type, "EMPTY")
        self.assertEqual(parent["type"], "Building")

        Up3date.unregister()
        self._addon_registered = False
        self.assertFalse(hasattr(bpy.types.Scene, "cityjsonfy_properties"))

    def test_import_then_export_cityjson(self) -> None:
        cityjson = {
            "type": "CityJSON",
            "version": "2.0",
            "CityObjects": {
                "building-1": {
                    "type": "Building",
                    "attributes": {
                        "height": 12,
                        "address": {"city": "Delft"},
                        "levels": [1, 2],
                    },
                    "geometry": [
                        {
                            "type": "MultiSurface",
                            "lod": "1",
                            "boundaries": [[[0, 1, 2]]],
                        },
                        {
                            "type": "MultiSurface",
                            "lod": "2",
                            "boundaries": [[[0, 1, 2]]],
                        },
                    ],
                },
                "part-1": {
                    "type": "BuildingPart",
                    "parents": ["building-1"],
                    "attributes": {"height": 3},
                    "geometry": [
                        {
                            "type": "MultiSurface",
                            "lod": "1",
                            "boundaries": [[[0, 1, 2]]],
                        }
                    ],
                },
                "no-geometry": {
                    "type": "Building",
                    "attributes": {"name": "No mesh"},
                },
            },
            "vertices": [[10, 20, 0], [11, 20, 0], [10, 21, 0]],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.city.json"
            output_path = Path(temp_dir) / "output.city.json"
            input_path.write_text(json.dumps(cityjson), encoding="utf-8")

            import_result = addon.ImportCityJSON.execute(
                SimpleNamespace(
                    filepath=str(input_path),
                    material_type="SURFACES",
                    reuse_materials=False,
                    clean_scene=True,
                ),
                bpy.context,
            )

            self.assertEqual(import_result, {"FINISHED"})
            self.assert_imported_attributes(cityjson)
            parent = bpy.data.objects["building-1"]
            geometry = bpy.data.objects["0: [LoD1] building-1"]
            self.assertEqual(parent["type"], "Building")
            self.assertNotIn("attributes.height", parent)
            self.assertEqual(geometry["attributes.height"], 12)
            self.assertIs(geometry.parent, parent)
            self.assertEqual(len(geometry.data.polygons), 1)

            export_result = addon.ExportCityJSON.execute(
                SimpleNamespace(
                    filepath=str(output_path),
                    check_for_duplicates=True,
                    precision=3,
                ),
                bpy.context,
            )
            exported = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(export_result, {"FINISHED"})
        self.assertEqual(exported["CityObjects"]["building-1"]["type"], "Building")
        for object_id, city_object in cityjson["CityObjects"].items():
            self.assertEqual(
                exported["CityObjects"][object_id]["attributes"],
                city_object["attributes"],
            )
        self.assertEqual(len(exported["CityObjects"]["building-1"]["geometry"]), 2)
        self.assertEqual(exported["CityObjects"]["part-1"]["parents"], ["building-1"])
        self.assertEqual(
            exported["CityObjects"]["building-1"]["geometry"][0]["type"],
            "MultiSurface",
        )
        self.assertEqual(len(exported["vertices"]), 3)

    def test_real_datasets_round_trip(self) -> None:
        data_dir = PROJECT_ROOT / "tests" / "data"
        input_paths = sorted(data_dir.glob("*.city.json"))
        external_path = os.environ.get("UP3DATE_TEST_CITYJSON")
        if external_path:
            self.assertTrue(Path(external_path).is_file(), external_path)
            input_paths.append(Path(external_path))

        for input_path in input_paths:
            with self.subTest(dataset=input_path.name):
                print(f"Round-tripping real dataset: {input_path.name}", flush=True)
                remove_scene_objects()
                source = json.loads(input_path.read_text(encoding="utf-8"))
                source_geometry_count = sum(
                    len(city_object.get("geometry", []))
                    for city_object in source["CityObjects"].values()
                )

                import_result = CityJSONImporter(
                    str(input_path),
                    material_type="SURFACES",
                    reuse_materials=True,
                    clear_scene=True,
                ).execute()

                self.assertEqual(import_result, {"FINISHED"})
                mesh_count, attribute_count = self.assert_imported_attributes(source)
                if external_path and input_path == Path(external_path):
                    self.assertGreater(mesh_count, 0)
                    self.assertGreater(attribute_count, 0)
                imported_city_object_ids = {
                    obj.get("cityjson_id", obj.name)
                    for obj in bpy.data.objects
                    if obj.get("cityjson_id", obj.name) in source["CityObjects"]
                }
                self.assertEqual(imported_city_object_ids, set(source["CityObjects"]))

                with tempfile.TemporaryDirectory() as temp_dir:
                    output_path = Path(temp_dir) / input_path.name
                    export_result = CityJSONExporter(
                        str(output_path), check_for_duplicates=True, precision=3
                    ).execute()
                    exported = json.loads(output_path.read_text(encoding="utf-8"))

                exported_geometry_count = sum(
                    len(city_object.get("geometry", []))
                    for city_object in exported["CityObjects"].values()
                )
                self.assertEqual(export_result, {"FINISHED"})
                self.assertEqual(
                    set(exported["CityObjects"]), set(source["CityObjects"])
                )
                self.assertEqual(exported_geometry_count, source_geometry_count)
                for object_id, city_object in source["CityObjects"].items():
                    self.assertEqual(
                        exported["CityObjects"][object_id].get("attributes", {}),
                        city_object.get("attributes", {}),
                        object_id,
                    )
                if "transform" in source:
                    self.assertEqual(exported["transform"], source["transform"])
                for key in ("extensions", "appearance", "geometry-templates"):
                    if source.get(key):
                        self.assertEqual(exported[key], source[key])
                CityJSONDocument.from_dict(exported)

    def test_transformed_solid_with_semantics_round_trip(self) -> None:
        cityjson = {
            "type": "CityJSON",
            "version": "2.0",
            "transform": {
                "scale": [0.1, 0.1, 0.1],
                "translate": [100.0, 200.0, 5.0],
            },
            "CityObjects": {
                "building-1": {
                    "type": "Building",
                    "geometry": [
                        {
                            "type": "Solid",
                            "lod": "2",
                            "boundaries": [[[[0, 1, 2, 3]]]],
                            "semantics": {
                                "surfaces": [{"type": "RoofSurface"}],
                                "values": [[0]],
                            },
                        }
                    ],
                }
            },
            "vertices": [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "solid.city.json"
            output_path = Path(temp_dir) / "solid-output.city.json"
            input_path.write_text(json.dumps(cityjson), encoding="utf-8")

            CityJSONImporter(
                str(input_path),
                material_type="SURFACES",
                reuse_materials=False,
                clear_scene=True,
            ).execute()
            CityJSONExporter(
                str(output_path), check_for_duplicates=True, precision=3
            ).execute()
            exported = json.loads(output_path.read_text(encoding="utf-8"))

        geometry = exported["CityObjects"]["building-1"]["geometry"][0]
        self.assertEqual(exported["transform"], cityjson["transform"])
        self.assertEqual(exported["vertices"], cityjson["vertices"])
        self.assertEqual(geometry["type"], "Solid")
        self.assertEqual(geometry["semantics"]["surfaces"], [{"type": "RoofSurface"}])
        self.assertEqual(geometry["semantics"]["values"], [[0]])


suite = unittest.defaultTestLoader.loadTestsFromTestCase(BlenderIntegrationTests)
result = unittest.TextTestRunner(verbosity=2).run(suite)

if coverage_session is not None:
    coverage_session.stop()
    coverage_session.save()

if not result.wasSuccessful():
    raise RuntimeError("Blender integration tests failed")
