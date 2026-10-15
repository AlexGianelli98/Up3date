"""Tests for CityJSON importer/exporter orchestration at the Blender boundary."""

import json
from types import SimpleNamespace

import bpy
import idprop
import pytest

from Up3date.blender import exporter, importer
from Up3date.blender.material import (
    BasicMaterialFactory,
    CityObjectTypeMaterialFactory,
    ReuseMaterialFactory,
)
from Up3date.models.city_objects import Building, BuildingPart, GenericCityObject
from Up3date.models.cityjson import CityJSONDocument, Transform
from Up3date.models.geomprimitives import MultiSurface


class FakeBlenderObject(dict):
    def __init__(self, name="object", **properties):
        super().__init__(properties)
        self.name = name
        self.type = "MESH"
        self.parent = None
        self.data = SimpleNamespace(materials=[], vertices=[], polygons=[])


@pytest.fixture(autouse=True)
def blender_state():
    bpy.context.scene = SimpleNamespace(world={})


@pytest.mark.parametrize(
    ("material_type", "reuse", "expected_factory"),
    [
        ("SURFACES", True, ReuseMaterialFactory),
        ("SURFACES", False, BasicMaterialFactory),
        ("CITY_OBJECTS", True, CityObjectTypeMaterialFactory),
    ],
)
def test_parser_selects_material_factory(material_type, reuse, expected_factory):
    parser = importer.CityJSONImporter("city.json", material_type, reuse)

    assert isinstance(parser.material_factory, expected_factory)


def test_prepare_vertices_translates_untransformed_document_to_origin():
    parser = importer.CityJSONImporter("city.json", "CITY_OBJECTS")
    parser.document = CityJSONDocument(vertices=[[5, 4, 3], [8, 10, 12]])

    parser.prepare_vertices()

    assert parser.vertices == ((0.0, 0.0, 0.0), (3.0, 6.0, 9.0))
    assert bpy.context.scene.world == {
        "Axis_Origin_X_translation": -5.0,
        "Axis_Origin_Y_translation": -4.0,
        "Axis_Origin_Z_translation": -3.0,
    }


def test_prepare_vertices_applies_cityjson_transform():
    parser = importer.CityJSONImporter("city.json", "CITY_OBJECTS")
    parser.document = CityJSONDocument(
        transform=Transform(scale=[2, 3, 4], translate=[10, 20, 30]),
        vertices=[[1, 2, 3], [2, 4, 5]],
    )

    parser.prepare_vertices()

    assert parser.vertices == ((0, 0, 0), (2, 6, 8))
    assert bpy.context.scene.world["transformed"] is True
    assert bpy.context.scene.world["transform.Y_scale"] == 3
    assert bpy.context.scene.world["transform.Z_translate"] == 30


def test_prepare_vertices_reuses_existing_axis_translation():
    bpy.context.scene.world.update(
        {
            "Axis_Origin_X_translation": -10,
            "Axis_Origin_Y_translation": -20,
            "Axis_Origin_Z_translation": -30,
        }
    )
    parser = importer.CityJSONImporter("city.json", "CITY_OBJECTS")
    parser.document = CityJSONDocument(vertices=[[12, 23, 34]])

    parser.prepare_vertices()

    assert parser.vertices == ((2.0, 3.0, 4.0),)


def test_prepare_vertices_requires_loaded_document():
    parser = importer.CityJSONImporter("city.json", "CITY_OBJECTS")

    with pytest.raises(RuntimeError, match="must be loaded"):
        parser.prepare_vertices()


def test_load_data_deserializes_cityjson(tmp_path):
    input_path = tmp_path / "minimal.city.json"
    input_path.write_text(
        json.dumps(
            {"type": "CityJSON", "version": "2.0", "CityObjects": {}, "vertices": []}
        )
    )
    parser = importer.CityJSONImporter(str(input_path), "CITY_OBJECTS")

    parser.load_data()

    assert parser.data["version"] == "2.0"
    assert parser.document is not None
    assert parser.document.city_objects == {}


def test_parse_geometry_builds_mesh_and_properties(monkeypatch):
    parser = importer.CityJSONImporter("city.json", "CITY_OBJECTS")
    parser.vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    parser.material_factory = SimpleNamespace(
        get_materials=lambda **_kwargs: (["material"], [0])
    )
    created = FakeBlenderObject()
    captured = {}

    def create_mesh_object(
        name, vertices, faces, materials, material_indices, uvs_by_face=None
    ):
        captured.update(
            name=name,
            vertices=vertices,
            faces=faces,
            materials=materials,
            material_indices=material_indices,
            uvs_by_face=uvs_by_face,
        )
        return created

    monkeypatch.setattr(importer, "create_mesh_object", create_mesh_object)
    geometry = MultiSurface(lod="2", boundaries=[[[0, 1, 2]]])

    result = parser.parse_geometry(
        "building-1", SimpleNamespace(type="Building"), geometry, 0
    )

    assert result is created
    assert captured == {
        "name": "0: [LoD2] building-1",
        "vertices": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        "faces": [(0, 1, 2)],
        "materials": ["material"],
        "material_indices": [0],
        "uvs_by_face": None,
    }
    assert created == {"lod": "2", "type": "MultiSurface"}


@pytest.mark.parametrize(
    "geometry_types", [[], ["EMPTY"], ["MESH"], ["MESH", "MESH"], ["EMPTY", "MESH"]]
)
@pytest.mark.parametrize("with_attributes", [False, True])
def test_importer_assigns_attributes_to_each_mesh(
    monkeypatch, geometry_types, with_attributes
):
    attributes = {"height": 12, "address": {"city": "Delft"}} if with_attributes else {}
    expected = (
        {"attributes.height": 12, "attributes.address.city": "Delft"}
        if with_attributes
        else {}
    )
    parser = importer.CityJSONImporter("city.json", "CITY_OBJECTS", clear_scene=False)
    parser.document = CityJSONDocument(
        city_objects={
            "building-1": Building(
                attributes=attributes,
                geometry=[
                    MultiSurface(lod=str(index)) for index in range(len(geometry_types))
                ],
            )
        }
    )
    parent = FakeBlenderObject("building-1")
    parent.type = "EMPTY"
    geometries = []
    linked = []
    collection = SimpleNamespace(objects=SimpleNamespace(link=linked.append))
    bpy.context.scene.collection = collection

    def parse_geometry(objid, _obj, geometry, index):
        result = FakeBlenderObject(
            f"geometry-{index}", type=geometry.type, lod=geometry.lod
        )
        result.type = geometry_types[index]
        geometries.append(result)
        return result

    monkeypatch.setattr(parser, "load_data", lambda: None)
    monkeypatch.setattr(parser, "prepare_vertices", lambda: None)
    monkeypatch.setattr(parser, "parse_geometry", parse_geometry)
    monkeypatch.setattr(importer, "create_empty_object", lambda _name: parent)
    monkeypatch.setattr(importer, "get_collection", lambda _name: collection)

    assert parser.execute() == {"FINISHED"}
    assert parent["type"] == "Building"
    assert parent["cityjson_id"] == "building-1"
    assert linked == [parent, *geometries]
    parent_attributes = {
        key: value for key, value in parent.items() if key.startswith("attributes.")
    }
    assert parent_attributes == ({} if "MESH" in geometry_types else expected)
    for index, geometry in enumerate(geometries):
        assert geometry.parent is parent
        assert geometry["cityjson_id"] == "building-1"
        assert geometry["type"] == "MultiSurface"
        assert geometry["lod"] == str(index)
        actual = {
            key: value
            for key, value in geometry.items()
            if key.startswith("attributes.")
        }
        assert actual == (expected if geometry.type == "MESH" else {})


def test_exporter_creates_typed_city_object_from_custom_properties():
    city_object = FakeBlenderObject(
        type="Building",
        **{
            "attributes.height": 12,
            "attributes.address.city": "Delft",
            "children": idprop.types.IDPropertyArray(["part-1"]),
            "_RNA_UI": "ignored",
        },
    )
    doc = CityJSONDocument()

    exporter.CityJSONExporter.get_custom_properties(city_object, doc, "building-1")

    result = doc.city_objects["building-1"]
    assert isinstance(result, Building)
    assert result.attributes == {"height": 12, "address": {"city": "Delft"}}
    assert result.children == ["part-1"]


def test_exporter_preserves_existing_geometry_and_relationships():
    geometry = MultiSurface(lod="1")
    doc = CityJSONDocument(
        city_objects={"building-1": GenericCityObject(geometry=[geometry])}
    )
    city_object = FakeBlenderObject(
        type="Building",
        parents=["parent"],
        geographicalExtent=[0, 0, 0, 1, 1, 1],
    )

    exporter.CityJSONExporter.get_custom_properties(city_object, doc, "building-1")

    result = doc.city_objects["building-1"]
    assert isinstance(result, Building)
    assert result.geometry == [geometry]
    assert result.parents == ["parent"]
    assert result.geographical_extent == [0, 0, 0, 1, 1, 1]


@pytest.mark.parametrize("legacy", [{}, {"height": 1, "stale": "ignored"}])
def test_export_mesh_attributes_reads_copies_and_preserves_structure(legacy):
    geometry = MultiSurface(lod="1")
    doc = CityJSONDocument(
        city_objects={
            "building-1": Building(attributes=legacy, geometry=[geometry]),
            "part-1": BuildingPart(parents=["building-1"]),
        }
    )
    meshes = {
        "building-1": [
            FakeBlenderObject(
                "first",
                type="MultiSurface",
                lod="1",
                **{
                    "attributes.height": 12,
                    "attributes.address.city": "Delft",
                    "attributes.values": idprop.types.IDPropertyArray([1, 2]),
                },
            ),
            FakeBlenderObject(
                "second",
                **{
                    "attributes.height": 12,
                    "attributes.address.city": "Delft",
                    "attributes.values": [1, 2],
                },
            ),
        ],
        "part-1": [FakeBlenderObject("part", **{"attributes.height": 3})],
    }

    exporter.CityJSONExporter.export_mesh_attributes(meshes, doc)

    building = doc.city_objects["building-1"]
    assert isinstance(building, Building)
    assert building.geometry == [geometry]
    assert building.attributes == {
        "height": 12,
        "address": {"city": "Delft"},
        "values": [1, 2],
    }
    assert doc.city_objects["part-1"].attributes == {"height": 3}
    assert doc.city_objects["part-1"].parents == ["building-1"]


@pytest.mark.parametrize("mesh_count", [0, 1, 2])
def test_export_mesh_attributes_preserves_legacy_without_mesh_attributes(mesh_count):
    doc = CityJSONDocument(
        city_objects={"building-1": Building(attributes={"height": 12})}
    )
    meshes = [FakeBlenderObject(f"mesh-{index}") for index in range(mesh_count)]

    exporter.CityJSONExporter.export_mesh_attributes({"building-1": meshes}, doc)

    assert doc.city_objects["building-1"].attributes == {"height": 12}


@pytest.mark.parametrize("mesh_first", [False, True])
def test_exporter_reads_mesh_attributes_regardless_of_object_order(
    monkeypatch, tmp_path, mesh_first
):
    parent = FakeBlenderObject(
        "building-1", type="Building", **{"attributes.height": 1}
    )
    parent.type = "EMPTY"
    mesh = FakeBlenderObject(
        "renamed geometry",
        type="MultiSurface",
        lod="1",
        cityjson_id="building-1",
        **{"attributes.height": 12},
    )
    mesh.parent = parent
    monkeypatch.setattr(
        bpy.data,
        "objects",
        [mesh, parent] if mesh_first else [parent, mesh],
        raising=False,
    )
    monkeypatch.setattr(exporter, "export_metadata", lambda _doc: None)
    output_path = tmp_path / "output.city.json"

    result = exporter.CityJSONExporter(
        str(output_path), check_for_duplicates=False
    ).execute()

    assert result == {"FINISHED"}
    building = json.loads(output_path.read_text())["CityObjects"]["building-1"]
    assert building["type"] == "Building"
    assert building["attributes"] == {"height": 12}
    assert len(building["geometry"]) == 1


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"attributes.height": 12}, {"attributes.height": 13}),
        ({"attributes.height": 12}, {}),
        ({}, {"attributes.height": 12}),
        ({"attributes.height": None}, {"attributes.height": 12}),
        ({"attributes.height": None}, {}),
        ({}, {"attributes.height": None}),
    ],
)
def test_exporter_rejects_attribute_conflicts_without_overwriting(
    monkeypatch, tmp_path, first, second
):
    meshes = [
        FakeBlenderObject(
            name,
            type="MultiSurface",
            lod="1",
            cityjson_id="building-1",
            **properties,
        )
        for name, properties in [("first", first), ("second", second)]
    ]
    monkeypatch.setattr(bpy.data, "objects", meshes, raising=False)
    output_path = tmp_path / "existing.city.json"
    output_path.write_text("original")

    with pytest.raises(
        ValueError, match=r"attributes.height.*building-1.*first.*second"
    ):
        exporter.CityJSONExporter(str(output_path)).execute()

    assert output_path.read_text() == "original"


def test_create_mesh_structure_adds_geometry_and_semantics():
    city_object = FakeBlenderObject(name="mesh", lod="1", type="MultiSurface")
    city_object.data.materials = ["wall"]
    doc = CityJSONDocument()

    city_object_id, vertices, polygons = (
        exporter.CityJSONExporter.create_mesh_structure(
            city_object, "0: [LoD1] building-1", doc
        )
    )

    assert city_object_id == "building-1"
    assert vertices == []
    assert polygons == []
    model = doc.city_objects[city_object_id]
    assert isinstance(model, GenericCityObject)
    assert model.geometry[0].lod == "1"
    assert model.geometry[0].semantics is not None


@pytest.mark.parametrize(
    "properties",
    [
        {"type": "MultiSurface"},
        {"lod": "1", "type": "MultiSolid"},
    ],
)
def test_create_mesh_structure_rejects_invalid_properties(properties):
    city_object = FakeBlenderObject(name="invalid", **properties)

    with pytest.raises(SystemExit):
        exporter.CityJSONExporter.create_mesh_structure(
            city_object, "0: [LoD1] building-1", CityJSONDocument()
        )


def test_export_geometry_and_semantics_writes_multi_surface(monkeypatch):
    geometry = MultiSurface(lod="1")
    doc = CityJSONDocument(
        city_objects={"building-1": GenericCityObject(geometry=[geometry])}
    )
    city_object = FakeBlenderObject(type="MultiSurface")
    city_object.matrix_world = SimpleNamespace()
    vertices = [SimpleNamespace(co=(0, 0, 0)), SimpleNamespace(co=(1, 0, 0))]
    faces = [SimpleNamespace(index=0, vertices=[0, 1], material_index=0)]
    linked_faces = []

    monkeypatch.setattr(exporter, "store_semantic_surfaces", lambda *_args: {})
    monkeypatch.setattr(
        exporter,
        "write_vertices_to_cityjson",
        lambda _obj, vertex, target: target.vertices.append(list(vertex)),
    )
    monkeypatch.setattr(
        exporter,
        "link_face_semantic_surface",
        lambda *_args: linked_faces.append(_args[-1]),
    )

    next_index = exporter.CityJSONExporter.export_geometry_and_semantics(
        city_object, doc, "building-1", faces, vertices, {}, 0
    )

    assert next_index == 2
    assert geometry.boundaries == [[[0, 1]]]
    assert doc.vertices == [[0, 0, 0], [1, 0, 0]]
    assert linked_faces == faces


def test_initialize_document_returns_new_documents():
    first = exporter.CityJSONExporter.initialize_document()
    second = exporter.CityJSONExporter.initialize_document()

    assert isinstance(first, CityJSONDocument)
    assert first is not second
