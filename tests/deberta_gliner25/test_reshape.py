"""Pioneer response-key mapping for the boundary plugin."""

from plugins.deberta_gliner25.processor import reshape_boundary_output


def test_reshape_maps_entities_classifications_structures_relations():
    sample = {
        "entities": {"person": [{"text": "Ada", "start": 0, "end": 3}]},
        "sentiment": "positive",
        "employee": [{"name": "Ada", "title": "VP"}],
        "works_at": [{"head": "Ada", "tail": "NVIDIA"}],
    }
    out = reshape_boundary_output(sample)
    assert out["entities"]["person"][0]["text"] == "Ada"
    assert out["classifications"]["sentiment"] == "positive"
    assert out["structures"]["employee"][0]["name"] == "Ada"
    assert out["relations"]["works_at"][0]["head"] == "Ada"


def test_reshape_merges_list_of_entity_dicts():
    sample = {
        "entities": [{"person": [{"text": "Ada"}]}, {"org": [{"text": "NVIDIA"}]}],
    }
    out = reshape_boundary_output(sample)
    assert "person" in out["entities"]
    assert "org" in out["entities"]


def test_reshape_empty_sample():
    out = reshape_boundary_output({})
    assert out == {
        "entities": {},
        "classifications": {},
        "structures": {},
        "relations": {},
    }
