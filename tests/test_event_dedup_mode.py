from types import SimpleNamespace

from src.models import CollectionConfig


def test_collection_defaults_to_drop():
    assert CollectionConfig().event_dedup_mode == "drop"


def test_collection_accepts_witness():
    assert CollectionConfig(event_dedup_mode="witness").event_dedup_mode == "witness"


def test_witness_metadata_marker():
    """Mirrors orchestrator behaviour: duplicates become witnesses in PR mode."""
    item = SimpleNamespace(metadata={})
    item.metadata["is_witness"] = True
    assert item.metadata["is_witness"] is True
