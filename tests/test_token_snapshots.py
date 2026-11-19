from src.ai.tokens import (
    get_usage_snapshot,
    record_usage,
    reset_usage,
    write_usage_snapshot,
)


def test_usage_snapshot_stays_consistent_after_more_requests():
    reset_usage()
    try:
        record_usage("openai", 10, 2)
        old = get_usage_snapshot()
        record_usage("openai", 20, 3)
        assert old.total_tokens == 12
        assert old.per_provider["openai"].total == 12
        assert get_usage_snapshot().per_provider["openai"].total == 35
    finally:
        reset_usage()


def test_mutating_snapshot_does_not_change_usage_accounting():
    reset_usage()
    try:
        record_usage("openai", 10, 2)
        snapshot = get_usage_snapshot()
        snapshot.per_provider["openai"].input_tokens = 999
        assert get_usage_snapshot().total_input_tokens == 10
    finally:
        reset_usage()


def test_write_usage_snapshot_creates_parent_and_json(tmp_path):
    import json

    reset_usage()
    try:
        record_usage("deepseek", 100, 20)
        target = tmp_path / "nested" / "usage.json"
        write_usage_snapshot(target, get_usage_snapshot())

        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["total_input_tokens"] == 100
        assert payload["total_output_tokens"] == 20
        assert payload["total_tokens"] == 120
        assert payload["per_provider"]["deepseek"]["total_tokens"] == 120
    finally:
        reset_usage()
