"""Machine-readable outputs: versioned JSON digest and per-run report."""

from .json_digest import (
    SCHEMA_VERSION,
    build_digest_payload,
    item_to_dict,
    write_digest_json,
    write_run_report,
)

__all__ = [
    "SCHEMA_VERSION",
    "build_digest_payload",
    "item_to_dict",
    "write_digest_json",
    "write_run_report",
]
