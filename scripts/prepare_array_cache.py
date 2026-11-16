#!/usr/bin/env python3
"""Prepare and verify a float64 beta cache from the two public signal matrices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from atomic_io import prepare_cache_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-signals", required=True, type=Path)
    parser.add_argument("--external-signals", required=True, type=Path)
    parser.add_argument("--discovery-metadata", required=True, type=Path)
    parser.add_argument("--external-metadata", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--protocol", type=Path,
                        default=Path(__file__).resolve().parents[1] / "config/classification_protocol.json")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    verification = prepare_cache_atomic(
        args.discovery_signals, args.external_signals,
        args.discovery_metadata, args.external_metadata,
        args.cache_dir, protocol, args.protocol)
    print(json.dumps({"stage": "cache_verified", "cache_dir": str(args.cache_dir),
                      **verification}), flush=True)


if __name__ == "__main__":
    main()
