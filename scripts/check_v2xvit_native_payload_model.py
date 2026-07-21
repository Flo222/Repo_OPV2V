#!/usr/bin/env python3
from __future__ import print_function

import argparse
import json

from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    hypes = yaml_utils.load_yaml(args.config, None)

    model = train_utils.create_model(hypes)
    adapter = getattr(model, "payload_transport", None)
    result = {
        "model_class": model.__class__.__name__,
        "has_payload_transport": adapter is not None,
        "payload_interface": (
            adapter.cfg.get("payload", {}).get("interface")
            if adapter is not None else None
        ),
        "transport_mode": (
            adapter.cfg.get("transport_mode")
            if adapter is not None else None
        ),
        "dataset_name": (
            adapter.dataset_name if adapter is not None else None
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if not result["has_payload_transport"]:
        raise SystemExit("payload adapter missing")
    if result["transport_mode"] != "payload_native":
        raise SystemExit("transport_mode is not payload_native")


if __name__ == "__main__":
    main()
