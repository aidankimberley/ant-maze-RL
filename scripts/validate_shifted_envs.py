# scripts/validate_shifted_envs.py

from __future__ import annotations

import argparse
import os

from envs.shifted_antmaze import build_shifted_ant_xml
from envs.xml_utils import read_default_geom_friction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_xml",
        type=str,
        required=True,
        help="Path to base ant.xml",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="antmaze-medium-navigate-v0",
    )
    parser.add_argument(
        "--generated_assets_dir",
        type=str,
        default="generated_assets",
    )
    parser.add_argument(
        "--shift_level",
        type=str,
        default="moderate_low",
    )
    args = parser.parse_args()

    print(f"Base XML: {args.source_xml}")
    print(f"Base default geom friction: {read_default_geom_friction(args.source_xml)}")

    spec = build_shifted_ant_xml(
        env_name=args.env_name,
        source_xml_path=args.source_xml,
        generated_assets_dir=args.generated_assets_dir,
        shift_family="friction",
        shift_level=args.shift_level,
    )

    print("\nGenerated shifted XML:")
    print(spec.generated_xml_path)
    print(f"Shift values: {spec.shift_values}")
    print(f"Generated default geom friction: {read_default_geom_friction(spec.generated_xml_path)}")


if __name__ == "__main__":
    main()