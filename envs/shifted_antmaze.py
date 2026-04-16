# envs/shifted_antmaze.py

from __future__ import annotations

import os
from dataclasses import dataclass

from envs.shift_configs import get_shift_values
from envs.xml_utils import modify_ant_xml_friction


@dataclass
class ShiftedEnvSpec:
    env_name: str
    shift_family: str
    shift_level: str
    source_xml_path: str
    generated_xml_path: str
    shift_values: tuple[float, float, float]


def build_shifted_ant_xml(
    env_name: str,
    source_xml_path: str,
    generated_assets_dir: str,
    shift_family: str,
    shift_level: str,
) -> ShiftedEnvSpec:
    if shift_family != "friction":
        raise NotImplementedError("Only friction shifts are supported in v1.")

    shift_values = get_shift_values(shift_family, shift_level)

    safe_env_name = env_name.replace("/", "_")
    out_name = f"{safe_env_name}_{shift_family}_{shift_level}.xml"
    generated_xml_path = os.path.join(generated_assets_dir, out_name)

    modify_ant_xml_friction(
        source_xml_path=source_xml_path,
        output_xml_path=generated_xml_path,
        friction_values=shift_values,
    )

    return ShiftedEnvSpec(
        env_name=env_name,
        shift_family=shift_family,
        shift_level=shift_level,
        source_xml_path=source_xml_path,
        generated_xml_path=generated_xml_path,
        shift_values=shift_values,
    )