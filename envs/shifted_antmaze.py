from __future__ import annotations

import os
from dataclasses import dataclass

from envs.shift_configs import get_shift_values
from envs.xml_utils import (
    modify_ant_xml_friction,
    modify_floor_friction,
    modify_joint_damping,
    modify_actuator_gear,
    modify_composite_shift,
)
from envs.shifted_maze_factory import make_maze_env


@dataclass
class ShiftedEnvSpec:
    env_name: str
    maze_type: str
    shift_family: str
    shift_level: str
    source_xml_path: str
    generated_xml_path: str
    shift_values: tuple[float, float, float]


def parse_maze_type(env_name: str) -> str:
    # e.g. antmaze-medium-navigate-v0 -> medium
    parts = env_name.split("-")
    if len(parts) < 2:
        raise ValueError(f"Could not parse maze type from env_name: {env_name}")
    return parts[1]


def build_shifted_ant_xml(
    env_name: str,
    source_xml_path: str,
    generated_assets_dir: str,
    shift_family: str,
    shift_level: str,
) -> ShiftedEnvSpec:
    supported = {
        "friction",
        "floor_friction",
        "joint_damping",
        "actuator_gear",
        "composite_shift",
    }
    if shift_family not in supported:
        raise NotImplementedError(
            f"Supported shift families: {sorted(supported)}"
        )

    shift_values = get_shift_values(shift_family, shift_level)
    maze_type = parse_maze_type(env_name)

    safe_env_name = env_name.replace("/", "_")
    out_name = f"{safe_env_name}_{shift_family}_{shift_level}.xml"
    generated_xml_path = os.path.join(generated_assets_dir, out_name)

    if shift_family == "friction":
        modify_ant_xml_friction(
            source_xml_path=source_xml_path,
            output_xml_path=generated_xml_path,
            friction_values=shift_values,
        )
    elif shift_family == "floor_friction":
        modify_floor_friction(
            source_xml_path=source_xml_path,
            output_xml_path=generated_xml_path,
            friction_values=shift_values,
        )
    elif shift_family == "joint_damping":
        modify_joint_damping(
            source_xml_path=source_xml_path,
            output_xml_path=generated_xml_path,
            damping_value=shift_values,
        )
    elif shift_family == "actuator_gear":
        modify_actuator_gear(
            source_xml_path=source_xml_path,
            output_xml_path=generated_xml_path,
            gear_value=shift_values,
        )
    elif shift_family == "composite_shift":
        modify_composite_shift(
            source_xml_path=source_xml_path,
            output_xml_path=generated_xml_path,
            floor_friction_values=shift_values["floor_friction"],
            damping_value=shift_values["joint_damping"],
            gear_value=shift_values["actuator_gear"],
        )

    return ShiftedEnvSpec(
        env_name=env_name,
        maze_type=maze_type,
        shift_family=shift_family,
        shift_level=shift_level,
        source_xml_path=source_xml_path,
        generated_xml_path=generated_xml_path,
        shift_values=shift_values,
    )

def make_shifted_antmaze_env(
    env_name: str,
    source_xml_path: str,
    generated_assets_dir: str,
    shift_family: str,
    shift_level: str,
    render_mode: str = "rgb_array",
    add_noise_to_goal: bool = True,
):
    spec = build_shifted_ant_xml(
        env_name=env_name,
        source_xml_path=source_xml_path,
        generated_assets_dir=generated_assets_dir,
        shift_family=shift_family,
        shift_level=shift_level,
    )

    env = make_maze_env(
        loco_env_type="ant",
        maze_env_type="maze",
        maze_type=spec.maze_type,
        ob_type="states",
        render_mode=render_mode,
        add_noise_to_goal=add_noise_to_goal,
        custom_ant_xml_file=spec.generated_xml_path,
    )

    return env, spec