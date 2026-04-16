# envs/shifted_maze_factory.py

from __future__ import annotations

from ogbench.locomaze.ant import AntEnv
from ogbench.locomaze.maze import make_maze_env as og_make_maze_env


def make_maze_env(
    loco_env_type,
    maze_env_type,
    *args,
    custom_ant_xml_file=None,
    **kwargs,
):
    """
    Thin wrapper around OGBench's make_maze_env.

    If loco_env_type == 'ant' and custom_ant_xml_file is provided,
    temporarily override AntEnv.xml_file so maze.py builds from the
    shifted XML instead of the default one.
    """
    if loco_env_type != "ant" or custom_ant_xml_file is None:
        return og_make_maze_env(loco_env_type, maze_env_type, *args, **kwargs)

    original_xml_file = AntEnv.xml_file
    try:
        AntEnv.xml_file = custom_ant_xml_file
        env = og_make_maze_env(loco_env_type, maze_env_type, *args, **kwargs)
    finally:
        AntEnv.xml_file = original_xml_file

    return env