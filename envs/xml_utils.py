# envs/xml_utils.py

from __future__ import annotations

import os
import xml.etree.ElementTree as ET


def format_friction(values: tuple[float, float, float]) -> str:
    return " ".join(str(v) for v in values)


def modify_ant_xml_friction(
    source_xml_path: str,
    output_xml_path: str,
    friction_values: tuple[float, float, float],
) -> str:
    tree = ET.parse(source_xml_path)
    root = tree.getroot()

    default_elem = root.find("default")
    if default_elem is None:
        raise ValueError("Could not find <default> section in XML.")

    geom_elem = default_elem.find("geom")
    if geom_elem is None:
        raise ValueError("Could not find <geom> under <default> in XML.")

    geom_elem.set("friction", format_friction(friction_values))

    os.makedirs(os.path.dirname(output_xml_path), exist_ok=True)
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=False)
    return output_xml_path


def read_default_geom_friction(xml_path: str) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    default_elem = root.find("default")
    if default_elem is None:
        raise ValueError("Could not find <default> section in XML.")

    geom_elem = default_elem.find("geom")
    if geom_elem is None:
        raise ValueError("Could not find <geom> under <default> in XML.")

    friction = geom_elem.get("friction")
    if friction is None:
        raise ValueError("No friction attribute found in default geom.")

    return friction