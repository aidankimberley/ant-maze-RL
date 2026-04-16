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
    """
    Modify the default geom friction in ant.xml.
    This affects many geoms that inherit from <default><geom>.
    """
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


def modify_floor_friction(
    source_xml_path: str,
    output_xml_path: str,
    friction_values: tuple[float, float, float],
) -> str:
    """
    Modify only the floor geom friction in ant.xml.
    This is a cleaner terrain-shift intervention than changing the default geom friction.
    """
    tree = ET.parse(source_xml_path)
    root = tree.getroot()

    floor_elem = root.find('.//geom[@name="floor"]')
    if floor_elem is None:
        raise ValueError('Could not find floor geom with name="floor".')

    floor_elem.set("friction", format_friction(friction_values))

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


def read_floor_friction(xml_path: str) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    floor_elem = root.find('.//geom[@name="floor"]')
    if floor_elem is None:
        raise ValueError('Could not find floor geom with name="floor".')

    friction = floor_elem.get("friction")
    if friction is not None:
        return friction

    # Fall back to default geom friction if floor has no explicit friction
    default_elem = root.find("default")
    if default_elem is None:
        raise ValueError("Could not find <default> section in XML.")

    geom_elem = default_elem.find("geom")
    if geom_elem is None:
        raise ValueError("Could not find <geom> under <default> in XML.")

    default_friction = geom_elem.get("friction")
    if default_friction is None:
        raise ValueError("No friction attribute found on floor geom or default geom.")

    return default_friction

def modify_joint_damping(
    source_xml_path: str,
    output_xml_path: str,
    damping_value: float,
) -> str:
    tree = ET.parse(source_xml_path)
    root = tree.getroot()

    default_elem = root.find("default")
    if default_elem is None:
        raise ValueError("Could not find <default> section in XML.")

    joint_elem = default_elem.find("joint")
    if joint_elem is None:
        raise ValueError("Could not find <joint> under <default> in XML.")

    joint_elem.set("damping", str(damping_value))

    os.makedirs(os.path.dirname(output_xml_path), exist_ok=True)
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=False)
    return output_xml_path


def modify_actuator_gear(
    source_xml_path: str,
    output_xml_path: str,
    gear_value: float,
) -> str:
    tree = ET.parse(source_xml_path)
    root = tree.getroot()

    actuator_elem = root.find("actuator")
    if actuator_elem is None:
        raise ValueError("Could not find <actuator> section in XML.")

    motors = actuator_elem.findall("motor")
    if not motors:
        raise ValueError("Could not find any <motor> elements in <actuator>.")

    for motor in motors:
        motor.set("gear", str(gear_value))

    os.makedirs(os.path.dirname(output_xml_path), exist_ok=True)
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=False)
    return output_xml_path


def modify_composite_shift(
    source_xml_path: str,
    output_xml_path: str,
    floor_friction_values: tuple[float, float, float],
    damping_value: float,
    gear_value: float,
) -> str:
    tree = ET.parse(source_xml_path)
    root = tree.getroot()

    # floor friction
    floor_elem = root.find('.//geom[@name="floor"]')
    if floor_elem is None:
        raise ValueError('Could not find floor geom with name="floor".')
    floor_elem.set("friction", format_friction(floor_friction_values))

    # default joint damping
    default_elem = root.find("default")
    if default_elem is None:
        raise ValueError("Could not find <default> section in XML.")

    joint_elem = default_elem.find("joint")
    if joint_elem is None:
        raise ValueError("Could not find <joint> under <default> in XML.")
    joint_elem.set("damping", str(damping_value))

    # actuator gear
    actuator_elem = root.find("actuator")
    if actuator_elem is None:
        raise ValueError("Could not find <actuator> section in XML.")

    motors = actuator_elem.findall("motor")
    if not motors:
        raise ValueError("Could not find any <motor> elements in <actuator>.")

    for motor in motors:
        motor.set("gear", str(gear_value))

    os.makedirs(os.path.dirname(output_xml_path), exist_ok=True)
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=False)
    return output_xml_path


def read_joint_damping(xml_path: str) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    default_elem = root.find("default")
    if default_elem is None:
        raise ValueError("Could not find <default> section in XML.")

    joint_elem = default_elem.find("joint")
    if joint_elem is None:
        raise ValueError("Could not find <joint> under <default> in XML.")

    damping = joint_elem.get("damping")
    if damping is None:
        raise ValueError("No damping attribute found on default joint.")
    return damping


def read_actuator_gear(xml_path: str) -> list[str]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    actuator_elem = root.find("actuator")
    if actuator_elem is None:
        raise ValueError("Could not find <actuator> section in XML.")

    motors = actuator_elem.findall("motor")
    if not motors:
        raise ValueError("Could not find any <motor> elements in <actuator>.")

    gears = []
    for motor in motors:
        gear = motor.get("gear")
        if gear is None:
            raise ValueError("A motor is missing its gear attribute.")
        gears.append(gear)
    return gears