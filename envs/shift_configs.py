# envs/shift_configs.py

SHIFT_CONFIGS = {
    "friction": {
        "base": (1.0, 0.5, 0.5),
        "mild_low": (0.8, 0.4, 0.4),
        "moderate_low": (0.6, 0.3, 0.3),
        "severe_low": (0.4, 0.2, 0.2),
        "mild_high": (1.2, 0.6, 0.6),
        "moderate_high": (1.4, 0.7, 0.7),
        "severe_high": (1.8, 0.9, 0.9),
    },

    "floor_friction": {
        "base": (1.0, 0.5, 0.5),
        "mild_low": (0.6, 0.3, 0.3),
        "moderate_low": (0.3, 0.15, 0.15),
        "severe_low": (0.15, 0.075, 0.075),
        "mild_high": (1.5, 0.75, 0.75),
        "moderate_high": (2.5, 1.25, 1.25),
        "severe_high": (4.0, 2.0, 2.0),
    },

    "joint_damping": {
        "base": 1.0,
        "mild_low": 0.7,
        "moderate_low": 0.4,
        "severe_low": 0.2,
        "mild_high": 1.5,
        "moderate_high": 2.5,
        "severe_high": 4.0,
    },

    "actuator_gear": {
        "base": 30.0,
        "mild_low": 24.0,
        "moderate_low": 18.0,
        "severe_low": 12.0,
        "mild_high": 36.0,
        "moderate_high": 45.0,
        "severe_high": 60.0,
    },

"composite_shift": {
    "base": {
        "floor_friction": (1.0, 0.5, 0.5),
        "joint_damping": 1.0,
        "actuator_gear": 30.0,
    },
    "mild": {
        "floor_friction": (1.5, 0.75, 0.75),
        "joint_damping": 2.2,
        "actuator_gear": 24.0,
    },
    "moderate": {
        "floor_friction": (1.8, 0.9, 0.9),
        "joint_damping": 2.8,
        "actuator_gear": 22.0,
    },
    "severe": {
        "floor_friction": (2.0, 1.0, 1.0),
        "joint_damping": 3.0,
        "actuator_gear": 21.0,
    },
}
}


def get_shift_values(shift_family: str, shift_level: str):
    if shift_family not in SHIFT_CONFIGS:
        raise ValueError(f"Unknown shift family: {shift_family}")
    if shift_level not in SHIFT_CONFIGS[shift_family]:
        raise ValueError(f"Unknown shift level '{shift_level}' for family '{shift_family}'")
    return SHIFT_CONFIGS[shift_family][shift_level]