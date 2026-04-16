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
    }
}


def get_shift_values(shift_family: str, shift_level: str) -> tuple[float, float, float]:
    if shift_family not in SHIFT_CONFIGS:
        raise ValueError(f"Unknown shift family: {shift_family}")
    if shift_level not in SHIFT_CONFIGS[shift_family]:
        raise ValueError(f"Unknown shift level '{shift_level}' for family '{shift_family}'")
    return SHIFT_CONFIGS[shift_family][shift_level]