# scripts/eval_checkpoint_under_shifts.py

"""
Evaluate a pretrained checkpoint across a family of shifted AntMaze environments.

V1 goal:
- define the evaluation loop
- define which shift levels are tested
- print intended experiment structure

TODO:
- wire in shifted env construction to actual OGBench env creation
"""

from __future__ import annotations

SHIFT_LEVELS = [
    "base",
    "mild_low",
    "moderate_low",
    "severe_low",
    "mild_high",
    "moderate_high",
    "severe_high",
]


def main():
    print("Planned evaluation across shifts:")
    for level in SHIFT_LEVELS:
        print(f"  - {level}")


if __name__ == "__main__":
    main()