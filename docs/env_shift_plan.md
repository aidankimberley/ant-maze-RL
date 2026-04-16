# Environment Shift Plan

## Goal

Build a reusable pipeline for controlled AntMaze dynamics shifts.

## Initial scope

- Environments:
  - antmaze-medium-navigate-v0
  - antmaze-large-navigate-v0
- Shift family:
  - friction only
- Severity levels:
  - mild / moderate / severe
  - low-friction and high-friction sides

## Deliverables

1. Generate shifted XMLs cleanly
2. Validate shifts are correctly applied
3. Evaluate pretrained checkpoint under shifts
4. Hand off shifted env interface for online finetuning
