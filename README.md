# minimal-grpo-pytorch

Minimal educational implementation of agentic RL with GRPO in a **single file** using PyTorch only.

## File
- `grpo_single_file.py`

## What it shows
- Multi-step agent behavior in a toy environment (LineWorld)
- Grouped rollouts per task
- Relative advantages within each group (GRPO-style)
- PPO-style clipping objective
- KL penalty to frozen reference policy

## Run
```bash
python3 grpo_single_file.py
```

## Notes
- This is intentionally tiny and didactic.
- No frameworks, no logging stack, no dataset loaders.
- Great for reading line-by-line and modifying.
