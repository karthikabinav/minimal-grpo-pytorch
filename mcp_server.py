#!/usr/bin/env python3
"""MCP server wrapper for the educational GRPO project.

Transport: streamable-http (default) so it runs cleanly in Docker.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "grpo_single_file.py"

mcp = FastMCP("minimal-grpo-pytorch")


@mcp.tool()
def run_training(
    updates: int = 100,
    tasks_per_update: int = 32,
    group_size: int = 8,
    max_steps: int = 8,
    lr: float = 3e-4,
) -> str:
    """Run the GRPO training script and return stdout."""
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--updates",
        str(updates),
        "--tasks-per-update",
        str(tasks_per_update),
        "--group-size",
        str(group_size),
        "--max-steps",
        str(max_steps),
        "--lr",
        str(lr),
    ]
    out = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    if out.returncode != 0:
        return f"ERROR\n{out.stderr}\n{out.stdout}"
    return out.stdout


@mcp.tool()
def project_summary() -> str:
    """Quick summary of what this project implements."""
    return (
        "Single-file educational GRPO in PyTorch. "
        "Includes toy multi-step environment, grouped rollouts, relative advantages, "
        "PPO clipping, and reference KL penalty."
    )


if __name__ == "__main__":
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8000"))
    mcp.run(transport="streamable-http", host=host, port=port)
