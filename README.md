# minimal-grpo-pytorch

Minimal educational implementation of agentic RL with GRPO in a **single file** using PyTorch.

## Core file
- `grpo_single_file.py`

## Run directly
```bash
python3 grpo_single_file.py
```

## Real MCP + Docker setup
This repo now includes a real MCP server wrapper:
- `mcp_server.py` (FastMCP server)
- `Dockerfile`
- `docker-compose.yml`

### 1) Build + run MCP server
```bash
docker compose up --build
```

Server runs on:
- `http://localhost:8000/mcp`

### 2) Available MCP tools
- `run_training(updates, tasks_per_update, group_size, max_steps, lr)`
- `project_summary()`

### Notes
- Educational and intentionally minimal.
- No logging stack, no trainer framework, no external RL libs.
