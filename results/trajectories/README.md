# Trajectory Logging Structure

This directory contains the trajectory logging structure for SWE-Bench evaluation.

## Structure

```
trajectories/
├── instance_id_1/
│   ├── trajectory.jsonl
│   ├── llm_interactions.jsonl
│   ├── tool_calls.jsonl
│   └── performance_metrics.json
├── instance_id_2/
│   └── ...
└── ...
```

## Format

Each trajectory file contains:
- **trajectory.jsonl**: Main execution steps
- **llm_interactions.jsonl**: All LLM inputs/outputs
- **tool_calls.jsonl**: Tool usage and results
- **performance_metrics.json**: Timing and resource usage

## Example trajectory.jsonl entry:

```json
{
  "step": 1,
  "timestamp": "2025-01-27T10:30:00Z",
  "action": "search_files",
  "input": {"query": "Field.__hash__", "file_pattern": "*.py"},
  "output": {"files_found": ["models/fields.py"], "matches": 5},
  "performance": {"duration_ms": 150, "memory_mb": 45.2},
  "metadata": {"step_type": "search", "total_steps": 25}
}
```

## Environment Variable

Trajectories are written to the path specified in `TRAJ_DIR` environment variable during evaluation.
