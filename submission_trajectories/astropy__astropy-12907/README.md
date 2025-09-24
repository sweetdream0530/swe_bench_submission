# Trajectory for astropy__astropy-12907

## Overview
- **Instance ID**: astropy__astropy-12907
- **Total Steps**: 23
- **Sessions**: 15
- **Status**: ✅ Success

## Files
- `trajectory.jsonl`: Complete trajectory with all model I/O and decisions
- `summary.json`: Structured summary of the trajectory
- `README.md`: This file

## Trajectory Structure
Each line in `trajectory.jsonl` represents a step in the agent's execution:
- **Initialization**: Problem setup and environment preparation
- **Repository Setup**: Code repository cloning and preparation
- **Agent Preparation**: Model selection and configuration
- **Agent Execution**: LLM calls, tool usage, and decision making
- **Final Result**: Success/failure status and generated patch

## Key Information
- All LLM interactions are logged with full input/output
- Tool calls include complete context and results
- Performance metrics track memory usage and execution time
- Session tracking allows following the complete decision path

## Usage
To analyze this trajectory:
```bash
# View the complete trajectory
cat trajectory.jsonl | jq '.'

# View only LLM calls
cat trajectory.jsonl | jq 'select(.action == "llm_call")'

# View only tool calls
cat trajectory.jsonl | jq 'select(.action == "tool_call")'

# View final result
cat trajectory.jsonl | jq 'select(.event == "final_result")'
```
