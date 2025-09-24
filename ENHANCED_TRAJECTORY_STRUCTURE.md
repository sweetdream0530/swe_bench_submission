# Enhanced SWE-Bench Trajectory Structure

This document describes the enhanced trajectory structure for SWE-bench submissions that includes all model I/O and filtering decisions, ensuring every LLM decision can be walked through.

## Overview

The enhanced trajectory structure provides comprehensive logging of:
- **LLM Interactions**: Complete input/output capture with metadata
- **Tool Calls**: Detailed tool usage with arguments and results
- **Filtering Decisions**: Model selection, strategy selection, and path selection decisions
- **Performance Metrics**: Resource usage and timing information
- **Walkability**: Ability to trace every LLM decision from start to finish

## Folder Structure

Each instance ID gets its own folder with the following structure:

```
{instance_id}/
├── llm_interactions/
│   └── interactions.json          # All LLM interactions with inputs/outputs
├── tool_calls/
│   └── calls.json                 # All tool calls with arguments/results
├── filtering_decisions/
│   └── decisions.json             # All filtering and decision points
├── performance_metrics/
│   └── metrics.json               # Performance and resource metrics
├── raw_logs/
│   └── trajectory.jsonl           # Original trajectory data
├── artifacts/
│   └── (generated files)          # Any generated artifacts
└── summary.json                   # Comprehensive summary and walkability info
```

## LLM Interactions Format

Each LLM interaction includes:

```json
{
  "timestamp": "2025-09-24T15:00:00.000000",
  "step": 1,
  "action": "llm_interaction",
  "details": {
    "model_name": "zai-org/GLM-4.5-FP8",
    "input_messages": [
      {
        "role": "system",
        "content": "You are a helpful coding assistant..."
      },
      {
        "role": "user", 
        "content": "Fix the bug in the separability matrix function..."
      }
    ],
    "response": "I'll analyze the separability matrix function...",
    "input_tokens": 150,
    "output_tokens": 200,
    "interaction_id": "astropy__astropy-12907_1_1758644314",
    "model_version": "4.5",
    "temperature": 0.0,
    "max_tokens": 4000,
    "stop_sequences": [],
    "system_prompt": "You are a helpful coding assistant...",
    "user_prompt": "Fix the bug in the separability matrix function...",
    "response_length": 200,
    "response_preview": "I'll analyze the separability matrix function..."
  },
  "performance": {
    "duration_seconds": 2.5,
    "memory_usage_mb": 150.0,
    "step_type": "reasoning"
  },
  "metadata": {
    "instance_id": "astropy__astropy-12907",
    "total_steps": 1,
    "session_id": "session_1758644314"
  }
}
```

## Tool Calls Format

Each tool call includes:

```json
{
  "timestamp": "2025-09-24T15:00:05.000000",
  "step": 2,
  "action": "tool_call",
  "details": {
    "tool_name": "read_file",
    "tool_args": {
      "file_path": "astropy/modeling/separable.py",
      "start_line": 240,
      "end_line": 250
    },
    "tool_result": "def _cstack(left, right):\n    # Implementation...",
    "execution_time": 0.1,
    "success": true
  },
  "performance": {
    "duration_seconds": 0.1,
    "memory_usage_mb": 152.0,
    "step_type": "execution"
  },
  "metadata": {
    "instance_id": "astropy__astropy-12907",
    "total_steps": 2,
    "session_id": "session_1758644314"
  }
}
```

## Filtering Decisions Format

Each filtering decision includes:

```json
{
  "timestamp": "2025-09-24T15:00:10.000000",
  "step": 3,
  "action": "model_selection",
  "details": {
    "decision_type": "model_selection",
    "available_models": [
      "zai-org/GLM-4.5-FP8",
      "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
      "moonshotai/Kimi-K2-Instruct"
    ],
    "selected_model": "zai-org/GLM-4.5-FP8",
    "selection_reason": "Best performance on code analysis tasks",
    "performance_metrics": {
      "confidence": 0.85,
      "previous_success_rate": 0.92
    },
    "decision_timestamp": "2025-09-24T15:00:10.000000",
    "confidence_score": 0.85
  },
  "performance": {
    "duration_seconds": 0.05,
    "memory_usage_mb": 153.0,
    "step_type": "decision"
  },
  "metadata": {
    "instance_id": "astropy__astropy-12907",
    "total_steps": 3,
    "session_id": "session_1758644314"
  }
}
```

## Summary Format

The summary.json provides a comprehensive overview:

```json
{
  "instance_id": "astropy__astropy-12907",
  "organization_timestamp": "2025-09-24T15:00:00.000000",
  "summary": {
    "total_trajectory_entries": 25,
    "llm_interactions_count": 8,
    "tool_calls_count": 12,
    "filtering_decisions_count": 5,
    "performance_metrics": {
      "total_steps": 25,
      "llm_interactions": 8,
      "tool_calls": 12,
      "filtering_decisions": 5,
      "total_duration": 45.2,
      "avg_memory_usage": 155.5,
      "max_memory_usage": 180.0
    }
  },
  "structure": {
    "llm_interactions": "llm_interactions/interactions.json",
    "tool_calls": "tool_calls/calls.json",
    "filtering_decisions": "filtering_decisions/decisions.json",
    "performance_metrics": "performance_metrics/metrics.json",
    "raw_trajectory": "raw_logs/trajectory.jsonl"
  },
  "walkability": {
    "has_llm_inputs": true,
    "has_llm_outputs": true,
    "has_filtering_decisions": true,
    "has_tool_calls": true,
    "is_walkable": true
  }
}
```

## Walkability Requirements

For a trajectory to be considered "walkable", it must include:

1. **LLM Inputs**: Complete input messages sent to the model
2. **LLM Outputs**: Full responses received from the model
3. **Tool Calls**: All tool usage with arguments and results
4. **Decision Points**: All filtering and selection decisions with reasoning
5. **Performance Data**: Timing and resource usage information

## Usage

### Organizing Existing Trajectories

```bash
python organize_trajectories.py --input_dir submission_trajectories --output_dir organized_trajectories
```

### Validating Trajectory Compliance

```bash
python validate_trajectory_format.py --trajectory_dir organized_trajectories
```

### Enhanced Logging in Code

```python
from trajectory_logger import TrajectoryLogger

logger = TrajectoryLogger("trajectories")

# Log LLM interaction
logger.log_llm_interaction(
    instance_id="astropy__astropy-12907",
    step_number=1,
    model_name="zai-org/GLM-4.5-FP8",
    messages=[{"role": "user", "content": "Fix the bug..."}],
    response="I'll analyze the code...",
    metadata={"temperature": 0.0, "max_tokens": 4000}
)

# Log model selection decision
logger.log_model_selection_decision(
    instance_id="astropy__astropy-12907",
    step_number=2,
    available_models=["model1", "model2"],
    selected_model="model1",
    selection_reason="Best performance on this task type",
    performance_metrics={"confidence": 0.85}
)

# Log filtering decision
logger.log_filtering_decision(
    instance_id="astropy__astropy-12907",
    step_number=3,
    filter_type="code_relevance",
    input_data="All code files",
    filtered_data="Relevant code files",
    filter_criteria={"relevance_threshold": 0.7},
    decision_reason="Focus on most relevant code"
)
```

## Compliance Checklist

- [ ] Each instance has its own folder
- [ ] LLM interactions include complete inputs and outputs
- [ ] Tool calls include arguments and results
- [ ] Filtering decisions include reasoning
- [ ] Performance metrics are captured
- [ ] Summary includes walkability assessment
- [ ] All LLM decisions can be traced from start to finish

## Benefits

1. **Transparency**: Complete visibility into model decision-making
2. **Debugging**: Easy to identify where issues occurred
3. **Analysis**: Rich data for performance analysis
4. **Compliance**: Meets SWE-bench submission requirements
5. **Reproducibility**: Full trace of execution path

This enhanced structure ensures that every LLM decision can be walked through, providing complete transparency and traceability for SWE-bench submissions.
