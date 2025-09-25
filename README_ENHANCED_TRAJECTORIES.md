# Enhanced SWE-Bench Trajectory Submission

This repository contains an enhanced SWE-bench submission with comprehensive trajectory logging that includes all model I/O and filtering decisions, ensuring every LLM decision can be walked through.

## 🎯 Overview

This enhanced submission addresses the SWE-bench requirement that "Trajectories must include all model I/O and filtering decisions. We should be able to walk every LLM decision." by providing:

- **Complete LLM Interaction Logging**: Every input and output to/from the model
- **Comprehensive Tool Call Tracking**: All tool usage with arguments and results  
- **Detailed Filtering Decisions**: Model selection, strategy selection, and path selection reasoning
- **Performance Metrics**: Resource usage and timing information
- **Walkable Trajectories**: Ability to trace every decision from start to finish

## 📁 Repository Structure

```
swe_bench_submission/
├── agent.py                          # Enhanced agent with comprehensive logging
├── main.py                           # Main entry point with trajectory support
├── trajectory_logger.py              # Enhanced trajectory logging system
├── organize_trajectories.py          # Script to organize trajectories
├── validate_trajectory_format.py     # Validation script for compliance
├── submission_trajectories/          # Original trajectory files
├── organized_trajectories/           # Organized trajectory structure
├── ENHANCED_TRAJECTORY_STRUCTURE.md  # Detailed structure documentation
└── README.md                         # This file
```

## 🚀 Key Features

### 1. Enhanced Trajectory Logging

The `TrajectoryLogger` class provides comprehensive logging capabilities:

```python
# Log LLM interactions with complete I/O
logger.log_llm_interaction(
    instance_id="astropy__astropy-12907",
    step_number=1,
    model_name="zai-org/GLM-4.5-FP8",
    messages=[{"role": "user", "content": "Fix the bug..."}],
    response="I'll analyze the code...",
    metadata={"temperature": 0.0, "max_tokens": 4000}
)

# Log model selection decisions
logger.log_model_selection_decision(
    instance_id="astropy__astropy-12907",
    step_number=2,
    available_models=["model1", "model2"],
    selected_model="model1",
    selection_reason="Best performance on this task type"
)

# Log filtering decisions
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

### 2. Organized Folder Structure

Each instance gets its own folder with comprehensive data:

```
{instance_id}/
├── llm_interactions/
│   └── interactions.json          # All LLM interactions
├── tool_calls/
│   └── calls.json                 # All tool calls
├── filtering_decisions/
│   └── decisions.json             # All filtering decisions
├── performance_metrics/
│   └── metrics.json               # Performance data
├── raw_logs/
│   └── trajectory.jsonl           # Original trajectory
└── summary.json                   # Comprehensive summary
```

### 3. Walkability Validation

The validation script ensures trajectories meet SWE-bench requirements:

```bash
python validate_trajectory_format.py --trajectory_dir organized_trajectories
```

## 📊 Current Status

### Validation Results

- **Total Instances**: 16
- **Compliant Instances**: 0 (current trajectories lack detailed LLM I/O)
- **Non-compliant Instances**: 16
- **Overall Compliance**: ❌ FAIL

### Recommendations for Compliance

All instances need:
1. **Enhanced LLM interaction logging** - capture inputs, outputs, and metadata
2. **Improved tool call logging** - capture tool names, arguments, and results  
3. **Added filtering decision tracking** - log decision types and reasoning
4. **Improved walkability** - ensure all LLM decisions can be traced

## 🛠️ Usage

### 1. Organize Existing Trajectories

```bash
python organize_trajectories.py --input_dir submission_trajectories --output_dir organized_trajectories
```

### 2. Validate Trajectory Compliance

```bash
python validate_trajectory_format.py --trajectory_dir organized_trajectories
```

### 3. Run Enhanced Evaluation

```bash
python evaluate_swe_bench.py --dataset_name princeton-nlp/SWE-bench_Verified --output_file predictions.jsonl
```

## 📋 Compliance Checklist

- [x] **Folder Structure**: One folder per instance ID
- [x] **LLM Interactions**: Complete input/output capture framework
- [x] **Tool Calls**: Detailed tool usage tracking framework
- [x] **Filtering Decisions**: Decision reasoning framework
- [x] **Performance Metrics**: Resource usage tracking
- [x] **Validation Script**: Compliance checking
- [x] **Documentation**: Comprehensive structure guide
- [ ] **Implementation**: Apply enhanced logging to agent execution

## 🔧 Implementation Notes

### Current Trajectories

The existing trajectories in `submission_trajectories/` contain basic step logging but lack the detailed LLM I/O and filtering decision data required for walkability.

### Enhanced Logging Required

To achieve compliance, the agent execution needs to be updated to use the enhanced logging methods:

1. **Replace basic logging** with comprehensive LLM interaction logging
2. **Add model selection decisions** when choosing between available models
3. **Log filtering decisions** when applying any filtering or selection criteria
4. **Capture tool call details** with complete arguments and results

### Example Integration

```python
# In agent.py, replace basic logging with:
logger.log_llm_interaction(
    instance_id=input_dict['instance_id'],
    step_number=current_step,
    model_name=selected_model,
    messages=messages,
    response=response,
    metadata={
        'temperature': temperature,
        'max_tokens': max_tokens,
        'model_version': model_version
    }
)

# Add decision logging:
logger.log_model_selection_decision(
    instance_id=input_dict['instance_id'],
    step_number=current_step,
    available_models=available_models,
    selected_model=selected_model,
    selection_reason=selection_reason,
    performance_metrics=performance_metrics
)
```

## 📈 Expected Benefits

1. **Complete Transparency**: Every LLM decision can be traced
2. **Easy Debugging**: Identify exactly where issues occurred
3. **Rich Analysis**: Comprehensive data for performance analysis
4. **SWE-bench Compliance**: Meets all submission requirements
5. **Reproducibility**: Full execution trace available

## 🎯 Next Steps

1. **Update Agent Execution**: Integrate enhanced logging into the main agent
2. **Test Enhanced Logging**: Run evaluation with comprehensive trajectory capture
3. **Validate Compliance**: Ensure all trajectories meet walkability requirements
4. **Submit Enhanced Version**: Provide SWE-bench with fully compliant trajectories

## 📚 Documentation

- [Enhanced Trajectory Structure](ENHANCED_TRAJECTORY_STRUCTURE.md) - Detailed format documentation
- [Trajectory Logging Guide](TRAJECTORY_LOGGING_GUIDE.md) - Usage examples
- [SWE-bench Evaluation Guide](SWE_BENCH_EVALUATION_GUIDE.md) - Evaluation instructions

## 🤝 Contributing

This enhanced submission framework provides the foundation for SWE-bench compliant trajectories. The key is implementing the enhanced logging throughout the agent execution to capture all model I/O and filtering decisions.

## 📄 License

This project follows the same license as the original SWE-bench submission framework.

---

**Note**: This enhanced submission addresses the specific requirement that "Trajectories must include all model I/O and filtering decisions. We should be able to walk every LLM decision." The framework is ready - implementation requires integrating the enhanced logging into the agent execution flow.

