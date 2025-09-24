# Trajectory Logging Guide for SWE-Bench

This guide explains the comprehensive trajectory logging requirements for SWE-bench evaluation and how the enhanced submission implements them.

## 📋 Overview

Trajectory logging is **mandatory** for SWE-bench evaluation. You must log every interaction with LLMs, tool calls, and reasoning steps to provide transparency and reproducibility.

## 🎯 Required Logging Elements

### 1. LLM Interactions
Every call to an LLM must be logged with:
- **Input messages** (complete conversation history)
- **Model name** used
- **Response content** (full response)
- **Timestamp** and **duration**
- **Token usage** (if available)
- **Temperature** and other parameters

### 2. Tool Calls
Every tool usage must be logged with:
- **Tool name** and **parameters**
- **Input** provided to the tool
- **Output** returned by the tool
- **Execution time**
- **Success/failure status**

### 3. Reasoning Steps
All reasoning and decision-making must be logged:
- **Analysis steps** (what the agent is thinking)
- **Search strategies** used
- **File operations** performed
- **Code modifications** made
- **Test executions** and results

### 4. Self-Consistency Logs
If using self-consistency algorithms:
- **Number of reasoning paths**
- **Individual path results**
- **Consensus mechanism**
- **Final decision** and confidence

### 5. Performance Metrics
System and resource information:
- **Memory usage** over time
- **CPU utilization**
- **Execution time** for each operation
- **Cache hit rates**
- **Error rates** and retry attempts

## 📁 Trajectory File Structure

Each instance generates a trajectory file: `{instance_id}.jsonl`

```json
{"step": 1, "action": "initialization", "timestamp": "2025-01-27T10:30:00Z", "data": {...}}
{"step": 2, "action": "llm_call", "timestamp": "2025-01-27T10:30:05Z", "model": "GLM-4.5-FP8", "messages": [...], "response": "...", "duration": 2.5, "tokens_used": 150}
{"step": 3, "action": "file_search", "timestamp": "2025-01-27T10:30:10Z", "query": "test file", "results": [...], "duration": 0.8}
{"step": 4, "action": "code_modification", "timestamp": "2025-01-27T10:30:15Z", "file": "src/main.py", "changes": "...", "duration": 1.2}
{"step": 5, "action": "test_execution", "timestamp": "2025-01-27T10:30:20Z", "command": "python -m pytest", "result": "PASSED", "duration": 5.3}
{"step": 6, "action": "final_result", "timestamp": "2025-01-27T10:30:25Z", "patch": "...", "success": true, "total_time": 25.0}
```

## 🔧 Enhanced Submission Implementation

The enhanced submission includes a comprehensive `TrajectoryLogger` class that automatically logs:

### Automatic Logging Features

```python
from trajectory_logger import TrajectoryLogger

# Initialize logger
logger = TrajectoryLogger(traj_dir="/path/to/trajectories")

# Log initialization
logger.log_step(instance_id, 1, "initialization", {
    "repo_path": repo_path,
    "problem_statement": problem_statement,
    "models_available": ["GLM-4.5-FP8", "Qwen3-Coder-480B", ...]
})

# Log LLM interactions
logger.log_llm_interaction(
    instance_id, 2, "GLM-4.5-FP8",
    messages=[{"role": "user", "content": "Analyze this code..."}],
    response="The code has a bug in line 42...",
    duration=2.5,
    tokens_used=150
)

# Log tool calls
logger.log_tool_call(
    instance_id, 3, "file_search",
    parameters={"query": "test files", "pattern": "test_*.py"},
    result=["test_main.py", "test_utils.py"],
    duration=0.8
)

# Log self-consistency steps
logger.log_self_consistency_step(
    instance_id, 4, path_id=0, strategy="analytical",
    reasoning_steps=["Step 1: Analyze problem", "Step 2: Find solution"],
    final_answer="The fix is to add error handling",
    confidence=0.85
)

# Log intelligent search
logger.log_intelligent_search(
    instance_id, 5, "find buggy function",
    strategies_used=["semantic_search", "file_search"],
    results_by_strategy={
        "semantic_search": [{"file": "main.py", "score": 0.9}],
        "file_search": [{"file": "main.py", "matches": 5}]
    },
    fused_results=[{"file": "main.py", "weight": 0.95, "confidence": 0.9}]
)

# Log final result
logger.log_final_result(
    instance_id, patch_content, success=True, 
    status_message="Successfully generated patch"
)
```

### Advanced Logging Features

#### 1. Performance Monitoring
```python
# Automatic resource monitoring
logger.log_performance_metrics(instance_id, {
    "memory_usage_mb": 2048,
    "cpu_percent": 45.2,
    "cache_hit_rate": 0.78,
    "total_operations": 25
})
```

#### 2. Error Tracking
```python
# Automatic error logging
logger.log_error(instance_id, step, "LLM timeout", {
    "error_type": "TimeoutError",
    "retry_count": 2,
    "fallback_model": "Qwen3-Coder-480B"
})
```

#### 3. Cache Operations
```python
# Cache hit/miss logging
logger.log_cache_operation(instance_id, "get", "repo_clone_django", {
    "hit": True,
    "ttl_remaining": 180,
    "size_mb": 2.5
})
```

## 📊 Trajectory Analysis

### Performance Summary
```python
# Get comprehensive performance summary
summary = logger.get_performance_summary(instance_id)
print(f"Total steps: {summary['total_steps']}")
print(f"LLM calls: {summary['llm_calls']}")
print(f"Tool calls: {summary['tool_calls']}")
print(f"Total time: {summary['total_time']:.2f}s")
print(f"Success rate: {summary['success_rate']:.1%}")
```

### Trajectory Export
```python
# Export trajectory for analysis
trajectory = logger.get_trajectory(instance_id)
with open(f"{instance_id}_trajectory.json", "w") as f:
    json.dump(trajectory, f, indent=2)
```

## 🎯 SWE-Bench Compliance

### Required Fields
Each trajectory entry must include:
- ✅ **step**: Sequential step number
- ✅ **action**: Type of action performed
- ✅ **timestamp**: ISO format timestamp
- ✅ **duration**: Time taken (in seconds)

### LLM Interaction Fields
- ✅ **model**: Model name used
- ✅ **messages**: Complete input messages
- ✅ **response**: Full LLM response
- ✅ **tokens_used**: Token count (if available)
- ✅ **temperature**: Sampling temperature

### Tool Call Fields
- ✅ **tool_name**: Name of the tool
- ✅ **parameters**: Input parameters
- ✅ **result**: Tool output
- ✅ **success**: Boolean success status

## 🚀 Best Practices

### 1. Comprehensive Logging
```python
# Log everything - better to have too much than too little
logger.log_step(instance_id, step, "detailed_action", {
    "input": full_input,
    "output": full_output,
    "metadata": additional_context,
    "performance": timing_info
})
```

### 2. Structured Data
```python
# Use consistent data structures
action_data = {
    "query": "find test files",
    "search_scope": "repository",
    "filters": ["*.py", "test_*"],
    "results": [{"file": "test_main.py", "score": 0.95}],
    "duration": 0.8,
    "success": True
}
logger.log_tool_call(instance_id, step, "file_search", action_data)
```

### 3. Error Handling
```python
try:
    result = perform_operation()
    logger.log_step(instance_id, step, "operation_success", {"result": result})
except Exception as e:
    logger.log_error(instance_id, step, "operation_failed", {
        "error": str(e),
        "error_type": type(e).__name__,
        "traceback": traceback.format_exc()
    })
```

### 4. Performance Tracking
```python
start_time = time.time()
# ... perform operation ...
duration = time.time() - start_time

logger.log_step(instance_id, step, "operation", {
    "duration": duration,
    "memory_before": get_memory_usage(),
    "memory_after": get_memory_usage()
})
```

## 🔍 Trajectory Validation

### Validation Checklist
- [ ] Every LLM call is logged
- [ ] Every tool usage is logged
- [ ] All file operations are logged
- [ ] Code modifications are logged
- [ ] Test executions are logged
- [ ] Error cases are logged
- [ ] Performance metrics are tracked
- [ ] Timestamps are accurate
- [ ] Step numbers are sequential
- [ ] Data is structured consistently

### Validation Script
```python
def validate_trajectory(instance_id, traj_dir):
    """Validate trajectory completeness and format."""
    logger = TrajectoryLogger(traj_dir)
    trajectory = logger.get_trajectory(instance_id)
    
    # Check required steps
    required_actions = ["initialization", "llm_call", "tool_call", "final_result"]
    found_actions = [step["action"] for step in trajectory]
    
    for action in required_actions:
        if action not in found_actions:
            print(f"⚠️ Missing required action: {action}")
    
    # Check LLM interactions
    llm_calls = [step for step in trajectory if step["action"] == "llm_call"]
    print(f"📊 LLM calls logged: {len(llm_calls)}")
    
    # Check tool calls
    tool_calls = [step for step in trajectory if step["action"] == "tool_call"]
    print(f"🔧 Tool calls logged: {len(tool_calls)}")
    
    return len(trajectory) > 0 and len(llm_calls) > 0
```

## 📈 Trajectory Analytics

### Success Metrics
```python
def analyze_trajectory_performance(traj_dir):
    """Analyze trajectory performance across instances."""
    total_instances = 0
    successful_instances = 0
    total_llm_calls = 0
    total_tool_calls = 0
    
    for traj_file in Path(traj_dir).glob("*.jsonl"):
        with open(traj_file) as f:
            trajectory = [json.loads(line) for line in f]
        
        total_instances += 1
        
        # Check if instance was successful
        final_step = trajectory[-1] if trajectory else {}
        if final_step.get("action") == "final_result" and final_step.get("success"):
            successful_instances += 1
        
        # Count interactions
        total_llm_calls += len([s for s in trajectory if s["action"] == "llm_call"])
        total_tool_calls += len([s for s in trajectory if s["action"] == "tool_call"])
    
    print(f"📊 Trajectory Analysis:")
    print(f"  Total instances: {total_instances}")
    print(f"  Success rate: {successful_instances/total_instances:.1%}")
    print(f"  Average LLM calls per instance: {total_llm_calls/total_instances:.1f}")
    print(f"  Average tool calls per instance: {total_tool_calls/total_instances:.1f}")
```

## 🎉 Summary

The enhanced submission provides comprehensive trajectory logging that meets all SWE-bench requirements:

- ✅ **Complete LLM interaction logging**
- ✅ **Detailed tool call tracking**
- ✅ **Performance monitoring**
- ✅ **Error handling and logging**
- ✅ **Structured data format**
- ✅ **Automatic resource tracking**
- ✅ **Self-consistency logging**
- ✅ **Intelligent search logging**

This ensures full transparency and reproducibility for your SWE-bench evaluation.


