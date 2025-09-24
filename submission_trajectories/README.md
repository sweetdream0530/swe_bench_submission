# SWE-Bench Submission Trajectories

This directory contains trajectories for 15 SWE-bench instances.

## Structure
Each instance has its own folder named `{instance_id}` containing:
- `trajectory.jsonl`: Complete trajectory with all model I/O and decisions
- `summary.json`: Structured summary of the trajectory
- `README.md`: Human-readable description

## Trajectory Format
Each trajectory file contains JSONL format where each line represents a step:
- **Initialization**: Problem setup and environment preparation
- **Repository Setup**: Code repository cloning and preparation  
- **Agent Preparation**: Model selection and configuration
- **Agent Execution**: LLM calls, tool usage, and decision making
- **Final Result**: Success/failure status and generated patch

## Key Features
- ✅ Complete LLM input/output logging
- ✅ Full tool call context and results
- ✅ Performance metrics (memory, time)
- ✅ Session tracking for decision paths
- ✅ Filtering decisions documented
- ✅ All model I/O preserved

## Instance List
- `astropy__astropy-12907/`
- `astropy__astropy-13033/`
- `astropy__astropy-13236/`
- `astropy__astropy-13398/`
- `astropy__astropy-13453/`
- `astropy__astropy-13579/`
- `astropy__astropy-13977/`
- `astropy__astropy-14096/`
- `astropy__astropy-14182/`
- `astropy__astropy-14309/`
- `astropy__astropy-14365/`
- `astropy__astropy-14369/`
- `astropy__astropy-14508/`
- `astropy__astropy-14539/`
- `astropy__astropy-14598/`
- `astropy__astropy-14995/`

## Usage Examples

### View all trajectories
```bash
find . -name "trajectory.jsonl" | head -5
```

### Analyze specific instance
```bash
cd astropy__astropy-12907
cat trajectory.jsonl | jq 'select(.action == "llm_call")'
```

### Check success rate
```bash
find . -name "summary.json" -exec jq '.final_result.success' {} \; | grep true | wc -l
```

## Submission Requirements Met
- ✅ One folder per instance ID
- ✅ Complete model I/O logging
- ✅ All filtering decisions documented
- ✅ Walkable LLM decision paths
- ✅ Comprehensive trajectory information
