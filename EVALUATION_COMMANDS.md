# SWE-Bench Evaluation Commands

This file provides ready-to-use commands for running SWE-bench verified test data evaluation and generating predictions.

## 🚀 Quick Start Commands

### 1. Basic Evaluation (Recommended)

```bash
# Generate predictions for all 500 verified instances
python evaluate_swe_bench.py \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --output_file results/predictions/predictions.jsonl \
    --traj_dir results/trajectories \
    --max_workers 4 \
    --timeout 2200
```

### 2. Parallel Evaluation (Faster)

```bash
# Use more workers for faster processing
python evaluate_swe_bench.py \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --output_file results/predictions/predictions.jsonl \
    --traj_dir results/trajectories \
    --max_workers 8 \
    --timeout 2200
```

### 3. Resume Interrupted Evaluation

```bash
# Resume from existing predictions file
python evaluate_swe_bench.py \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --output_file results/predictions/predictions.jsonl \
    --traj_dir results/trajectories \
    --max_workers 4 \
    --timeout 2200 \
    --resume
```

## 🔧 SWE-Bench Harness Evaluation

After generating predictions, run the official evaluation harness:

### Standard Evaluation

```bash
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path results/predictions/predictions.jsonl \
    --max_workers 8 \
    --run_id enhanced_swe_bench_evaluation
```

### High-Performance Evaluation

```bash
# Use more workers and longer timeout
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path results/predictions/predictions.jsonl \
    --max_workers 12 \
    --run_id enhanced_swe_bench_parallel \
    --timeout 3600
```

### Evaluation with Custom Output

```bash
# Specify custom output directory
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path results/predictions/predictions.jsonl \
    --max_workers 8 \
    --run_id enhanced_swe_bench_evaluation \
    --output_dir evaluation_results
```

## 🧪 Testing Commands

### Test on Subset

```bash
# Test on first 10 instances
python evaluate_swe_bench.py \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --output_file results/predictions/test_predictions.jsonl \
    --traj_dir results/trajectories \
    --max_workers 2 \
    --timeout 1200 \
    --max_instances 10
```

### Debug Mode (Sequential)

```bash
# Run sequentially for debugging
python evaluate_swe_bench.py \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --output_file results/predictions/debug_predictions.jsonl \
    --traj_dir results/trajectories \
    --max_workers 1 \
    --timeout 3600 \
    --max_instances 5
```

### Test Agent Components

```bash
# Test the enhanced submission components
python test_enhanced_submission.py
```

## 🔍 Monitoring Commands

### Check Proxy Server Status

```bash
# Check if proxy server is running
pm2 status
pm2 logs proxy-server --lines 20

# Restart proxy server if needed
pm2 restart proxy-server
```

### Monitor Resource Usage

```bash
# Monitor system resources
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%'); print(f'CPU: {psutil.cpu_percent()}%')"

# Monitor PM2 processes
pm2 monit
```

### Check Trajectory Logs

```bash
# View recent trajectory files
ls -la results/trajectories/ | head -10

# Check trajectory file content
head -5 results/trajectories/django__django-10097.jsonl

# Count completed trajectories
ls results/trajectories/*.jsonl | wc -l
```

## 📊 Analysis Commands

### Validate Predictions Format

```bash
# Check predictions file format
python -c "
import json
with open('results/predictions/predictions.jsonl') as f:
    lines = f.readlines()
    print(f'Total predictions: {len(lines)}')
    if lines:
        sample = json.loads(lines[0])
        print(f'Sample prediction keys: {list(sample.keys())}')
"

# Count predictions by status
python -c "
import json
with open('results/predictions/predictions.jsonl') as f:
    predictions = [json.loads(line) for line in f]
    empty_patches = sum(1 for p in predictions if not p['model_patch'].strip())
    print(f'Empty patches: {empty_patches}')
    print(f'Valid patches: {len(predictions) - empty_patches}')
"
```

### Analyze Trajectory Performance

```bash
# Analyze trajectory statistics
python -c "
import json
from pathlib import Path

traj_dir = Path('results/trajectories')
trajectories = list(traj_dir.glob('*.jsonl'))

total_steps = 0
total_llm_calls = 0
total_tool_calls = 0

for traj_file in trajectories:
    with open(traj_file) as f:
        steps = [json.loads(line) for line in f]
    total_steps += len(steps)
    total_llm_calls += len([s for s in steps if s.get('action') == 'llm_call'])
    total_tool_calls += len([s for s in steps if s.get('action') == 'tool_call'])

print(f'Total trajectories: {len(trajectories)}')
print(f'Total steps: {total_steps}')
print(f'Average steps per trajectory: {total_steps/len(trajectories):.1f}')
print(f'Total LLM calls: {total_llm_calls}')
print(f'Total tool calls: {total_tool_calls}')
"
```

## 🚨 Troubleshooting Commands

### Fix Common Issues

```bash
# Fix permission issues
chmod +x evaluate_swe_bench.py
chmod -R 755 results/

# Clean up failed processes
pm2 delete proxy-server
pm2 start "source venv/bin/activate && python3 ./ridges.py proxy run --no-auto-update" --name proxy-server

# Clear cache and restart
rm -rf __pycache__/
rm -rf .pytest_cache/
python -c "import gc; gc.collect()"
```

### Check Dependencies

```bash
# Verify required packages
python -c "
try:
    import datasets
    print('✅ datasets available')
except ImportError:
    print('❌ datasets not available - install with: pip install datasets')

try:
    import psutil
    print('✅ psutil available')
except ImportError:
    print('❌ psutil not available - install with: pip install psutil')

try:
    import requests
    print('✅ requests available')
except ImportError:
    print('❌ requests not available - install with: pip install requests')
"
```

### Validate Environment

```bash
# Check environment variables
echo "TRAJ_DIR: $TRAJ_DIR"
echo "AI_PROXY_URL: $AI_PROXY_URL"
echo "AGENT_TIMEOUT: $AGENT_TIMEOUT"

# Test proxy connectivity
python -c "
import requests
import os
proxy_url = os.getenv('AI_PROXY_URL', 'http://localhost:8001')
try:
    response = requests.get(f'{proxy_url}/health', timeout=5)
    print(f'✅ Proxy server responding: {response.status_code}')
except Exception as e:
    print(f'❌ Proxy server not responding: {e}')
"
```

## 📈 Performance Optimization Commands

### Memory Management

```bash
# Monitor memory usage during evaluation
python -c "
import psutil
import time
while True:
    memory = psutil.virtual_memory()
    print(f'Memory: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)')
    time.sleep(30)
" &
```

### Parallel Processing Tuning

```bash
# Find optimal worker count
python -c "
import os
cpu_count = os.cpu_count()
print(f'Available CPUs: {cpu_count}')
print(f'Recommended workers: {min(cpu_count, 8)}')
print(f'Maximum workers: {cpu_count}')
"
```

### Cache Management

```bash
# Clear evaluation cache
rm -rf /tmp/swe_bench_*
rm -rf results/logs/*
rm -rf results/trajectories/*.jsonl

# Clear system cache
sudo sync
echo 3 | sudo tee /proc/sys/vm/drop_caches
```

## 🎯 Complete Evaluation Pipeline

### Full Evaluation Script

```bash
#!/bin/bash
# complete_evaluation.sh

echo "🚀 Starting Complete SWE-Bench Evaluation"

# 1. Setup environment
export TRAJ_DIR="results/trajectories"
export AI_PROXY_URL="http://localhost:8001"
export AGENT_TIMEOUT="2200"

# 2. Start proxy server
echo "📡 Starting proxy server..."
pm2 start "source venv/bin/activate && python3 ./ridges.py proxy run --no-auto-update" --name proxy-server
sleep 5

# 3. Generate predictions
echo "🤖 Generating predictions..."
python evaluate_swe_bench.py \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --output_file results/predictions/predictions.jsonl \
    --traj_dir results/trajectories \
    --max_workers 8 \
    --timeout 2200

# 4. Run evaluation harness
echo "📊 Running evaluation harness..."
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path results/predictions/predictions.jsonl \
    --max_workers 8 \
    --run_id enhanced_swe_bench_evaluation

# 5. Analyze results
echo "📈 Analyzing results..."
python -c "
import json
from pathlib import Path

# Check predictions
with open('results/predictions/predictions.jsonl') as f:
    predictions = [json.loads(line) for line in f]
print(f'Total predictions: {len(predictions)}')

# Check trajectories
traj_dir = Path('results/trajectories')
trajectories = list(traj_dir.glob('*.jsonl'))
print(f'Total trajectories: {len(trajectories)}')

# Check evaluation results
eval_dir = Path('evaluation_results')
if eval_dir.exists():
    results_file = eval_dir / 'results.json'
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        print(f'Evaluation results: {results}')
"

echo "✅ Evaluation completed!"
```

### Make it executable and run:

```bash
chmod +x complete_evaluation.sh
./complete_evaluation.sh
```

## 📋 Command Reference

### Environment Variables
```bash
export TRAJ_DIR="/path/to/trajectories"           # Trajectory output directory
export AI_PROXY_URL="http://localhost:8001"       # Proxy server URL
export AGENT_TIMEOUT="2200"                       # Agent timeout in seconds
export MAX_STEPS="200"                            # Maximum agent steps
```

### Key Directories
```bash
results/predictions/     # Predictions JSONL files
results/trajectories/    # Trajectory JSONL files
results/logs/           # Evaluation logs
evaluation_results/     # SWE-bench harness results
```

### Important Files
```bash
predictions.jsonl       # Main predictions file
evaluate_swe_bench.py   # Evaluation script
main.py                 # Entry point with run() function
trajectory_logger.py    # Trajectory logging system
```

## 🎉 Success Indicators

Your evaluation is successful when:

- ✅ `predictions.jsonl` contains 500 predictions
- ✅ Each prediction has correct format (instance_id, model_name_or_path, model_patch)
- ✅ Trajectory files exist for all instances
- ✅ Evaluation harness completes without errors
- ✅ Results show improvement over baseline scores
- ✅ All logs are properly generated

Use these commands to verify success and troubleshoot any issues during your SWE-bench evaluation.

