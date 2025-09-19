# SWE-Bench Submission

This repository contains a submission for the SWE-Bench evaluation benchmark.

## Overview

This submission aims to beat the top SWE-Bench Verified score by ≥2%. The implementation includes:

- A main entry point (`main.py`) that implements the required `run` function
- Trajectory logging for all model inputs, outputs, and decisions
- Patch generation utilities for creating Git patches
- Comprehensive error handling and recovery mechanisms

## Repository Structure

```
swe_bench_submission/
├── main.py                    # Main entry point with run() function
├── info.json                  # Submission metadata (score, models used)
├── requirements.txt           # Python dependencies
├── trajectory_logger.py      # Utility for logging model trajectories
├── patch_generator.py        # Utility for generating Git patches
├── trajectories/             # Directory for trajectory logs
├── evaluation_results/       # Directory for evaluation results
└── README.md                 # This file
```

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The main entry point is the `run` function in `main.py`. This function is called by the SWE-Bench evaluation harness with the following parameters:

- `problem_statement`: Description of the issue to fix
- `repo_path`: Path to the repository to modify
- `instance_id`: Unique identifier for the task
- `traj_dir`: Directory to write trajectory logs
- `temp_dir`: Temporary directory for intermediate files
- `log_path`: Path to the log file
- `test_file_path`: Optional path to specific test file
- `test_case_name`: Optional name of specific test case
- `timeout`: Maximum time allowed (default: 900 seconds)

The function returns a Git patch string that fixes the described issue.

## Trajectory Logging

All model interactions and decisions are logged to the trajectory directory. Each instance gets its own trajectory file containing:

- Step-by-step actions taken
- Model inputs and outputs
- File modifications
- Test results
- Final patch generation

## Evaluation

This submission will be evaluated on the SWE-Bench dataset with the following considerations:

- **Environment limits**: CHUTES_ONLY=true (only Chutes-hosted models allowed)
- **Auto-validation**: Scored on a subset of tasks
- **Manual review**: Triggered for high-scoring submissions
- **Model discrepancy**: Contact required if performance differs significantly between model sets

## Contributing

This is a competition submission. Please refer to the SWE-Bench evaluation guidelines for submission requirements.

## License

[Add your license here]