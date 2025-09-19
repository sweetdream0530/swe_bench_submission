# SWE-Bench Submission Checklist

## Pre-Submission Checklist

### Repository Structure
- [ ] Repository URL is accessible
- [ ] `main.py` exists with `run()` function matching the required interface
- [ ] `info.json` contains submission metadata (score, models used, date)
- [ ] `requirements.txt` lists all necessary dependencies
- [ ] `trajectories/` directory exists for trajectory logs
- [ ] `evaluation_results/` directory exists for evaluation results

### Function Interface
- [ ] `run()` function accepts all required parameters:
  - `problem_statement: str`
  - `repo_path: str`
  - `instance_id: str`
  - `traj_dir: str`
  - `temp_dir: str`
  - `log_path: str`
  - `test_file_path: str = None`
  - `test_case_name: str = None`
  - `timeout: int = 900`
  - `**kwargs`
- [ ] `run()` function returns a Git patch string
- [ ] Function is easy to run and test

### Trajectory Logging
- [ ] All model I/O is logged to trajectory files
- [ ] Filtering decisions are recorded
- [ ] One folder per instance ID
- [ ] Every LLM decision is walkable
- [ ] Trajectories follow official SWE-Bench format

### Dependencies
- [ ] `requirements.txt` includes all necessary packages
- [ ] Dependencies can be installed via `pip install -r requirements.txt`
- [ ] No missing dependencies that would cause runtime errors

### Evaluation Results
- [ ] Official SWE-Bench results folder is included
- [ ] Results are properly formatted
- [ ] Score claims are accurate and verifiable

### Documentation
- [ ] README.md explains the submission
- [ ] Code is well-commented
- [ ] Submission follows SWE-Bench guidelines

## Post-Submission Verification

### Environment Testing
- [ ] Submission works with CHUTES_ONLY=true
- [ ] All dependencies install correctly
- [ ] No runtime errors in test environment

### Score Validation
- [ ] Claimed score is achievable
- [ ] Score is ≥2% higher than current top verified score
- [ ] Results are reproducible

### Final Review
- [ ] All checklist items completed
- [ ] Submission ready for evaluation
- [ ] Contact information provided for manual review if needed

## Notes

- Ensure your submission beats the top SWE-Bench Verified score by at least 2%
- The evaluation will run with CHUTES_ONLY=true initially
- High-scoring submissions will trigger manual review
- Contact the organizers if there are model discrepancy issues