#!/usr/bin/env python3
"""
Test script for SWE-Bench submission
"""

import os
import json
import subprocess
import sys

# Add the submission directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from main import run

def test_submission_structure():
    """Test if the submission repository has the correct structure"""
    print("Testing submission structure...")
    
    required_files = [
        "main.py",
        "info.json", 
        "requirements.txt",
        "trajectory_logger.py",
        "patch_generator.py",
        "README.md",
        "LICENSE",
        "SUBMISSION_CHECKLIST.md"
    ]
    
    required_dirs = [
        "trajectories",
        "evaluation_results"
    ]
    
    for file_path in required_files:
        assert os.path.exists(file_path), f"Required file {file_path} not found"
        print(f"✓ {file_path}")
    
    for dir_path in required_dirs:
        assert os.path.isdir(dir_path), f"Required directory {dir_path} not found"
        print(f"✓ {dir_path}/")
    
    print("✓ All required files and directories present")

def test_main_run_function():
    """Test the run function with dummy data"""
    print("\nTesting main run function...")
    
    # Set testing mode to avoid network requests
    os.environ["TESTING_MODE"] = "true"
    
    # Create a temporary test repository
    test_repo = "/tmp/swe_bench_test_repo"
    test_traj_dir = "/tmp/swe_bench_test_trajectories"
    test_temp_dir = "/tmp/swe_bench_test_temp"
    test_log_path = "/tmp/swe_bench_test_log.txt"
    
    try:
        # Setup test repository
        os.makedirs(test_repo, exist_ok=True)
        os.chdir(test_repo)
        
        # Initialize git repository
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], check=True, capture_output=True)
        
        # Create initial file
        with open("test_file.py", "w") as f:
            f.write("print('Hello, World!')\n")
        
        subprocess.run(["git", "add", "."], check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], check=True, capture_output=True)
        
        # Change back to submission directory
        os.chdir(os.path.dirname(__file__))
        
        # Test the run function
        patch = run(
            repo_path=test_repo,
            instance_id="test_instance_001",
            base_commit="test_commit_123",
            problem_statement="Fix a bug in the test file",
            version="1.0.0",
            traj_dir=test_traj_dir,
            temp_dir=test_temp_dir,
            log_path=test_log_path
        )
        
        # Verify the patch is a string
        assert isinstance(patch, str), "run function should return a string"
        assert len(patch) > 0, "run function should return a non-empty patch"
        
        # Verify trajectory file was created
        traj_file = os.path.join(test_traj_dir, "test_instance_001.jsonl")
        assert os.path.exists(traj_file), f"Trajectory file {traj_file} not found"
        
        # Read and verify trajectory content
        trajectory_entries = []
        with open(traj_file, "r") as f:
            for line in f:
                try:
                    trajectory_entries.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass
        
        assert len(trajectory_entries) > 0, "Trajectory file should contain entries"
        
        # Check for final result entry
        final_entries = [entry for entry in trajectory_entries if entry.get("event") == "final_result"]
        assert len(final_entries) > 0, "Should have final result entry"
        assert final_entries[0]["success"] == True, "Final result should be successful"
        
        print("✓ run function works correctly")
        print(f"✓ Generated patch length: {len(patch)} characters")
        print(f"✓ Trajectory file created: {traj_file}")
        
    finally:
        # Cleanup
        import shutil
        if os.path.exists(test_repo):
            shutil.rmtree(test_repo)
        if os.path.exists(test_traj_dir):
            shutil.rmtree(test_traj_dir)
        if os.path.exists(test_temp_dir):
            shutil.rmtree(test_temp_dir)
        if os.path.exists(test_log_path):
            os.remove(test_log_path)

def test_info_json():
    """Test that info.json has the correct structure"""
    print("\nTesting info.json...")
    
    with open("info.json", "r") as f:
        info = json.load(f)
    
    required_fields = ["claimed_score", "models_used", "submission_date", "notes"]
    for field in required_fields:
        assert field in info, f"Required field {field} missing from info.json"
    
    assert isinstance(info["claimed_score"], (int, float)), "claimed_score should be a number"
    assert isinstance(info["models_used"], list), "models_used should be a list"
    assert isinstance(info["submission_date"], str), "submission_date should be a string"
    assert isinstance(info["notes"], str), "notes should be a string"
    
    print("✓ info.json has correct structure")

def test_requirements_txt():
    """Test that requirements.txt exists and has content"""
    print("\nTesting requirements.txt...")
    
    assert os.path.exists("requirements.txt"), "requirements.txt not found"
    
    with open("requirements.txt", "r") as f:
        content = f.read().strip()
    
    assert len(content) > 0, "requirements.txt should not be empty"
    
    # Check for some expected dependencies
    expected_deps = ["torch", "transformers", "gitpython", "pytest"]
    for dep in expected_deps:
        assert dep in content.lower(), f"Expected dependency {dep} not found in requirements.txt"
    
    print("✓ requirements.txt has content and expected dependencies")

if __name__ == "__main__":
    print("Running SWE-Bench submission tests...\n")
    
    try:
        test_submission_structure()
        test_info_json()
        test_requirements_txt()
        test_main_run_function()
        
        print("\n🎉 All tests passed! Submission is ready.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)