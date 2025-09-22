#!/usr/bin/env python3
"""
Test script to verify repository cloning functionality in main.py
"""

import os
import sys
import subprocess
import shutil

# Add the submission directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from main import run

def test_github_url_cloning():
    """Test cloning from GitHub URL"""
    print("Testing GitHub URL cloning...")
    
    # Test parameters
    repo_path = "astropy/astropy"
    instance_id = "test_github_clone"
    base_commit = "b16c7d12ccbc7b2d20364b89fb44285bcbfede54"
    problem_statement = "Test problem for GitHub cloning"
    version = "5.2"
    traj_dir = "/tmp/test_trajectories"
    temp_dir = "/tmp/test_temp"
    log_path = "/tmp/test_log.txt"
    
    # Cleanup
    for d in [traj_dir, temp_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
    if os.path.exists(log_path):
        os.remove(log_path)
    
    try:
        # Run the main function
        patch = run(
            repo_path=repo_path,
            instance_id=instance_id,
            base_commit=base_commit,
            problem_statement=problem_statement,
            version=version,
            traj_dir=traj_dir,
            temp_dir=temp_dir,
            log_path=log_path
        )
        
        assert isinstance(patch, str), "run function should return a string"
        assert len(patch) > 0, "run function should return a non-empty patch"
        print(f"✓ GitHub URL cloning test passed - patch length: {len(patch)} characters")
        
    except Exception as e:
        print(f"❌ GitHub URL cloning test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        for d in [traj_dir, temp_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
        if os.path.exists(log_path):
            os.remove(log_path)

def test_repo_name_cloning():
    """Test cloning from repository name pattern"""
    print("\nTesting repository name pattern cloning...")
    
    # Test parameters
    repo_path = "astropy/astropy"  # This should trigger GitHub cloning
    instance_id = "test_repo_name_clone"
    base_commit = "b16c7d12ccbc7b2d20364b89fb44285bcbfede54"
    problem_statement = "Test problem for repo name cloning"
    version = "5.2"
    traj_dir = "/tmp/test_trajectories2"
    temp_dir = "/tmp/test_temp2"
    log_path = "/tmp/test_log2.txt"
    
    # Cleanup
    for d in [traj_dir, temp_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
    if os.path.exists(log_path):
        os.remove(log_path)
    
    try:
        # Run the main function
        patch = run(
            repo_path=repo_path,
            instance_id=instance_id,
            base_commit=base_commit,
            problem_statement=problem_statement,
            version=version,
            traj_dir=traj_dir,
            temp_dir=temp_dir,
            log_path=log_path
        )
        
        assert isinstance(patch, str), "run function should return a string"
        assert len(patch) > 0, "run function should return a non-empty patch"
        print(f"✓ Repository name cloning test passed - patch length: {len(patch)} characters")
        
    except Exception as e:
        print(f"❌ Repository name cloning test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        for d in [traj_dir, temp_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
        if os.path.exists(log_path):
            os.remove(log_path)

if __name__ == "__main__":
    print("Testing repository cloning functionality...")
    test_github_url_cloning()
    test_repo_name_cloning()
    print("\nRepository cloning tests completed!")
