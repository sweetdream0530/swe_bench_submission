#!/usr/bin/env python3
"""
Test script for Astropy NDDataRef mask propagation issue
"""

import os
import sys
import subprocess
import tempfile
import shutil

# Add the submission directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from main import run

def clone_astropy_repo():
    """Clone the actual Astropy repository from GitHub"""
    test_repo = "/tmp/astropy_real_repo"
    
    # Clean up if exists
    if os.path.exists(test_repo):
        shutil.rmtree(test_repo)
    
    print(f"Cloning Astropy repository from GitHub...")
    
    # Clone the repository
    result = subprocess.run([
        "git", "clone", 
        "https://github.com/astropy/astropy.git", 
        test_repo
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception(f"Failed to clone Astropy repository: {result.stderr}")
    
    print(f"✓ Successfully cloned Astropy repository to {test_repo}")
    
    # Change to the repository directory
    os.chdir(test_repo)
    
    # Checkout the specific commit mentioned in the issue (if available)
    # The issue mentions v5.3, so let's try to find a relevant commit
    try:
        # Try to checkout a commit from around v5.3 timeframe
        subprocess.run([
            "git", "checkout", "b16c7d12ccbc7b2d20364b89fb44285bcbfede54"
        ], capture_output=True, text=True, check=True)
        print("✓ Checked out specific commit")
    except subprocess.CalledProcessError:
        print("⚠ Could not checkout specific commit, using latest")
    
    # Configure git user
    subprocess.run(["git", "config", "user.email", "test@example.com"], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], check=True, capture_output=True)
    
    return test_repo

def test_astropy_issue():
    """Test the Astropy NDDataRef mask propagation issue"""
    print("Testing Astropy NDDataRef mask propagation issue...")
    
    # Set testing mode
    os.environ["TESTING_MODE"] = "true"
    
    # Clone the actual Astropy repository
    test_repo = clone_astropy_repo()
    
    try:
        # Change back to submission directory
        os.chdir(os.path.dirname(__file__))
        
        # Test data from the issue
        problem_statement = """
In v5.3, NDDataRef mask propagation fails when one of the operand does not have a mask

### Description
This applies to v5.3. It looks like when one of the operand does not have a mask, the mask propagation when doing arithmetic, in particular with `handle_mask=np.bitwise_or` fails. This is not a problem in v5.2.

### Expected behavior
When one of the operand does not have mask, the mask that exists should just be copied over to the output.

### How to Reproduce
```python
import numpy as np
from astropy.nddata import NDDataRef

array = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
mask = np.array([[0, 1, 64], [8, 0, 1], [2, 1, 0]])

nref_nomask = NDDataRef(array)
nref_mask = NDDataRef(array, mask=mask)

# This should work but fails with TypeError
result = nref_mask.multiply(1., handle_mask=np.bitwise_or)
```

The error occurs because the code tries to do bitwise_or between an integer and None.
"""
        
        # Run the SWE-Bench submission
        patch = run(
            repo_path=test_repo,
            instance_id="astropy__astropy-14995",
            base_commit="b16c7d12ccbc7b2d20364b89fb44285bcbfede54",
            problem_statement=problem_statement,
            version="5.2",
            traj_dir="/tmp/astropy_test_trajectories",
            temp_dir="/tmp/astropy_test_temp",
            log_path="/tmp/astropy_test_log.txt"
        )
        
        print(f"✓ Generated patch length: {len(patch)} characters")
        print("✓ Patch generated successfully")
        
        # Verify the patch contains relevant changes
        if "mask" in patch.lower() and "nddata" in patch.lower():
            print("✓ Patch appears to address the NDDataRef mask issue")
        else:
            print("⚠ Patch may not address the specific mask issue")
        
        # Print the patch for inspection
        print("\n" + "="*50)
        print("GENERATED PATCH:")
        print("="*50)
        print(patch)
        print("="*50)
        
    finally:
        # Cleanup (comment out to keep repository for inspection)
        # if os.path.exists(test_repo):
        #     shutil.rmtree(test_repo)
        if os.path.exists("/tmp/astropy_test_trajectories"):
            shutil.rmtree("/tmp/astropy_test_trajectories")
        if os.path.exists("/tmp/astropy_test_temp"):
            shutil.rmtree("/tmp/astropy_test_temp")
        if os.path.exists("/tmp/astropy_test_log.txt"):
            os.remove("/tmp/astropy_test_log.txt")
        
        print(f"\nRepository kept at: {test_repo}")
        print("You can inspect the cloned Astropy repository there.")

if __name__ == "__main__":
    test_astropy_issue()
