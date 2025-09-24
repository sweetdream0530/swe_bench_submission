#!/usr/bin/env python3
"""
Extract predictions from existing trajectory files.
This script reads trajectory files and creates a predictions.jsonl file.
"""

import json
import os
from pathlib import Path

def extract_predictions_from_trajectories(traj_dir, output_file):
    """Extract predictions from trajectory files."""
    traj_path = Path(traj_dir)
    predictions = []
    
    print(f"🔍 Scanning trajectory directory: {traj_path}")
    
    trajectory_files = list(traj_path.glob("*.jsonl"))
    print(f"📁 Found {len(trajectory_files)} trajectory files")
    
    for traj_file in trajectory_files:
        try:
            # Extract instance_id from filename
            instance_id = traj_file.stem  # e.g., "astropy__astropy-13398"
            
            # Read the trajectory file
            with open(traj_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                print(f"⚠️ Empty trajectory file: {traj_file.name}")
                continue
            
            # Look for the final result in the last line
            final_line = lines[-1].strip()
            if not final_line:
                print(f"⚠️ Empty last line in: {traj_file.name}")
                continue
            
            try:
                final_entry = json.loads(final_line)
            except json.JSONDecodeError:
                print(f"⚠️ Invalid JSON in last line of: {traj_file.name}")
                continue
            
            # Check if this is a final result
            if final_entry.get("event") == "final_result":
                success = final_entry.get("success", False)
                final_patch = final_entry.get("final_patch", "")
                
                if success and final_patch:
                    prediction = {
                        "instance_id": instance_id,
                        "model_name_or_path": "enhanced-swe-bench-submission",
                        "model_patch": final_patch
                    }
                    predictions.append(prediction)
                    print(f"✅ Extracted prediction for {instance_id}")
                else:
                    print(f"⚠️ No successful patch found for {instance_id}")
            else:
                print(f"⚠️ No final result found for {instance_id}")
                
        except Exception as e:
            print(f"❌ Error processing {traj_file.name}: {e}")
    
    # Write predictions to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + '\n')
    
    print(f"💾 Saved {len(predictions)} predictions to {output_path}")
    return len(predictions)

if __name__ == "__main__":
    import sys
    
    traj_dir = "results/trajectories"
    output_file = "results/predictions/predictions.jsonl"
    
    if len(sys.argv) > 1:
        traj_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print("🚀 Extracting predictions from trajectory files...")
    count = extract_predictions_from_trajectories(traj_dir, output_file)
    print(f"🎉 Extracted {count} predictions successfully!")
