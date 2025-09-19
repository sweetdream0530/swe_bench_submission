import os
import json
from datetime import datetime
from typing import Dict, Any, List

class TrajectoryLogger:
    def __init__(self, traj_dir: str):
        self.traj_dir = traj_dir
        os.makedirs(self.traj_dir, exist_ok=True)
    
    def log_step(self, instance_id: str, step_number: int, action: str, details: Dict[str, Any]):
        """
        Logs a single step of the agent's trajectory.
        
        Args:
            instance_id (str): The unique ID of the current task instance.
            step_number (int): The sequential number of the current step.
            action (str): A brief description of the action taken (e.g., "read_file", "apply_edit", "run_tests").
            details (Dict[str, Any]): A dictionary containing detailed information about the action,
                                     such as tool calls, LLM inputs/outputs, file paths, etc.
        """
        log_file_path = os.path.join(self.traj_dir, f"{instance_id}.jsonl")
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step_number,
            "action": action,
            "details": details
        }
        
        with open(log_file_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def log_final_result(self, instance_id: str, final_patch: str, success: bool, message: str = ""):
        """
        Logs the final result of the agent's attempt for a given instance.
        
        Args:
            instance_id (str): The unique ID of the current task instance.
            final_patch (str): The generated Git patch string.
            success (bool): True if the task was considered successful, False otherwise.
            message (str): An optional message about the final outcome.
        """
        log_file_path = os.path.join(self.traj_dir, f"{instance_id}.jsonl")
        
        final_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "final_result",
            "success": success,
            "message": message,
            "final_patch": final_patch
        }
        
        with open(log_file_path, "a") as f:
            f.write(json.dumps(final_entry) + "\n")
    
    def get_trajectory(self, instance_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves the full trajectory for a given instance.
        
        Args:
            instance_id (str): The unique ID of the task instance.
        
        Returns:
            List[Dict[str, Any]]: A list of all logged entries for the instance.
        """
        log_file_path = os.path.join(self.traj_dir, f"{instance_id}.jsonl")
        trajectory = []
        if os.path.exists(log_file_path):
            with open(log_file_path, "r") as f:
                for line in f:
                    try:
                        trajectory.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # Skip malformed lines
        return trajectory

# Example Usage:
if __name__ == "__main__":
    logger = TrajectoryLogger(traj_dir="./trajectories")
    instance_id = "example_instance_001"
    
    logger.log_step(instance_id, 1, "initial_analysis", {
        "llm_input": "problem statement...",
        "llm_output": "plan..."
    })
    logger.log_step(instance_id, 2, "read_file", {
        "file_path": "src/main.py",
        "content_preview": "def func..."
    })
    logger.log_step(instance_id, 3, "apply_edit", {
        "file_path": "src/main.py",
        "search": "old_code",
        "replace": "new_code"
    })
    
    dummy_patch = "--- a/src/main.py\n+++ b/src/main.py\n@@ -1,2 +1,3 @@\n-old_code\n+new_code\n"
    logger.log_final_result(instance_id, dummy_patch, True, "Successfully fixed the issue.")
    
    print(f"Trajectory for {instance_id}:\n{json.dumps(logger.get_trajectory(instance_id), indent=2)}")