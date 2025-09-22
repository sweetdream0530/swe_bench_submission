import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import time
import threading
from collections import defaultdict

class TrajectoryLogger:
    def __init__(self, traj_dir: str):
        self.traj_dir = traj_dir
        os.makedirs(self.traj_dir, exist_ok=True)
        self.log_lock = threading.Lock()
        self.step_counts = defaultdict(int)
        self.performance_metrics = defaultdict(list)
    
    def log_step(self, instance_id: str, step_number: int, action: str, details: Dict[str, Any]):
        """
        Logs a single step of the agent's trajectory with enhanced details.
        
        Args:
            instance_id (str): The unique ID of the current task instance.
            step_number (int): The sequential number of the current step.
            action (str): A brief description of the action taken (e.g., "read_file", "apply_edit", "run_tests").
            details (Dict[str, Any]): A dictionary containing detailed information about the action,
                                     such as tool calls, LLM inputs/outputs, file paths, etc.
        """
        with self.log_lock:
            log_file_path = os.path.join(self.traj_dir, f"{instance_id}.jsonl")
            
            # Add performance metrics
            step_start_time = details.get('start_time', time.time())
            step_duration = time.time() - step_start_time
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "step": step_number,
                "action": action,
                "details": details,
                "performance": {
                    "duration_seconds": step_duration,
                    "memory_usage_mb": self._get_memory_usage(),
                    "step_type": self._classify_step_type(action)
                },
                "metadata": {
                    "instance_id": instance_id,
                    "total_steps": self.step_counts[instance_id] + 1,
                    "session_id": self._get_session_id()
                }
            }
            
            # Track performance metrics
            self.performance_metrics[action].append(step_duration)
            self.step_counts[instance_id] += 1
            
            with open(log_file_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
    
    def log_llm_interaction(self, instance_id: str, step_number: int, 
                           model_name: str, messages: List[Dict], 
                           response: str, metadata: Dict[str, Any] = None):
        """
        Logs LLM interactions with detailed information.
        
        Args:
            instance_id (str): The unique ID of the current task instance.
            step_number (int): The sequential number of the current step.
            model_name (str): The name of the LLM model used.
            messages (List[Dict]): The input messages sent to the LLM.
            response (str): The response received from the LLM.
            metadata (Dict[str, Any]): Additional metadata about the interaction.
        """
        details = {
            "model_name": model_name,
            "input_messages": messages,
            "response": response,
            "input_tokens": self._estimate_tokens(messages),
            "output_tokens": self._estimate_tokens([response]),
            "metadata": metadata or {}
        }
        
        self.log_step(instance_id, step_number, "llm_interaction", details)
    
    def log_tool_call(self, instance_id: str, step_number: int,
                     tool_name: str, tool_args: Dict[str, Any],
                     tool_result: Any, execution_time: float):
        """
        Logs tool calls with detailed information.
        
        Args:
            instance_id (str): The unique ID of the current task instance.
            step_number (int): The sequential number of the current step.
            tool_name (str): The name of the tool called.
            tool_args (Dict[str, Any]): The arguments passed to the tool.
            tool_result (Any): The result returned by the tool.
            execution_time (float): The time taken to execute the tool.
        """
        details = {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_result": str(tool_result)[:1000],  # Truncate large results
            "execution_time": execution_time,
            "success": not isinstance(tool_result, Exception)
        }
        
        self.log_step(instance_id, step_number, "tool_call", details)
    
    def log_self_consistency_step(self, instance_id: str, step_number: int,
                                 path_id: int, approach: str, 
                                 reasoning_steps: List[str], 
                                 final_answer: str, confidence: float):
        """
        Logs self-consistency reasoning steps.
        
        Args:
            instance_id (str): The unique ID of the current task instance.
            step_number (int): The sequential number of the current step.
            path_id (int): The ID of the reasoning path.
            approach (str): The reasoning approach used.
            reasoning_steps (List[str]): The steps in the reasoning process.
            final_answer (str): The final answer from this path.
            confidence (float): The confidence score for this path.
        """
        details = {
            "path_id": path_id,
            "approach": approach,
            "reasoning_steps": reasoning_steps,
            "final_answer": final_answer,
            "confidence": confidence,
            "step_type": "self_consistency"
        }
        
        self.log_step(instance_id, step_number, "self_consistency", details)
    
    def log_intelligent_search(self, instance_id: str, step_number: int,
                              query: str, strategies_used: List[str],
                              search_results: Dict[str, Any],
                              fused_results: Dict[str, Any]):
        """
        Logs intelligent search operations.
        
        Args:
            instance_id (str): The unique ID of the current task instance.
            step_number (int): The sequential number of the current step.
            query (str): The search query used.
            strategies_used (List[str]): The search strategies employed.
            search_results (Dict[str, Any]): Results from individual strategies.
            fused_results (Dict[str, Any]): The fused search results.
        """
        details = {
            "query": query,
            "strategies_used": strategies_used,
            "search_results": search_results,
            "fused_results": fused_results,
            "step_type": "intelligent_search"
        }
        
        self.log_step(instance_id, step_number, "intelligent_search", details)
    
    def log_final_result(self, instance_id: str, final_patch: str, success: bool, message: str = ""):
        """
        Logs the final result of the agent's attempt for a given instance.
        
        Args:
            instance_id (str): The unique ID of the current task instance.
            final_patch (str): The generated Git patch string.
            success (bool): True if the task was considered successful, False otherwise.
            message (str): An optional message about the final outcome.
        """
        with self.log_lock:
            log_file_path = os.path.join(self.traj_dir, f"{instance_id}.jsonl")
            
            # Calculate final performance metrics
            total_steps = self.step_counts.get(instance_id, 0)
            avg_step_time = sum(self.performance_metrics.get('llm_interaction', [0])) / max(1, len(self.performance_metrics.get('llm_interaction', [1])))
            
            final_entry = {
                "timestamp": datetime.now().isoformat(),
                "event": "final_result",
                "success": success,
                "message": message,
                "final_patch": final_patch,
                "performance_summary": {
                    "total_steps": total_steps,
                    "avg_step_time": avg_step_time,
                    "total_llm_interactions": len(self.performance_metrics.get('llm_interaction', [])),
                    "total_tool_calls": len(self.performance_metrics.get('tool_call', [])),
                    "memory_usage_mb": self._get_memory_usage()
                },
                "metadata": {
                    "instance_id": instance_id,
                    "session_id": self._get_session_id(),
                    "patch_length": len(final_patch) if final_patch else 0
                }
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
    
    def get_performance_summary(self, instance_id: str) -> Dict[str, Any]:
        """
        Get performance summary for a specific instance.
        
        Args:
            instance_id (str): The unique ID of the task instance.
        
        Returns:
            Dict[str, Any]: Performance summary including timing and resource usage.
        """
        trajectory = self.get_trajectory(instance_id)
        
        llm_interactions = [step for step in trajectory if step.get('action') == 'llm_interaction']
        tool_calls = [step for step in trajectory if step.get('action') == 'tool_call']
        
        return {
            "total_steps": len(trajectory),
            "llm_interactions": len(llm_interactions),
            "tool_calls": len(tool_calls),
            "self_consistency_steps": len([step for step in trajectory if step.get('action') == 'self_consistency']),
            "intelligent_search_steps": len([step for step in trajectory if step.get('action') == 'intelligent_search']),
            "avg_step_duration": sum(step.get('performance', {}).get('duration_seconds', 0) for step in trajectory) / max(1, len(trajectory)),
            "total_duration": sum(step.get('performance', {}).get('duration_seconds', 0) for step in trajectory)
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _classify_step_type(self, action: str) -> str:
        """Classify the step type based on action."""
        if action == "llm_interaction":
            return "reasoning"
        elif action == "tool_call":
            return "execution"
        elif action == "self_consistency":
            return "consensus"
        elif action == "intelligent_search":
            return "search"
        else:
            return "general"
    
    def _estimate_tokens(self, text_list: List[Any]) -> int:
        """Estimate token count for text."""
        total_text = " ".join(str(item) for item in text_list)
        return len(total_text.split())  # Rough estimation
    
    def _get_session_id(self) -> str:
        """Generate a session ID."""
        return f"session_{int(time.time())}"

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