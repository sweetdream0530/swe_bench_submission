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
        self.session_id = f"session_{int(time.time())}_{id(self)}"
    
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
        Logs LLM interactions with detailed information including all inputs and outputs.
        
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
            "metadata": metadata or {},
            "interaction_id": f"{instance_id}_{step_number}_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "model_version": metadata.get('model_version', 'unknown') if metadata else 'unknown',
            "temperature": metadata.get('temperature', 0.0) if metadata else 0.0,
            "max_tokens": metadata.get('max_tokens', None) if metadata else None,
            "stop_sequences": metadata.get('stop_sequences', []) if metadata else [],
            "system_prompt": self._extract_system_prompt(messages),
            "user_prompt": self._extract_user_prompt(messages),
            "response_length": len(response),
            "response_preview": response[:200] + "..." if len(response) > 200 else response
        }
        
        self.log_step(instance_id, step_number, "llm_interaction", details)
    
    def log_model_selection_decision(self, instance_id: str, step_number: int,
                                    available_models: List[str], selected_model: str,
                                    selection_reason: str, performance_metrics: Dict[str, Any] = None):
        """
        Logs model selection decisions with reasoning.
        
        Args:
            instance_id (str): The unique ID of the current task instance.
            step_number (int): The sequential number of the current step.
            available_models (List[str]): List of available models.
            selected_model (str): The selected model.
            selection_reason (str): Reason for selecting this model.
            performance_metrics (Dict[str, Any]): Performance metrics for the decision.
        """
        details = {
            "decision_type": "model_selection",
            "available_models": available_models,
            "selected_model": selected_model,
            "selection_reason": selection_reason,
            "performance_metrics": performance_metrics or {},
            "decision_timestamp": datetime.now().isoformat(),
            "confidence_score": performance_metrics.get('confidence', 0.0) if performance_metrics else 0.0
        }
        
        self.log_step(instance_id, step_number, "model_selection", details)
    
    def log_strategy_selection_decision(self, instance_id: str, step_number: int,
                                       available_strategies: List[str], selected_strategy: str,
                                       selection_reason: str, context: Dict[str, Any] = None):
        """
        Logs strategy selection decisions with reasoning.
        
        Args:
            instance_id (str): The unique ID of the current task instance.
            step_number (int): The sequential number of the current step.
            available_strategies (List[str]): List of available strategies.
            selected_strategy (str): The selected strategy.
            selection_reason (str): Reason for selecting this strategy.
            context (Dict[str, Any]): Context information for the decision.
        """
        details = {
            "decision_type": "strategy_selection",
            "available_strategies": available_strategies,
            "selected_strategy": selected_strategy,
            "selection_reason": selection_reason,
            "context": context or {},
            "decision_timestamp": datetime.now().isoformat(),
            "decision_id": f"{instance_id}_{step_number}_strategy_{int(time.time())}"
        }
        
        self.log_step(instance_id, step_number, "strategy_selection", details)
    
    def log_filtering_decision(self, instance_id: str, step_number: int,
                              filter_type: str, input_data: Any, filtered_data: Any,
                              filter_criteria: Dict[str, Any], decision_reason: str):
        """
        Logs filtering decisions with detailed information.
        
        Args:
            instance_id (str): The unique ID of the current task instance.
            step_number (int): The sequential number of the current step.
            filter_type (str): Type of filtering applied.
            input_data (Any): Data before filtering.
            filtered_data (Any): Data after filtering.
            filter_criteria (Dict[str, Any]): Criteria used for filtering.
            decision_reason (str): Reason for the filtering decision.
        """
        details = {
            "filter_type": filter_type,
            "input_data": str(input_data)[:1000],  # Truncate large data
            "filtered_data": str(filtered_data)[:1000],
            "filter_criteria": filter_criteria,
            "decision_reason": decision_reason,
            "filter_timestamp": datetime.now().isoformat(),
            "input_size": len(str(input_data)) if input_data else 0,
            "output_size": len(str(filtered_data)) if filtered_data else 0,
            "filter_efficiency": len(str(filtered_data)) / max(1, len(str(input_data))) if input_data else 0
        }
        
        self.log_step(instance_id, step_number, "filtering_decision", details)
    
    def log_path_selection_decision(self, instance_id: str, step_number: int,
                                  available_paths: List[Dict[str, Any]], selected_path: Dict[str, Any],
                                  selection_reason: str, path_metrics: Dict[str, Any] = None):
        """
        Logs path selection decisions in self-consistency reasoning.
        
        Args:
            instance_id (str): The unique ID of the current task instance.
            step_number (int): The sequential number of the current step.
            available_paths (List[Dict[str, Any]]): List of available reasoning paths.
            selected_path (Dict[str, Any]): The selected path.
            selection_reason (str): Reason for selecting this path.
            path_metrics (Dict[str, Any]): Metrics for path evaluation.
        """
        details = {
            "decision_type": "path_selection",
            "available_paths": available_paths,
            "selected_path": selected_path,
            "selection_reason": selection_reason,
            "path_metrics": path_metrics or {},
            "decision_timestamp": datetime.now().isoformat(),
            "total_paths": len(available_paths),
            "selected_path_id": selected_path.get('path_id', 'unknown')
        }
        
        self.log_step(instance_id, step_number, "path_selection", details)
    
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
    
    def _extract_system_prompt(self, messages: List[Dict]) -> str:
        """Extract system prompt from messages."""
        for message in messages:
            if message.get('role') == 'system':
                return message.get('content', '')
        return ''
    
    def _extract_user_prompt(self, messages: List[Dict]) -> str:
        """Extract user prompt from messages."""
        user_prompts = []
        for message in messages:
            if message.get('role') == 'user':
                user_prompts.append(message.get('content', ''))
        return '\n'.join(user_prompts)
    
    def _get_session_id(self) -> str:
        """Get the unique session ID for this logger instance."""
        return self.session_id
    
    def log_comprehensive_interaction(self, instance_id: str, step_number: int,
                                   interaction_data: Dict[str, Any]):
        """
        Logs a comprehensive interaction with all available data.
        
        Args:
            instance_id (str): The unique ID of the current task instance.
            step_number (int): The sequential number of the current step.
            interaction_data (Dict[str, Any]): Complete interaction data.
        """
        details = {
            "interaction_type": interaction_data.get('type', 'unknown'),
            "model_name": interaction_data.get('model_name', 'unknown'),
            "input_data": interaction_data.get('input_data', {}),
            "output_data": interaction_data.get('output_data', {}),
            "metadata": interaction_data.get('metadata', {}),
            "performance": interaction_data.get('performance', {}),
            "decisions": interaction_data.get('decisions', []),
            "filtering_applied": interaction_data.get('filtering_applied', []),
            "timestamp": datetime.now().isoformat(),
            "interaction_id": f"{instance_id}_{step_number}_{int(time.time())}"
        }
        
        self.log_step(instance_id, step_number, "comprehensive_interaction", details)

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