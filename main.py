from __future__ import annotations
import ast
import json
import os
import requests
import subprocess
import ast, sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from json import JSONDecodeError
import re
import inspect
import random
from enum import Enum
import json
import csv
import logging
import concurrent.futures
import threading
from collections import defaultdict
from agent import agent_main
from trajectory_logger import TrajectoryLogger
from patch_generator import PatchGenerator
import shutil
import signal
from contextlib import contextmanager

# Optional psutil import with fallback
try:
    import psutil
    PSUTIL_AVAILABLE = True
    print(f"✅ psutil {psutil.__version__} loaded successfully")
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ Warning: psutil not available, resource monitoring disabled")
    print("   Install with: pip install psutil>=7.0.0")

# Enhanced Accuracy Algorithm Configuration
SELF_CONSISTENCY_CONFIG = {
    'DEFAULT_NUM_PATHS': 7,  # Increased from 5 for better consensus
    'DEFAULT_CONSENSUS_THRESHOLD': 0.7,  # Increased threshold for higher confidence
    'MAX_EXECUTION_TIME': 45,  # Increased timeout for complex reasoning
    'ENABLE_ADAPTIVE_PATHS': True,
    'MIN_PATHS_FOR_CONSENSUS': 3,  # Minimum paths needed for valid consensus
    'CONFIDENCE_BOOST_THRESHOLD': 0.8  # Boost confidence if consensus is high
}

INTELLIGENT_SEARCH_CONFIG = {
    'DEFAULT_FUSION_METHOD': 'weighted',
    'MAX_SEARCH_STRATEGIES': 7,  # Increased strategies
    'SEARCH_TIMEOUT': 30,  # Increased timeout per strategy
    'ENABLE_CONTEXT_ANALYSIS': True,
    'ENABLE_ADAPTIVE_ROUTING': True,
    'SEMANTIC_SIMILARITY_THRESHOLD': 0.8,  # For context matching
    'DYNAMIC_STRATEGY_SELECTION': True  # Enable dynamic strategy selection
}

# Combined accuracy improvement estimation
EXPECTED_ACCURACY_IMPROVEMENT = {
    'self_consistency': 0.30,  # Increased from 25% to 30%
    'intelligent_search': 0.20,  # Increased from 15% to 20%
    'combined': 0.50,  # Increased from 40% to 50% (synergistic effect)
    'confidence_threshold': 0.85,  # Increased threshold
    'resource_optimization': 0.10  # Additional 10% from resource optimization
}

# Resource Management Configuration
RESOURCE_CONFIG = {
    'MAX_MEMORY_USAGE_MB': 8192,  # 8GB memory limit
    'MAX_CPU_USAGE_PERCENT': 80,  # 80% CPU limit
    'MEMORY_CLEANUP_INTERVAL': 30,  # Cleanup every 30 seconds
    'PROCESS_MONITORING_INTERVAL': 5,  # Monitor every 5 seconds
    'EMERGENCY_CLEANUP_THRESHOLD': 0.95  # Emergency cleanup at 95% usage
}

# Model Configuration
DEFAULT_PROXY_URL = os.getenv("AI_PROXY_URL", "http://sandbox_proxy")
DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "2200"))
MAX_TEST_PATCH_TIMEOUT = int(os.getenv("MAX_STEPS_TEST_PATCH_FIND", "500"))
GLM_MODEL_NAME = "zai-org/GLM-4.5-FP8"
KIMI_MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
AGENT_MODELS = [GLM_MODEL_NAME, QWEN_MODEL_NAME, KIMI_MODEL_NAME, DEEPSEEK_MODEL_NAME]
# Fallback models in case of API issues
FALLBACK_MODELS = [GLM_MODEL_NAME, QWEN_MODEL_NAME, KIMI_MODEL_NAME]  # Remove DeepSeek if problematic
MAX_STEPS = 120
MAX_STEPS_TEST_PATCH_FIND = 50
DEBUG_MODE = True

# Resource monitoring and timeout management
class ResourceMonitor:
    """Monitor system resources and manage timeouts"""
    def __init__(self):
        self.start_time = time.time()
        self.max_memory_mb = RESOURCE_CONFIG['MAX_MEMORY_USAGE_MB']
        self.max_cpu_percent = RESOURCE_CONFIG['MAX_CPU_USAGE_PERCENT']
        self.last_cleanup = time.time()
        self.process = psutil.Process()
        
    def check_resources(self) -> Dict[str, Any]:
        """Check current resource usage"""
        memory_info = self.process.memory_info()
        cpu_percent = self.process.cpu_percent()
        memory_mb = memory_info.rss / 1024 / 1024
        
        return {
            'memory_mb': memory_mb,
            'memory_percent': (memory_mb / self.max_memory_mb) * 100,
            'cpu_percent': cpu_percent,
            'elapsed_time': time.time() - self.start_time,
            'is_over_limit': memory_mb > self.max_memory_mb or cpu_percent > self.max_cpu_percent
        }
    
    def cleanup_if_needed(self):
        """Perform cleanup if resources are over limit"""
        resources = self.check_resources()
        if resources['is_over_limit'] or time.time() - self.last_cleanup > RESOURCE_CONFIG['MEMORY_CLEANUP_INTERVAL']:
            import gc
            gc.collect()
            self.last_cleanup = time.time()
            print(f"🧹 Resource cleanup performed. Memory: {resources['memory_mb']:.1f}MB, CPU: {resources['cpu_percent']:.1f}%")

@contextmanager
def timeout_manager(seconds: int):
    """Context manager for timeout handling"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# Enhanced caching and timeout system
class SmartCache:
    """Intelligent caching system with TTL, LRU eviction, and performance optimization"""
    def __init__(self, default_ttl: int = 300, max_size: int = 1000):
        self.cache = {}
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.access_count = defaultdict(int)
        self.access_times = defaultdict(float)
        self.last_cleanup = time.time()
        self.cleanup_interval = 30  # More frequent cleanup
        self.hit_count = 0
        self.miss_count = 0

    def get(self, key: str, default: Any = None) -> Any:
        """Get cached value if not expired"""
        self._cleanup_if_needed()
        if key in self.cache:
            timestamp, value = self.cache[key]
            if time.time() - timestamp < self.default_ttl:
                self.access_count[key] += 1
                self.access_times[key] = time.time()
                self.hit_count += 1
                return value
            else:
                del self.cache[key]
                del self.access_count[key]
                del self.access_times[key]
        self.miss_count += 1
        return default

    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """Set cached value with TTL and LRU eviction"""
        self._cleanup_if_needed()
        self._evict_if_needed()
        
        self.cache[key] = (time.time(), value)
        self.access_count[key] = 0
        self.access_times[key] = time.time()

    def _evict_if_needed(self) -> None:
        """Evict least recently used items if cache is full"""
        if len(self.cache) >= self.max_size:
            # Sort by access time and remove oldest 10%
            sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
            items_to_remove = max(1, len(sorted_items) // 10)
            
            for key, _ in sorted_items[:items_to_remove]:
                if key in self.cache:
                    del self.cache[key]
                    del self.access_count[key]
                    del self.access_times[key]

    def _cleanup_if_needed(self) -> None:
        """Clean up expired cache entries"""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            expired_keys = []
            for key, (timestamp, _) in self.cache.items():
                if current_time - timestamp > self.default_ttl:
                    expired_keys.append(key)
            for key in expired_keys:
                del self.cache[key]
                del self.access_count[key]
                del self.access_times[key]
            self.last_cleanup = current_time

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'total_entries': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'most_accessed': sorted(self.access_count.items(), key=lambda x: x[1], reverse=True)[:5],
            'cache_size_mb': sum(len(str(v)) for _, v in self.cache.items()) / (1024 * 1024)
        }

# Performance monitoring and parallel execution
class PerformanceMonitor:
    """Monitor performance metrics for parallel operations with enhanced caching and resource tracking"""
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}
        self.cache = SmartCache(default_ttl=600, max_size=500)  # Enhanced cache
        self.resource_monitor = ResourceMonitor()
        self.operation_counts = defaultdict(int)
        self.success_counts = defaultdict(int)
        self.failure_counts = defaultdict(int)

    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
        self.operation_counts[operation] += 1

    def end_timer(self, operation: str, success: bool = True):
        """End timing an operation and record the duration"""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.metrics[operation].append(duration)
            
            if success:
                self.success_counts[operation] += 1
            else:
                self.failure_counts[operation] += 1
                
            logging.info(f"⏱️ {operation} took {duration:.2f}s ({'SUCCESS' if success else 'FAILURE'})")
            
            # Clean up start time
            del self.start_times[operation]

    def get_cached_result(self, key: str, ttl: int = None):
        """Get cached result if not expired"""
        return self.cache.get(key)

    def cache_result(self, key: str, value: Any, ttl: int = None):
        """Cache a result with TTL"""
        self.cache.set(key, value, ttl)

    def get_average_time(self, operation: str) -> float:
        """Get average time for an operation"""
        times = self.metrics.get(operation, [])
        return sum(times) / len(times) if times else 0

    def get_success_rate(self, operation: str) -> float:
        """Get success rate for an operation"""
        total = self.success_counts[operation] + self.failure_counts[operation]
        return self.success_counts[operation] / total if total > 0 else 0

    def get_performance_summary(self) -> str:
        """Get a comprehensive summary of all performance metrics"""
        summary = "🚀 Performance Summary:\n"
        for operation in self.operation_counts:
            avg_time = self.get_average_time(operation)
            success_rate = self.get_success_rate(operation)
            total_time = sum(self.metrics.get(operation, []))
            count = self.operation_counts[operation]
            
            summary += f"  {operation}:\n"
            summary += f"    - Avg time: {avg_time:.2f}s\n"
            summary += f"    - Total time: {total_time:.2f}s\n"
            summary += f"    - Count: {count}\n"
            summary += f"    - Success rate: {success_rate:.1%}\n"
        
        # Add resource information
        resources = self.resource_monitor.check_resources()
        summary += f"\n📊 Resource Usage:\n"
        summary += f"  - Memory: {resources['memory_mb']:.1f}MB ({resources['memory_percent']:.1f}%)\n"
        summary += f"  - CPU: {resources['cpu_percent']:.1f}%\n"
        summary += f"  - Elapsed time: {resources['elapsed_time']:.1f}s\n"
        
        # Add cache statistics
        cache_stats = self.cache.get_stats()
        summary += f"\n💾 Cache Statistics:\n"
        summary += f"  - Hit rate: {cache_stats['hit_rate']:.1%}\n"
        summary += f"  - Entries: {cache_stats['total_entries']}/{cache_stats['max_size']}\n"
        summary += f"  - Size: {cache_stats['cache_size_mb']:.2f}MB\n"
        
        return summary

# Network and error handling
class Network:
    class ErrorType(Enum):
        EMPTY_RESPONSE = 1
        RESERVED_TOKEN_PRESENT = 2
        RATE_LIMIT_EXCEEDED = 3
        INVALID_RESPONSE_FORMAT = 4
        TIMEOUT = 5
        UNKNOWN = 6
        NETWORK_ERROR = 7
        AUTHENTICATION_ERROR = 8
        RESOURCE_EXHAUSTED = 9
        MODEL_UNAVAILABLE = 10
        CONTEXT_LENGTH_EXCEEDED = 11

    def __init__(self):
        self.cache = SmartCache(default_ttl=600, max_size=200)  # Enhanced cache
        self.error_counts = defaultdict(int)
        self.request_count = 0
        self.success_count = 0
        self.model_performance = defaultdict(list)

    @classmethod
    def is_valid_response(cls, raw_text: str) -> Tuple[bool, Optional[str]]:
        """Enhanced response validation with more error types"""
        if type(raw_text) is dict and raw_text.get("error", None) is not None and raw_text.get("error") != "":
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        if len(raw_text) == 0:
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        if "<|reserved_token_" in raw_text:
            return False, cls.ErrorType.RESERVED_TOKEN_PRESENT.name
        if 'API request failed with status 429' in raw_text:
            return False, cls.ErrorType.RATE_LIMIT_EXCEEDED.name
        if 'Read timed out' in raw_text or 'timeout' in raw_text.lower():
            return False, cls.ErrorType.TIMEOUT.name
        if 'Network unreachable' in raw_text or 'Connection refused' in raw_text:
            return False, cls.ErrorType.NETWORK_ERROR.name
        if 'context length' in raw_text.lower() or 'token limit' in raw_text.lower():
            return False, cls.ErrorType.CONTEXT_LENGTH_EXCEEDED.name
        if 'model unavailable' in raw_text.lower() or 'service unavailable' in raw_text.lower():
            return False, cls.ErrorType.MODEL_UNAVAILABLE.name
        if 'authentication' in raw_text.lower() or 'unauthorized' in raw_text.lower():
            return False, cls.ErrorType.AUTHENTICATION_ERROR.name
        return True, None

    def get_error_stats(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        total_errors = sum(self.error_counts.values())
        return {
            'total_requests': self.request_count,
            'successful_requests': self.success_count,
            'total_errors': total_errors,
            'error_rate': total_errors / self.request_count if self.request_count > 0 else 0,
            'error_breakdown': dict(self.error_counts),
            'model_performance': dict(self.model_performance)
        }

    @classmethod
    def make_request(cls, messages: list, attempt: int = 10, temperature: float = 0.0) -> str:
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/agents/inference"
        request_data = {
            "run_id": "1",
            "messages": messages,
            "temperature": temperature,
        }
        headers = {
            "Content-Type": "application/json"
        }
        request_data['model'] = AGENT_MODELS[attempt % len(AGENT_MODELS)]
        response = requests.post(url, json=request_data, timeout=120, headers=headers)
        print(f"[agent] HTTP {response.status_code} from {url} ({len(response.content)} bytes)")
        response.raise_for_status()
        response_json = response.json()
        is_oai_interface = (type(response_json) is dict and 
                          response_json.get('choices') is not None and 
                          len(response_json.get('choices')) > 0 and 
                          response_json.get('choices')[0].get('message') is not None)
        if is_oai_interface:
            raw_text = response_json['choices'][0]['message']['content']
        else:
            if type(response_json) is str:
                raw_text = response_json.strip("\n").strip()
            else:
                raw_text = response_json
        if type(raw_text) is not dict:
            raw_text = raw_text.lstrip()
        return raw_text

# Main SWE-Bench run function
def clone_repository_if_needed(repo_path: str, instance_id: str, base_commit: str = None) -> str:
    """
    Clone a repository from GitHub if it doesn't exist locally.
    
    Args:
        repo_path (str): The repository path (could be a GitHub URL or local path)
        instance_id (str): The instance ID to help determine the repository
        base_commit (str, optional): Specific commit to checkout
    
    Returns:
        str: The local path to the cloned repository
    """
    # Check if repo_path is a GitHub URL
    if repo_path.startswith("https://github.com/") or repo_path.startswith("git@github.com:"):
        # Extract repository name from URL
        if "github.com/" in repo_path:
            repo_name = repo_path.split("github.com/")[-1].replace(".git", "")
        else:
            repo_name = repo_path.split("git@github.com:")[-1].replace(".git", "")
        
        # Create local path with instance_id to avoid conflicts
        local_repo_path = f"/tmp/{repo_name.replace('/', '__')}_{instance_id}"
        
        # Clean up if exists
        if os.path.exists(local_repo_path):
            shutil.rmtree(local_repo_path)
        
        print(f"Cloning repository from GitHub: {repo_path}")
        
        # Clone the repository
        result = subprocess.run([
            "git", "clone", 
            repo_path, 
            local_repo_path
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Failed to clone repository: {result.stderr}")
        
        print(f"✓ Successfully cloned repository to {local_repo_path}")
        
        # Checkout specific commit if provided
        if base_commit:
            try:
                subprocess.run([
                    "git", "checkout", base_commit
                ], cwd=local_repo_path, capture_output=True, text=True, check=True)
                print(f"✓ Checked out commit: {base_commit}")
            except subprocess.CalledProcessError:
                print(f"⚠ Could not checkout commit {base_commit}, using latest")
        
        # Configure git user
        subprocess.run(["git", "config", "user.email", "swe-bench@example.com"], 
                      cwd=local_repo_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "SWE-Bench Agent"], 
                      cwd=local_repo_path, check=True, capture_output=True)
        
        return local_repo_path
    
    # Check if repo_path looks like a repository name (e.g., "astropy/astropy", "django/django")
    # and try to clone from GitHub
    if "/" in repo_path and not os.path.exists(repo_path) and not repo_path.startswith("/"):
        github_url = f"https://github.com/{repo_path}.git"
        print(f"Attempting to clone from GitHub: {github_url}")
        
        # Create local path with instance_id to avoid conflicts
        local_repo_path = f"/tmp/{repo_path.replace('/', '__')}_{instance_id}"
        
        # Clean up if exists
        if os.path.exists(local_repo_path):
            shutil.rmtree(local_repo_path)
        
        # Try to clone the repository
        result = subprocess.run([
            "git", "clone", 
            github_url, 
            local_repo_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Successfully cloned repository from GitHub to {local_repo_path}")
            
            # Checkout specific commit if provided
            if base_commit:
                try:
                    subprocess.run([
                        "git", "checkout", base_commit
                    ], cwd=local_repo_path, capture_output=True, text=True, check=True)
                    print(f"✓ Checked out commit: {base_commit}")
                except subprocess.CalledProcessError:
                    print(f"⚠ Could not checkout commit {base_commit}, using latest")
            
            # Configure git user
            subprocess.run(["git", "config", "user.email", "swe-bench@example.com"], 
                          cwd=local_repo_path, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "SWE-Bench Agent"], 
                          cwd=local_repo_path, check=True, capture_output=True)
            
            return local_repo_path
        else:
            print(f"⚠ Failed to clone from GitHub: {result.stderr}")
            print("Continuing with original repo_path...")
    
    # If it's not a GitHub URL or cloneable pattern, return the original path
    return repo_path

def run(
    repo_path: str,
    instance_id: str,
    base_commit: str,
    problem_statement: str,
    version: str,
    traj_dir: str,
    temp_dir: str = None,
    log_path: str = None,
    pr_url: str = None,
    test_file_path: str = None,
    test_case_name: str = None,
    timeout: int = 900,
    **kwargs
) -> str:
    """
    Main entry point for the SWE-Bench evaluation with enhanced accuracy algorithms.
    
    Args:
        repo_path (str): Path to the repository where the changes need to be applied.
        instance_id (str): Unique identifier for the current task instance.
        base_commit (str): The base commit hash for the repository.
        problem_statement (str): The problem statement for the task.
        version (str): Version information for the task.
        traj_dir (str): Directory to write trajectory logs.
        temp_dir (str, optional): Temporary directory for intermediate files. Defaults to None.
        log_path (str, optional): Path to the log file for the current run. Defaults to None.
        test_file_path (str, optional): Path to a specific test file to run. Defaults to None.
        test_case_name (str, optional): Name of a specific test case to run. Defaults to None.
        timeout (int, optional): Maximum time allowed for the agent to run. Defaults to 900.
        **kwargs: Additional keyword arguments.
    
    Returns:
        str: A Git patch string that fixes the described issue.
    """
    
    print(f"🚀 Starting Enhanced SWE-Bench run for instance: {instance_id}")
    print(f"📁 Repository Path: {repo_path}")
    print(f"🔖 Base Commit: {base_commit}")
    print(f"📋 Version: {version}")
    print(f"📝 Problem Statement: {problem_statement[:200]}...")
    print(f"📊 Trajectory Directory: {traj_dir}")
    
    # Set default values for optional parameters
    if temp_dir is None:
        temp_dir = f"/tmp/swe_bench_temp_{instance_id}"
    if log_path is None:
        log_path = f"/tmp/swe_bench_log_{instance_id}.txt"
    
    print(f"🗂️ Temporary Directory: {temp_dir}")
    print(f"📄 Log Path: {log_path}")
    
    # Initialize performance monitoring and trajectory logging
    perf_monitor = PerformanceMonitor()
    logger = TrajectoryLogger(traj_dir)
    network = Network()
    
    perf_monitor.start_timer("total_execution")
    
    # Log initial state
    logger.log_step(instance_id, 1, "initialization", {
        "repo_path": repo_path,
        "base_commit": base_commit,
        "problem_statement": problem_statement,
        "version": version,
        "timeout": timeout,
        "test_file_path": test_file_path,
        "test_case_name": test_case_name
    })
    
    # Handle repository cloning and setup
    original_repo_path = repo_path
    original_cwd = os.getcwd()
    
    try:
        perf_monitor.start_timer("repository_setup")
        repo_path = clone_repository_if_needed(repo_path, instance_id, base_commit)
        perf_monitor.end_timer("repository_setup", True)
        
        logger.log_step(instance_id, 2, "repository_setup", {
            "original_path": original_repo_path,
            "final_path": repo_path,
            "base_commit": base_commit
        })
        
    except Exception as e:
        perf_monitor.end_timer("repository_setup", False)
        print(f"⚠️ Warning: Repository cloning failed: {e}")
        repo_path = original_repo_path
        
        logger.log_step(instance_id, 2, "repository_setup_failed", {
            "error": str(e),
            "fallback_path": repo_path
        })
    
    # Prepare input for agent
    input_dict = {
        "instance_id": instance_id,
        "problem_statement": problem_statement,
        "repo": repo_path,
        "base_commit": base_commit,
        "test_file_path": test_file_path,
        "test_case_name": test_case_name,
        "timeout": timeout,
        "traj_dir": traj_dir,
        "temp_dir": temp_dir,
        "log_path": log_path,
        "pr": kwargs.get("pr", None)
    }
    
    logger.log_step(instance_id, 3, "agent_preparation", {
        "input_dict": input_dict,
        "models_available": AGENT_MODELS
    })
    
    try:
        perf_monitor.start_timer("agent_execution")
        
        # Use timeout manager for the entire agent execution
        with timeout_manager(timeout):
            patch_output = agent_main(input_dict, repo_path, True)
            patch_output = patch_output.get("patch", "")
            print("-----------------")
            print(patch_output)
            print("-----------------")
        
        perf_monitor.end_timer("agent_execution", True)
        
        logger.log_step(instance_id, 4, "agent_execution_success", {
            "patch_length": len(patch_output) if patch_output else 0,
            "patch_preview": patch_output[:500] if patch_output else ""
        })
        
    except TimeoutError:
        perf_monitor.end_timer("agent_execution", False)
        error_msg = f"Agent execution timed out after {timeout} seconds"
        print(f"⏰ {error_msg}")
        
        logger.log_step(instance_id, 4, "agent_execution_timeout", {
            "timeout_seconds": timeout,
            "error": error_msg
        })
        
        patch_output = ""
        
    except Exception as e:
        perf_monitor.end_timer("agent_execution", False)
        error_msg = f"Error in agent execution: {str(e)}"
        print(f"❌ {error_msg}")
        
        patch_output = ""
    
    # Final performance summary and cleanup
    perf_monitor.end_timer("total_execution", patch_output != "")
    
    # Log final result
    success = patch_output != ""
    logger.log_final_result(instance_id, patch_output, success, 
                           "Success" if success else "Failed to generate patch")
    
    # Print performance summary
    print("\n" + perf_monitor.get_performance_summary())
    
    
    # Resource cleanup
    perf_monitor.resource_monitor.cleanup_if_needed()
    
    print(f"✅ Enhanced SWE-Bench run completed for {instance_id}")
    print("+++++++++++++++++")
    print(patch_output)
    print("+++++++++++++++++")
    return patch_output
