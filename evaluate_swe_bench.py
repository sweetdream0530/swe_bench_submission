#!/usr/bin/env python3
"""
SWE-Bench Verified Dataset Evaluation Script

This script generates predictions for the SWE-bench Verified dataset using the enhanced submission.
It creates a predictions.jsonl file that can be evaluated using the official SWE-bench harness.

Usage:
    python evaluate_swe_bench.py --dataset_name princeton-nlp/SWE-bench_Verified --output_file predictions.jsonl

Features:
    - Parallel processing for faster evaluation
    - Comprehensive trajectory logging
    - Resource monitoring and management
    - Progress tracking and error handling
    - Resume capability for interrupted runs
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import logging

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import run
from trajectory_logger import TrajectoryLogger

# Optional imports with fallback
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("⚠️ Warning: datasets library not available. Install with: pip install datasets")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ Warning: psutil not available. Install with: pip install psutil")

class SWEBenchEvaluator:
    """Enhanced SWE-Bench evaluation with parallel processing and comprehensive logging."""
    
    def __init__(self, 
                 dataset_name: str = "princeton-nlp/SWE-bench_Verified",
                 output_file: str = "predictions.jsonl",
                 traj_dir: str = "results/trajectories",
                 log_dir: str = "results/logs",
                 max_workers: int = 4,
                 timeout: int = 2200,
                 max_instances: Optional[int] = None,
                 resume: bool = False):
        
        self.dataset_name = dataset_name
        self.output_file = Path(output_file)
        self.traj_dir = Path(traj_dir)
        self.log_dir = Path(log_dir)
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_instances = max_instances
        self.resume = resume
        
        # Create directories
        self.traj_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Statistics
        self.stats = {
            'total_instances': 0,
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Load existing predictions if resuming
        self.existing_predictions = set()
        if self.resume and self.output_file.exists():
            self.load_existing_predictions()
    
    def setup_logging(self):
        """Setup comprehensive logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"evaluation_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🚀 Starting SWE-Bench evaluation")
        self.logger.info(f"📁 Output file: {self.output_file}")
        self.logger.info(f"📊 Trajectory directory: {self.traj_dir}")
        self.logger.info(f"⚙️ Max workers: {self.max_workers}")
        self.logger.info(f"⏰ Timeout: {self.timeout}s")
    
    def load_existing_predictions(self):
        """Load existing predictions for resume functionality."""
        try:
            with open(self.output_file, 'r') as f:
                for line in f:
                    if line.strip():
                        prediction = json.loads(line)
                        self.existing_predictions.add(prediction['instance_id'])
            
            self.logger.info(f"📋 Loaded {len(self.existing_predictions)} existing predictions")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load existing predictions: {e}")
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load the SWE-bench dataset."""
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library not available. Install with: pip install datasets")
        
        self.logger.info(f"📥 Loading dataset: {self.dataset_name}")
        
        try:
            dataset = load_dataset(self.dataset_name)
            instances = dataset['test']  # SWE-bench Verified has 'test' split
            
            self.logger.info(f"✅ Loaded {len(instances)} instances from dataset")
            
            # Debug: Check the type of first instance
            if instances:
                first_instance = instances[0]
                self.logger.info(f"🔍 First instance type: {type(first_instance)}")
                if isinstance(first_instance, dict):
                    self.logger.info(f"🔍 First instance keys: {list(first_instance.keys())}")
                else:
                    self.logger.warning(f"⚠️ First instance is not a dict: {first_instance}")
            
            # Convert to regular Python list for multiprocessing compatibility
            instances = list(instances)
            self.logger.info(f"🔄 Converted to Python list with {len(instances)} instances")
            
            # Apply max_instances limit if specified
            if self.max_instances:
                instances = instances[:self.max_instances]
                self.logger.info(f"🔢 Limited to {len(instances)} instances")
            
            return instances
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load dataset: {e}")
            raise
    
    def process_instance(self, instance: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single SWE-bench instance and generate prediction."""
        instance_id = instance['instance_id']
        
        # Skip if already processed (resume functionality)
        if instance_id in self.existing_predictions:
            self.logger.info(f"⏭️ Skipping {instance_id} (already processed)")
            return None
        
        self.logger.info(f"🔄 Processing {instance_id}")
        
        try:
            # Prepare input for the run function
            input_params = {
                'repo_path': instance['repo'],
                'instance_id': instance_id,
                'base_commit': instance['base_commit'],
                'problem_statement': instance['problem_statement'],
                'version': instance.get('version', '1.0'),
                'traj_dir': str(self.traj_dir),
                'timeout': self.timeout,
                'test_file_path': instance.get('test_file'),
                'test_case_name': instance.get('test_cases', [None])[0] if instance.get('test_cases') else None
            }
            
            # Generate prediction using the main run function
            start_time = time.time()
            patch = run(**input_params)
            duration = time.time() - start_time
            
            # Create prediction entry
            prediction = {
                "instance_id": instance_id,
                "model_name_or_path": "enhanced-swe-bench-submission",
                "model_patch": patch if patch else ""
            }
            
            self.logger.info(f"✅ Completed {instance_id} in {duration:.2f}s")
            return prediction
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process {instance_id}: {e}")
            self.logger.error(traceback.format_exc())
            
            # Return empty prediction for failed instances
            return {
                "instance_id": instance_id,
                "model_name_or_path": "enhanced-swe-bench-submission",
                "model_patch": ""
            }
    
    def process_instances_parallel(self, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process instances in parallel using ProcessPoolExecutor."""
        predictions = []
        
        self.logger.info(f"🚀 Starting parallel processing with {self.max_workers} workers")
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_instance = {
                executor.submit(self.process_instance, instance): instance 
                for instance in instances
            }
            
            # Process completed tasks
            for future in as_completed(future_to_instance):
                instance = future_to_instance[future]
                
                # Debug: Check instance type
                self.logger.info(f"🔍 Instance type in parallel processing: {type(instance)}")
                if isinstance(instance, dict):
                    self.logger.info(f"🔍 Instance keys: {list(instance.keys())}")
                else:
                    self.logger.warning(f"⚠️ Instance is not a dict: {instance}")
                
                instance_id = instance['instance_id']
                
                try:
                    prediction = future.result()
                    if prediction:
                        predictions.append(prediction)
                        self.stats['successful'] += 1
                    else:
                        self.stats['skipped'] += 1
                    
                    self.stats['processed'] += 1
                    
                    # Log progress
                    progress = (self.stats['processed'] / len(instances)) * 100
                    self.logger.info(f"📊 Progress: {self.stats['processed']}/{len(instances)} ({progress:.1f}%)")
                    
                except Exception as e:
                    self.logger.error(f"❌ Task failed for {instance_id}: {e}")
                    self.stats['failed'] += 1
                    self.stats['processed'] += 1
        
        return predictions
    
    def process_instances_sequential(self, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process instances sequentially (fallback for debugging)."""
        predictions = []
        
        self.logger.info("🔄 Starting sequential processing")
        
        for i, instance in enumerate(instances):
            instance_id = instance['instance_id']
            
            # Skip if already processed
            if instance_id in self.existing_predictions:
                self.stats['skipped'] += 1
                continue
            
            try:
                prediction = self.process_instance(instance)
                if prediction:
                    predictions.append(prediction)
                    self.stats['successful'] += 1
                else:
                    self.stats['skipped'] += 1
                
                self.stats['processed'] += 1
                
                # Log progress every 10 instances
                if (i + 1) % 10 == 0:
                    progress = ((i + 1) / len(instances)) * 100
                    self.logger.info(f"📊 Progress: {i + 1}/{len(instances)} ({progress:.1f}%)")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to process {instance_id}: {e}")
                self.stats['failed'] += 1
                self.stats['processed'] += 1
        
        return predictions
    
    def save_predictions(self, predictions: List[Dict[str, Any]]):
        """Save predictions to JSONL file."""
        self.logger.info(f"💾 Saving {len(predictions)} predictions to {self.output_file}")
        
        # Load existing predictions if resuming
        existing_predictions = []
        if self.resume and self.output_file.exists():
            with open(self.output_file, 'r') as f:
                for line in f:
                    if line.strip():
                        existing_predictions.append(json.loads(line))
        
        # Combine existing and new predictions
        all_predictions = existing_predictions + predictions
        
        # Save to file
        with open(self.output_file, 'w') as f:
            for prediction in all_predictions:
                f.write(json.dumps(prediction) + '\n')
        
        self.logger.info(f"✅ Saved {len(all_predictions)} total predictions")
    
    def print_final_stats(self):
        """Print final evaluation statistics."""
        duration = time.time() - self.stats['start_time']
        
        self.logger.info("=" * 60)
        self.logger.info("📊 FINAL EVALUATION STATISTICS")
        self.logger.info("=" * 60)
        self.logger.info(f"📈 Total instances: {self.stats['total_instances']}")
        self.logger.info(f"✅ Successful: {self.stats['successful']}")
        self.logger.info(f"❌ Failed: {self.stats['failed']}")
        self.logger.info(f"⏭️ Skipped: {self.stats['skipped']}")
        self.logger.info(f"⏰ Total time: {duration:.2f}s")
        self.logger.info(f"⚡ Average time per instance: {duration/max(self.stats['processed'], 1):.2f}s")
        self.logger.info(f"📁 Predictions saved to: {self.output_file}")
        self.logger.info(f"📊 Trajectories saved to: {self.traj_dir}")
        
        # Resource usage summary
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            self.logger.info(f"💾 Peak memory usage: {memory.used / (1024**3):.2f}GB")
        
        self.logger.info("=" * 60)
    
    def run_evaluation(self):
        """Run the complete SWE-bench evaluation."""
        self.stats['start_time'] = time.time()
        
        try:
            # Load dataset
            instances = self.load_dataset()
            self.stats['total_instances'] = len(instances)
            
            # Process instances
            if self.max_workers > 1:
                predictions = self.process_instances_parallel(instances)
            else:
                predictions = self.process_instances_sequential(instances)
            
            # Save predictions
            self.save_predictions(predictions)
            
            # Print final statistics
            self.print_final_stats()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Evaluation failed: {e}")
            self.logger.error(traceback.format_exc())
            return False
        
        finally:
            self.stats['end_time'] = time.time()

def main():
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate SWE-bench Verified dataset with enhanced submission",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python evaluate_swe_bench.py --dataset_name princeton-nlp/SWE-bench_Verified
  
  # Parallel evaluation with custom output
  python evaluate_swe_bench.py --max_workers 8 --output_file results/predictions.jsonl
  
  # Resume interrupted evaluation
  python evaluate_swe_bench.py --resume --max_instances 100
  
  # Debug mode (sequential processing)
  python evaluate_swe_bench.py --max_workers 1 --timeout 3600
        """
    )
    
    parser.add_argument(
        '--dataset_name',
        default='princeton-nlp/SWE-bench_Verified',
        help='Hugging Face dataset name (default: princeton-nlp/SWE-bench_Verified)'
    )
    
    parser.add_argument(
        '--output_file',
        default='results/predictions/predictions.jsonl',
        help='Output predictions file (default: results/predictions/predictions.jsonl)'
    )
    
    parser.add_argument(
        '--traj_dir',
        default='results/trajectories',
        help='Trajectory logging directory (default: results/trajectories)'
    )
    
    parser.add_argument(
        '--log_dir',
        default='results/logs',
        help='Log directory (default: results/logs)'
    )
    
    parser.add_argument(
        '--max_workers',
        type=int,
        default=4,
        help='Maximum number of parallel workers (default: 4)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=2200,
        help='Timeout per instance in seconds (default: 2200)'
    )
    
    parser.add_argument(
        '--max_instances',
        type=int,
        default=None,
        help='Maximum number of instances to process (default: all)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume evaluation from existing predictions file'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.max_workers < 1:
        print("❌ Error: max_workers must be at least 1")
        sys.exit(1)
    
    if args.timeout < 60:
        print("❌ Error: timeout must be at least 60 seconds")
        sys.exit(1)
    
    # Create evaluator and run
    evaluator = SWEBenchEvaluator(
        dataset_name=args.dataset_name,
        output_file=args.output_file,
        traj_dir=args.traj_dir,
        log_dir=args.log_dir,
        max_workers=args.max_workers,
        timeout=args.timeout,
        max_instances=args.max_instances,
        resume=args.resume
    )
    
    success = evaluator.run_evaluation()
    
    if success:
        print("\n🎉 Evaluation completed successfully!")
        print(f"📁 Predictions saved to: {args.output_file}")
        print(f"📊 Run evaluation harness with:")
        print(f"python -m swebench.harness.run_evaluation \\")
        print(f"    --dataset_name {args.dataset_name} \\")
        print(f"    --predictions_path {args.output_file} \\")
        print(f"    --max_workers 8 \\")
        print(f"    --run_id enhanced_swe_bench_evaluation")
        sys.exit(0)
    else:
        print("\n❌ Evaluation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
