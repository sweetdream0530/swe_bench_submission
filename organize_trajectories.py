#!/usr/bin/env python3
"""
Trajectory Organization Script for SWE-Bench Submission

This script organizes trajectories into the proper folder structure required for SWE-bench submission,
ensuring each instance ID has its own folder with comprehensive trajectory data.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime

class TrajectoryOrganizer:
    """Organizes trajectories into proper folder structure for SWE-bench submission."""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for the organizer."""
        # Ensure output directory exists before setting up logging
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'organization.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_folder_structure(self, instance_id: str) -> Path:
        """Create folder structure for a specific instance ID."""
        instance_dir = self.output_dir / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different types of data
        subdirs = [
            'llm_interactions',
            'tool_calls', 
            'filtering_decisions',
            'performance_metrics',
            'raw_logs',
            'artifacts'
        ]
        
        for subdir in subdirs:
            (instance_dir / subdir).mkdir(exist_ok=True)
        
        return instance_dir
    
    def parse_trajectory_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse a trajectory JSONL file."""
        trajectory = []
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            trajectory.append(entry)
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Invalid JSON on line {line_num} in {file_path}: {e}")
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
        
        return trajectory
    
    def extract_llm_interactions(self, trajectory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract LLM interactions from trajectory."""
        llm_interactions = []
        
        for entry in trajectory:
            if entry.get('action') == 'llm_interaction':
                llm_interactions.append(entry)
            elif entry.get('action') == 'agent_execution_success':
                # Extract LLM interactions from agent execution details
                details = entry.get('details', {})
                if 'llm_interactions' in details:
                    llm_interactions.extend(details['llm_interactions'])
        
        return llm_interactions
    
    def extract_tool_calls(self, trajectory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract tool calls from trajectory."""
        tool_calls = []
        
        for entry in trajectory:
            if entry.get('action') == 'tool_call':
                tool_calls.append(entry)
            elif entry.get('action') in ['read_file', 'write_file', 'run_command', 'search_code']:
                # Convert general actions to tool calls
                tool_call_entry = {
                    'timestamp': entry.get('timestamp'),
                    'step': entry.get('step'),
                    'action': 'tool_call',
                    'details': {
                        'tool_name': entry.get('action'),
                        'tool_args': entry.get('details', {}),
                        'tool_result': entry.get('details', {}).get('result', ''),
                        'execution_time': entry.get('performance', {}).get('duration_seconds', 0),
                        'success': True
                    },
                    'performance': entry.get('performance', {}),
                    'metadata': entry.get('metadata', {})
                }
                tool_calls.append(tool_call_entry)
        
        return tool_calls
    
    def extract_filtering_decisions(self, trajectory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract filtering decisions from trajectory."""
        filtering_decisions = []
        
        for entry in trajectory:
            # Look for decision-making steps
            if entry.get('action') in ['model_selection', 'strategy_selection', 'path_selection']:
                filtering_decisions.append(entry)
            elif 'decision' in entry.get('details', {}):
                filtering_decisions.append(entry)
            elif 'filtering' in entry.get('action', '').lower():
                filtering_decisions.append(entry)
        
        return filtering_decisions
    
    def extract_performance_metrics(self, trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract performance metrics from trajectory."""
        metrics = {
            'total_steps': len(trajectory),
            'llm_interactions': 0,
            'tool_calls': 0,
            'filtering_decisions': 0,
            'total_duration': 0,
            'memory_usage': [],
            'step_types': {}
        }
        
        for entry in trajectory:
            # Count different action types
            action = entry.get('action', 'unknown')
            if action == 'llm_interaction':
                metrics['llm_interactions'] += 1
            elif action == 'tool_call':
                metrics['tool_calls'] += 1
            elif 'decision' in action or 'filtering' in action.lower():
                metrics['filtering_decisions'] += 1
            
            # Aggregate performance data
            performance = entry.get('performance', {})
            metrics['total_duration'] += performance.get('duration_seconds', 0)
            
            memory_usage = performance.get('memory_usage_mb', 0)
            if memory_usage > 0:
                metrics['memory_usage'].append(memory_usage)
            
            # Track step types
            step_type = performance.get('step_type', 'general')
            metrics['step_types'][step_type] = metrics['step_types'].get(step_type, 0) + 1
        
        # Calculate averages
        if metrics['memory_usage']:
            metrics['avg_memory_usage'] = sum(metrics['memory_usage']) / len(metrics['memory_usage'])
            metrics['max_memory_usage'] = max(metrics['memory_usage'])
        else:
            metrics['avg_memory_usage'] = 0
            metrics['max_memory_usage'] = 0
        
        return metrics
    
    def organize_instance(self, instance_id: str, trajectory_file: Path) -> bool:
        """Organize trajectory data for a specific instance."""
        try:
            self.logger.info(f"Organizing trajectory for {instance_id}")
            
            # Create folder structure
            instance_dir = self.create_folder_structure(instance_id)
            
            # Parse trajectory file
            trajectory = self.parse_trajectory_file(trajectory_file)
            
            if not trajectory:
                self.logger.warning(f"No trajectory data found for {instance_id}")
                return False
            
            # Extract different types of data
            llm_interactions = self.extract_llm_interactions(trajectory)
            tool_calls = self.extract_tool_calls(trajectory)
            filtering_decisions = self.extract_filtering_decisions(trajectory)
            performance_metrics = self.extract_performance_metrics(trajectory)
            
            # Save organized data
            self.save_llm_interactions(instance_dir, llm_interactions)
            self.save_tool_calls(instance_dir, tool_calls)
            self.save_filtering_decisions(instance_dir, filtering_decisions)
            self.save_performance_metrics(instance_dir, performance_metrics)
            self.save_raw_trajectory(instance_dir, trajectory)
            self.save_summary(instance_dir, instance_id, trajectory, llm_interactions, tool_calls, filtering_decisions, performance_metrics)
            
            self.logger.info(f"Successfully organized {instance_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error organizing {instance_id}: {e}")
            return False
    
    def save_llm_interactions(self, instance_dir: Path, llm_interactions: List[Dict[str, Any]]):
        """Save LLM interactions to separate file."""
        llm_file = instance_dir / 'llm_interactions' / 'interactions.json'
        with open(llm_file, 'w') as f:
            json.dump(llm_interactions, f, indent=2)
    
    def save_tool_calls(self, instance_dir: Path, tool_calls: List[Dict[str, Any]]):
        """Save tool calls to separate file."""
        tools_file = instance_dir / 'tool_calls' / 'calls.json'
        with open(tools_file, 'w') as f:
            json.dump(tool_calls, f, indent=2)
    
    def save_filtering_decisions(self, instance_dir: Path, filtering_decisions: List[Dict[str, Any]]):
        """Save filtering decisions to separate file."""
        decisions_file = instance_dir / 'filtering_decisions' / 'decisions.json'
        with open(decisions_file, 'w') as f:
            json.dump(filtering_decisions, f, indent=2)
    
    def save_performance_metrics(self, instance_dir: Path, performance_metrics: Dict[str, Any]):
        """Save performance metrics to separate file."""
        metrics_file = instance_dir / 'performance_metrics' / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(performance_metrics, f, indent=2)
    
    def save_raw_trajectory(self, instance_dir: Path, trajectory: List[Dict[str, Any]]):
        """Save raw trajectory data."""
        raw_file = instance_dir / 'raw_logs' / 'trajectory.jsonl'
        with open(raw_file, 'w') as f:
            for entry in trajectory:
                f.write(json.dumps(entry) + '\n')
    
    def save_summary(self, instance_dir: Path, instance_id: str, trajectory: List[Dict[str, Any]], 
                    llm_interactions: List[Dict[str, Any]], tool_calls: List[Dict[str, Any]], 
                    filtering_decisions: List[Dict[str, Any]], performance_metrics: Dict[str, Any]):
        """Save a comprehensive summary of the trajectory."""
        summary = {
            'instance_id': instance_id,
            'organization_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_trajectory_entries': len(trajectory),
                'llm_interactions_count': len(llm_interactions),
                'tool_calls_count': len(tool_calls),
                'filtering_decisions_count': len(filtering_decisions),
                'performance_metrics': performance_metrics
            },
            'structure': {
                'llm_interactions': 'llm_interactions/interactions.json',
                'tool_calls': 'tool_calls/calls.json',
                'filtering_decisions': 'filtering_decisions/decisions.json',
                'performance_metrics': 'performance_metrics/metrics.json',
                'raw_trajectory': 'raw_logs/trajectory.jsonl'
            },
            'walkability': {
                'has_llm_inputs': len(llm_interactions) > 0,
                'has_llm_outputs': len(llm_interactions) > 0,
                'has_filtering_decisions': len(filtering_decisions) > 0,
                'has_tool_calls': len(tool_calls) > 0,
                'is_walkable': len(llm_interactions) > 0 and len(tool_calls) > 0
            }
        }
        
        summary_file = instance_dir / 'summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def organize_all_trajectories(self) -> Dict[str, Any]:
        """Organize all trajectories in the input directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        # Find all trajectory files (both direct .jsonl files and in subdirectories)
        trajectory_files = []
        
        # Look for direct .jsonl files
        direct_files = list(self.input_dir.glob('*.jsonl'))
        trajectory_files.extend(direct_files)
        
        # Look for .jsonl files in subdirectories
        subdir_files = list(self.input_dir.glob('*/trajectory.jsonl'))
        trajectory_files.extend(subdir_files)
        
        results['total_files'] = len(trajectory_files)
        
        self.logger.info(f"Found {len(trajectory_files)} trajectory files to organize")
        
        for trajectory_file in trajectory_files:
            # Extract instance ID from filename or parent directory
            if trajectory_file.parent == self.input_dir:
                # Direct file
                instance_id = trajectory_file.stem
            else:
                # File in subdirectory
                instance_id = trajectory_file.parent.name
            
            if self.organize_instance(instance_id, trajectory_file):
                results['successful'] += 1
            else:
                results['failed'] += 1
                results['errors'].append(f"Failed to organize {instance_id}")
        
        # Save overall results
        results_file = self.output_dir / 'organization_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Organization complete: {results['successful']}/{results['total_files']} successful")
        return results

def main():
    """Main entry point for the trajectory organizer."""
    parser = argparse.ArgumentParser(
        description="Organize SWE-bench trajectories into proper folder structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic organization
  python organize_trajectories.py --input_dir submission_trajectories --output_dir organized_trajectories
  
  # With custom paths
  python organize_trajectories.py -i results/trajectories -o results/organized
        """
    )
    
    parser.add_argument(
        '--input_dir', '-i',
        default='submission_trajectories',
        help='Input directory containing trajectory files (default: submission_trajectories)'
    )
    
    parser.add_argument(
        '--output_dir', '-o',
        default='organized_trajectories',
        help='Output directory for organized trajectories (default: organized_trajectories)'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory '{args.input_dir}' does not exist")
        return 1
    
    # Create organizer and run
    organizer = TrajectoryOrganizer(args.input_dir, args.output_dir)
    results = organizer.organize_all_trajectories()
    
    if results['successful'] > 0:
        print(f"\n🎉 Successfully organized {results['successful']} trajectories!")
        print(f"📁 Organized trajectories saved to: {args.output_dir}")
        print(f"📊 Results: {results['successful']}/{results['total_files']} successful")
        
        if results['errors']:
            print(f"⚠️ Errors encountered:")
            for error in results['errors']:
                print(f"  - {error}")
        
        return 0
    else:
        print(f"\n❌ Failed to organize any trajectories!")
        print(f"📊 Results: {results['successful']}/{results['total_files']} successful")
        return 1

if __name__ == "__main__":
    exit(main())