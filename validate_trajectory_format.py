#!/usr/bin/env python3
"""
Trajectory Validation Script for SWE-Bench Submission

This script validates that trajectories include all required information for walkable LLM decisions,
ensuring compliance with SWE-bench submission requirements.

Usage:
    python validate_trajectory_format.py --trajectory_dir organized_trajectories
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime

class TrajectoryValidator:
    """Validates trajectory format for SWE-bench submission compliance."""
    
    def __init__(self, trajectory_dir: str):
        self.trajectory_dir = Path(trajectory_dir)
        self.setup_logging()
        self.validation_results = {
            'total_instances': 0,
            'compliant_instances': 0,
            'non_compliant_instances': 0,
            'validation_details': {},
            'overall_compliance': False
        }
        
    def setup_logging(self):
        """Setup logging for the validator."""
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.trajectory_dir / 'validation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def validate_instance_structure(self, instance_dir: Path) -> Dict[str, Any]:
        """Validate the folder structure for a single instance."""
        validation_result = {
            'instance_id': instance_dir.name,
            'structure_valid': False,
            'missing_components': [],
            'present_components': [],
            'walkability_score': 0.0,
            'compliance_issues': []
        }
        
        required_components = [
            'llm_interactions',
            'tool_calls',
            'filtering_decisions',
            'performance_metrics',
            'raw_logs',
            'summary.json'
        ]
        
        for component in required_components:
            component_path = instance_dir / component
            if component_path.exists():
                validation_result['present_components'].append(component)
            else:
                validation_result['missing_components'].append(component)
                validation_result['compliance_issues'].append(f"Missing required component: {component}")
        
        validation_result['structure_valid'] = len(validation_result['missing_components']) == 0
        
        return validation_result
    
    def validate_llm_interactions(self, instance_dir: Path) -> Dict[str, Any]:
        """Validate LLM interactions data."""
        validation_result = {
            'has_interactions': False,
            'interaction_count': 0,
            'has_inputs': False,
            'has_outputs': False,
            'has_metadata': False,
            'compliance_score': 0.0,
            'issues': []
        }
        
        interactions_file = instance_dir / 'llm_interactions' / 'interactions.json'
        
        if not interactions_file.exists():
            validation_result['issues'].append("LLM interactions file not found")
            return validation_result
        
        try:
            with open(interactions_file, 'r') as f:
                interactions = json.load(f)
            
            validation_result['interaction_count'] = len(interactions)
            validation_result['has_interactions'] = len(interactions) > 0
            
            if interactions:
                # Check first interaction for required fields
                first_interaction = interactions[0]
                
                required_fields = ['model_name', 'input_messages', 'response']
                for field in required_fields:
                    if field in first_interaction.get('details', {}):
                        if field == 'input_messages':
                            validation_result['has_inputs'] = True
                        elif field == 'response':
                            validation_result['has_outputs'] = True
                    else:
                        validation_result['issues'].append(f"Missing required field: {field}")
                
                # Check for metadata
                if 'metadata' in first_interaction.get('details', {}):
                    validation_result['has_metadata'] = True
                
                # Calculate compliance score
                score_components = [
                    validation_result['has_interactions'],
                    validation_result['has_inputs'],
                    validation_result['has_outputs'],
                    validation_result['has_metadata']
                ]
                validation_result['compliance_score'] = sum(score_components) / len(score_components)
                
        except Exception as e:
            validation_result['issues'].append(f"Error reading interactions file: {e}")
        
        return validation_result
    
    def validate_tool_calls(self, instance_dir: Path) -> Dict[str, Any]:
        """Validate tool calls data."""
        validation_result = {
            'has_tool_calls': False,
            'tool_call_count': 0,
            'has_tool_names': False,
            'has_tool_args': False,
            'has_tool_results': False,
            'compliance_score': 0.0,
            'issues': []
        }
        
        tools_file = instance_dir / 'tool_calls' / 'calls.json'
        
        if not tools_file.exists():
            validation_result['issues'].append("Tool calls file not found")
            return validation_result
        
        try:
            with open(tools_file, 'r') as f:
                tool_calls = json.load(f)
            
            validation_result['tool_call_count'] = len(tool_calls)
            validation_result['has_tool_calls'] = len(tool_calls) > 0
            
            if tool_calls:
                # Check first tool call for required fields
                first_tool_call = tool_calls[0]
                details = first_tool_call.get('details', {})
                
                required_fields = ['tool_name', 'tool_args', 'tool_result']
                for field in required_fields:
                    if field in details:
                        if field == 'tool_name':
                            validation_result['has_tool_names'] = True
                        elif field == 'tool_args':
                            validation_result['has_tool_args'] = True
                        elif field == 'tool_result':
                            validation_result['has_tool_results'] = True
                    else:
                        validation_result['issues'].append(f"Missing required field: {field}")
                
                # Calculate compliance score
                score_components = [
                    validation_result['has_tool_calls'],
                    validation_result['has_tool_names'],
                    validation_result['has_tool_args'],
                    validation_result['has_tool_results']
                ]
                validation_result['compliance_score'] = sum(score_components) / len(score_components)
                
        except Exception as e:
            validation_result['issues'].append(f"Error reading tool calls file: {e}")
        
        return validation_result
    
    def validate_filtering_decisions(self, instance_dir: Path) -> Dict[str, Any]:
        """Validate filtering decisions data."""
        validation_result = {
            'has_decisions': False,
            'decision_count': 0,
            'has_decision_types': False,
            'has_reasoning': False,
            'compliance_score': 0.0,
            'issues': []
        }
        
        decisions_file = instance_dir / 'filtering_decisions' / 'decisions.json'
        
        if not decisions_file.exists():
            validation_result['issues'].append("Filtering decisions file not found")
            return validation_result
        
        try:
            with open(decisions_file, 'r') as f:
                decisions = json.load(f)
            
            validation_result['decision_count'] = len(decisions)
            validation_result['has_decisions'] = len(decisions) > 0
            
            if decisions:
                # Check first decision for required fields
                first_decision = decisions[0]
                details = first_decision.get('details', {})
                
                required_fields = ['decision_type', 'selection_reason']
                for field in required_fields:
                    if field in details:
                        if field == 'decision_type':
                            validation_result['has_decision_types'] = True
                        elif field == 'selection_reason':
                            validation_result['has_reasoning'] = True
                    else:
                        validation_result['issues'].append(f"Missing required field: {field}")
                
                # Calculate compliance score
                score_components = [
                    validation_result['has_decisions'],
                    validation_result['has_decision_types'],
                    validation_result['has_reasoning']
                ]
                validation_result['compliance_score'] = sum(score_components) / len(score_components)
                
        except Exception as e:
            validation_result['issues'].append(f"Error reading decisions file: {e}")
        
        return validation_result
    
    def validate_walkability(self, instance_dir: Path) -> Dict[str, Any]:
        """Validate that the trajectory is walkable (can trace every LLM decision)."""
        validation_result = {
            'is_walkable': False,
            'walkability_score': 0.0,
            'missing_for_walkability': [],
            'walkability_components': {
                'has_llm_interactions': False,
                'has_tool_calls': False,
                'has_decision_trace': False,
                'has_performance_data': False
            }
        }
        
        # Check for LLM interactions
        interactions_file = instance_dir / 'llm_interactions' / 'interactions.json'
        if interactions_file.exists():
            try:
                with open(interactions_file, 'r') as f:
                    interactions = json.load(f)
                validation_result['walkability_components']['has_llm_interactions'] = len(interactions) > 0
            except:
                pass
        
        # Check for tool calls
        tools_file = instance_dir / 'tool_calls' / 'calls.json'
        if tools_file.exists():
            try:
                with open(tools_file, 'r') as f:
                    tool_calls = json.load(f)
                validation_result['walkability_components']['has_tool_calls'] = len(tool_calls) > 0
            except:
                pass
        
        # Check for decision trace
        decisions_file = instance_dir / 'filtering_decisions' / 'decisions.json'
        if decisions_file.exists():
            try:
                with open(decisions_file, 'r') as f:
                    decisions = json.load(f)
                validation_result['walkability_components']['has_decision_trace'] = len(decisions) > 0
            except:
                pass
        
        # Check for performance data
        metrics_file = instance_dir / 'performance_metrics' / 'metrics.json'
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                validation_result['walkability_components']['has_performance_data'] = len(metrics) > 0
            except:
                pass
        
        # Calculate walkability score
        components = validation_result['walkability_components']
        score_components = list(components.values())
        validation_result['walkability_score'] = sum(score_components) / len(score_components)
        
        # Determine if walkable (requires at least LLM interactions and tool calls)
        validation_result['is_walkable'] = (
            components['has_llm_interactions'] and 
            components['has_tool_calls'] and
            validation_result['walkability_score'] >= 0.5
        )
        
        # Identify missing components
        for component, present in components.items():
            if not present:
                validation_result['missing_for_walkability'].append(component)
        
        return validation_result
    
    def validate_instance(self, instance_dir: Path) -> Dict[str, Any]:
        """Validate a single instance completely."""
        self.logger.info(f"Validating instance: {instance_dir.name}")
        
        validation_result = {
            'instance_id': instance_dir.name,
            'timestamp': datetime.now().isoformat(),
            'overall_compliant': False,
            'overall_score': 0.0,
            'structure_validation': {},
            'llm_interactions_validation': {},
            'tool_calls_validation': {},
            'filtering_decisions_validation': {},
            'walkability_validation': {},
            'recommendations': []
        }
        
        # Run all validation checks
        validation_result['structure_validation'] = self.validate_instance_structure(instance_dir)
        validation_result['llm_interactions_validation'] = self.validate_llm_interactions(instance_dir)
        validation_result['tool_calls_validation'] = self.validate_tool_calls(instance_dir)
        validation_result['filtering_decisions_validation'] = self.validate_filtering_decisions(instance_dir)
        validation_result['walkability_validation'] = self.validate_walkability(instance_dir)
        
        # Calculate overall score
        scores = [
            validation_result['structure_validation'].get('structure_valid', False),
            validation_result['llm_interactions_validation'].get('compliance_score', 0.0),
            validation_result['tool_calls_validation'].get('compliance_score', 0.0),
            validation_result['filtering_decisions_validation'].get('compliance_score', 0.0),
            validation_result['walkability_validation'].get('walkability_score', 0.0)
        ]
        
        validation_result['overall_score'] = sum(scores) / len(scores)
        validation_result['overall_compliant'] = validation_result['overall_score'] >= 0.7
        
        # Generate recommendations
        if not validation_result['structure_validation']['structure_valid']:
            validation_result['recommendations'].append("Fix folder structure - ensure all required components are present")
        
        if validation_result['llm_interactions_validation']['compliance_score'] < 0.5:
            validation_result['recommendations'].append("Enhance LLM interaction logging - capture inputs, outputs, and metadata")
        
        if validation_result['tool_calls_validation']['compliance_score'] < 0.5:
            validation_result['recommendations'].append("Improve tool call logging - capture tool names, arguments, and results")
        
        if validation_result['filtering_decisions_validation']['compliance_score'] < 0.5:
            validation_result['recommendations'].append("Add filtering decision tracking - log decision types and reasoning")
        
        if not validation_result['walkability_validation']['is_walkable']:
            validation_result['recommendations'].append("Improve walkability - ensure all LLM decisions can be traced")
        
        return validation_result
    
    def validate_all_trajectories(self) -> Dict[str, Any]:
        """Validate all trajectories in the directory."""
        self.logger.info("Starting trajectory validation")
        
        # Find all instance directories
        instance_dirs = [d for d in self.trajectory_dir.iterdir() if d.is_dir() and d.name != '__pycache__']
        
        self.validation_results['total_instances'] = len(instance_dirs)
        
        for instance_dir in instance_dirs:
            try:
                validation_result = self.validate_instance(instance_dir)
                self.validation_results['validation_details'][instance_dir.name] = validation_result
                
                if validation_result['overall_compliant']:
                    self.validation_results['compliant_instances'] += 1
                else:
                    self.validation_results['non_compliant_instances'] += 1
                    
            except Exception as e:
                self.logger.error(f"Error validating {instance_dir.name}: {e}")
                self.validation_results['non_compliant_instances'] += 1
        
        # Calculate overall compliance
        if self.validation_results['total_instances'] > 0:
            compliance_rate = self.validation_results['compliant_instances'] / self.validation_results['total_instances']
            self.validation_results['overall_compliance'] = compliance_rate >= 0.8
        
        # Save validation results
        results_file = self.trajectory_dir / 'validation_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        self.logger.info(f"Validation complete: {self.validation_results['compliant_instances']}/{self.validation_results['total_instances']} compliant")
        
        return self.validation_results

def main():
    """Main entry point for the trajectory validator."""
    parser = argparse.ArgumentParser(
        description="Validate SWE-bench trajectory format compliance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation
  python validate_trajectory_format.py --trajectory_dir organized_trajectories
  
  # Validate specific directory
  python validate_trajectory_format.py -d results/trajectories
        """
    )
    
    parser.add_argument(
        '--trajectory_dir', '-d',
        default='organized_trajectories',
        help='Directory containing organized trajectories (default: organized_trajectories)'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.trajectory_dir):
        print(f"❌ Error: Trajectory directory '{args.trajectory_dir}' does not exist")
        return 1
    
    # Create validator and run
    validator = TrajectoryValidator(args.trajectory_dir)
    results = validator.validate_all_trajectories()
    
    # Print summary
    print(f"\n📊 Validation Results:")
    print(f"  Total instances: {results['total_instances']}")
    print(f"  Compliant instances: {results['compliant_instances']}")
    print(f"  Non-compliant instances: {results['non_compliant_instances']}")
    print(f"  Overall compliance: {'✅ PASS' if results['overall_compliance'] else '❌ FAIL'}")
    
    if results['overall_compliance']:
        print(f"\n🎉 Trajectories are compliant with SWE-bench submission requirements!")
        print(f"📁 Validation results saved to: {args.trajectory_dir}/validation_results.json")
        return 0
    else:
        print(f"\n⚠️ Trajectories need improvement for SWE-bench submission compliance")
        print(f"📁 Detailed validation results saved to: {args.trajectory_dir}/validation_results.json")
        
        # Print recommendations for non-compliant instances
        print(f"\n📋 Recommendations:")
        for instance_id, details in results['validation_details'].items():
            if not details['overall_compliant']:
                print(f"  {instance_id}:")
                for rec in details['recommendations']:
                    print(f"    - {rec}")
        
        return 1

if __name__ == "__main__":
    exit(main())
