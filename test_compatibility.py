#!/usr/bin/env python3
"""
Test script to verify the submission works without psutil dependency.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_without_psutil():
    """Test that the submission works even without psutil"""
    print("🧪 Testing submission without psutil...")
    
    # Mock psutil as unavailable
    import main
    main.PSUTIL_AVAILABLE = False
    
    # Test ResourceMonitor
    print("  Testing ResourceMonitor without psutil...")
    resource_monitor = main.ResourceMonitor()
    resources = resource_monitor.check_resources()
    print(f"    Resources: {resources}")
    assert resources['memory_mb'] == 0.0, "Should return 0 when psutil unavailable"
    
    # Test PerformanceMonitor
    print("  Testing PerformanceMonitor without psutil...")
    perf_monitor = main.PerformanceMonitor()
    perf_monitor.start_timer("test")
    import time
    time.sleep(0.1)
    perf_monitor.end_timer("test", True)
    
    summary = perf_monitor.get_performance_summary()
    print(f"    Performance summary generated: {len(summary)} characters")
    
    # Test TrajectoryLogger
    print("  Testing TrajectoryLogger without psutil...")
    from trajectory_logger import TrajectoryLogger
    
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = TrajectoryLogger(temp_dir)
        instance_id = "test_no_psutil"
        
        logger.log_step(instance_id, 1, "test_action", {
            "test_data": "test_value"
        })
        
        logger.log_final_result(instance_id, "test patch", True, "Success!")
        
        trajectory = logger.get_trajectory(instance_id)
        assert len(trajectory) == 2, f"Expected 2 steps, got {len(trajectory)}"
        
        perf_summary = logger.get_performance_summary(instance_id)
        assert perf_summary['total_steps'] == 2, "Performance summary incorrect"
    
    print("✅ All tests passed without psutil!")
    return True

def test_import_safety():
    """Test that all imports work correctly"""
    print("🧪 Testing import safety...")
    
    try:
        # Test main imports
        from main import run, PerformanceMonitor, ResourceMonitor, SmartCache, Network
        print("  ✅ Main imports successful")
        
        # Test agent imports
        from agent import agent_main
        print("  ✅ Agent imports successful")
        
        # Test trajectory logger imports
        from trajectory_logger import TrajectoryLogger
        print("  ✅ TrajectoryLogger imports successful")
        
        # Test patch generator imports
        from patch_generator import PatchGenerator
        print("  ✅ PatchGenerator imports successful")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Enhanced SWE-Bench Submission Compatibility")
    print("=" * 60)
    
    try:
        # Test imports first
        if not test_import_safety():
            print("❌ Import tests failed")
            return 1
        
        print()
        
        # Test without psutil
        if not test_without_psutil():
            print("❌ psutil-free tests failed")
            return 1
        
        print()
        print("=" * 60)
        print("🎉 All compatibility tests passed!")
        print("\n📋 Test Summary:")
        print("  ✅ Import safety - All modules import correctly")
        print("  ✅ psutil-free operation - Works without psutil")
        print("  ✅ ResourceMonitor - Graceful degradation")
        print("  ✅ PerformanceMonitor - Works without psutil")
        print("  ✅ TrajectoryLogger - Works without psutil")
        
        print("\n🚀 Enhanced SWE-Bench submission is compatible!")
        print("   - Handles missing dependencies gracefully")
        print("   - Falls back to basic functionality when needed")
        print("   - Maintains core functionality without optional features")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
