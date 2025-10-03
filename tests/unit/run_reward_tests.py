#!/usr/bin/env python3
"""
Simple test runner for reward function tests
Can be used without pytest installation

Student: Ali Riyaz (C3412624)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import test classes
from test_reward_function import (
    TestRewardFunctionBounds,
    TestRewardTerminalConditions,
    TestAntiExploitationMeasures,
    TestRewardComponents,
    TestRewardScaling
)


def run_test_class(test_class, class_name):
    """Run all tests in a test class"""
    print(f"\n{'='*70}")
    print(f"Running: {class_name}")
    print('='*70)

    test_instance = test_class()
    test_methods = [m for m in dir(test_instance) if m.startswith('test_')]

    passed = 0
    failed = 0

    for method_name in test_methods:
        try:
            # Setup
            if hasattr(test_instance, 'setup_method'):
                test_instance.setup_method()

            # Run test
            method = getattr(test_instance, method_name)
            print(f"\n  {method_name}...", end=' ')
            method()
            print("✓ PASSED")
            passed += 1

            # Teardown
            if hasattr(test_instance, 'teardown_method'):
                test_instance.teardown_method()

        except Exception as e:
            print(f"✗ FAILED")
            print(f"    Error: {str(e)}")
            failed += 1

            # Teardown even on failure
            if hasattr(test_instance, 'teardown_method'):
                try:
                    test_instance.teardown_method()
                except:
                    pass

    print(f"\n  Results: {passed} passed, {failed} failed")
    return passed, failed


def main():
    """Run all reward function tests"""
    print("\n" + "="*70)
    print("REWARD FUNCTION UNIT TESTS")
    print("Soccer RL FYP - Ali Riyaz (C3412624)")
    print("="*70)

    total_passed = 0
    total_failed = 0

    # Run each test class
    test_classes = [
        (TestRewardFunctionBounds, "TestRewardFunctionBounds"),
        (TestRewardTerminalConditions, "TestRewardTerminalConditions"),
        (TestAntiExploitationMeasures, "TestAntiExploitationMeasures"),
        (TestRewardComponents, "TestRewardComponents"),
        (TestRewardScaling, "TestRewardScaling"),
    ]

    for test_class, name in test_classes:
        passed, failed = run_test_class(test_class, name)
        total_passed += passed
        total_failed += failed

    # Summary
    print("\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)
    print(f"Total tests passed: {total_passed}")
    print(f"Total tests failed: {total_failed}")
    print(f"Success rate: {total_passed/(total_passed+total_failed)*100:.1f}%")
    print("="*70 + "\n")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())