#!/usr/bin/env python3
"""
Main evaluation entry point for NUbots RL FYP
Usage: python evaluate.py
"""

from src.evaluation.test_trained_model import main

if __name__ == "__main__":
    print("=" * 80)
    print("NUbots RL FYP - Model Evaluation System")
    print("Test and compare trained PPO and DDPG models")
    print("=" * 80)

    # Run the main evaluation function
    main()