#!/usr/bin/env python3
"""
Main training entry point for NUbots RL FYP
This script uses the extended multi-model training system with tensorboard support.
Usage: python train.py
"""

from src.training.extended_train_script import main

if __name__ == "__main__":
    print("=" * 80)
    print("NUbots RL FYP - Multi-Model Training System")
    print("This system will train multiple variants of PPO and DDPG algorithms")
    print("TensorBoard logs will be saved for monitoring training progress")
    print("=" * 80)

    # Run the main training function
    main()