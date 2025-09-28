# CODE FROM src/training/extended_training_script.py
# Code that was used to continue the training of existing models
# with modified parameters MODIFIED REWARD FUNCTIONS, longer durations, or different environments.
def continue_ddpg_training(training_system, model_path, run_name, total_timesteps=800000):
    """Continue training an existing DDPG model with comprehensive logging"""
    
    training_system.logger.info(f"Loading existing DDPG model: {model_path}")
    training_system.logger.info(f"Continue training run: '{run_name}'")
    
    try:
        # Load the existing model
        model = DDPG.load(model_path)
        training_system.logger.info("Model loaded successfully!")

        initial_timesteps = model.num_timesteps  # Remember where we started
        print(f"🔍 Starting retraining from timestep: {initial_timesteps}")
        
        # Create environments
        train_env = Monitor(SoccerEnv(difficulty="easy", config_path=training_system.config_path))
        eval_env = Monitor(SoccerEnv(difficulty="easy", config_path=training_system.config_path))
        
        # Update model's environment and tensorboard logging
        model.set_env(train_env)
        model.tensorboard_log = f"{training_system.output_dir}/tensorboard_logs"

        # Setup callbacks
        # model_tracker = ModelTracker("DDPG_Continue", training_system, run_name, verbose=1)
        model_tracker = ModelTracker("DDPG", training_system, run_name, verbose=1)
        model_tracker.set_model(model)
        
        train_env.total_timesteps_trained = model.num_timesteps  # Pass current timesteps to env
        train_env.initial_timesteps = initial_timesteps  # Store the starting point
        train_env.relative_timesteps = 0  # Reset relative timesteps
        model_tracker.train_env = train_env  # Pass env to tracker for timestep tracking

        # ADD THIS DEBUG:
        print(f"🔍 train_env type: {type(train_env)}")
        print(f"🔍 train_env has total_timesteps_trained: {hasattr(train_env, 'total_timesteps_trained')}")
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{training_system.output_dir}/models/ddpg/",
            log_path=f"{training_system.output_dir}/logs/",
            eval_freq=15000,  # Evaluate every 15k steps
            deterministic=True,
            render=False,
            n_eval_episodes=5,
            verbose=1
        )
        
        # Train model (continue from existing state)
        training_system.logger.info(f"Continue training {run_name} for {total_timesteps:,} additional timesteps...")
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=[model_tracker, eval_callback],
            tb_log_name=None,
            progress_bar=True,
            reset_num_timesteps=False  # Continue from existing timestep count
        )
        
        training_time = time.time() - start_time
        training_system.logger.info(f"{run_name} continued training completed in {training_time/3600:.2f} hours")
        
        # Final evaluation and save
        final_score = model_tracker._comprehensive_evaluation()
        final_model_path = f"{training_system.output_dir}/models/ddpg/{run_name}_{training_system.timestamp}"
        model.save(final_model_path)
        
        # Save training information
        continue_info = {
            'original_model_path': model_path,
            'run_name': run_name,
            'additional_timesteps': total_timesteps,
            'training_time_hours': training_time/3600,
            'final_score': final_score,
            'continued_model_path': final_model_path,
            'timestamp': training_system.timestamp
        }
        
        info_file = f"{training_system.output_dir}/summaries/continued_{run_name}_info.json"
        with open(info_file, 'w') as f:
            json.dump(continue_info, f, indent=2)
        
        training_system.logger.info(f"Continued model saved: {final_model_path}")
        training_system.logger.info(f"Final score: {final_score:.2f}")
        
        return final_model_path, final_score, run_name
        
    except Exception as e:
        training_system.logger.error(f"Continue training {run_name} failed: {e}")
        return None, -np.inf, run_name
        
    finally:
        train_env.close()
        eval_env.close()
    

def train_new_ddpg_model(training_system, run_name, hyperparams=None, total_timesteps=1500000, n_envs=4):
    """Train a brand new DDPG model from scratch with comprehensive logging"""
    
    training_system.logger.info(f"Creating new DDPG model from scratch")
    training_system.logger.info(f"New training run: '{run_name}'")
    
    # Use default hyperparameters if none provided
    if hyperparams is None:
        hyperparams = {
            "learning_rate": 1e-3,
            "buffer_size": 1000000,
            "learning_starts": 25000,
            "batch_size": 256,
            "tau": 0.01,
            "gamma": 0.99,
            "noise_sigma": 0.1,
            "train_freq": (2, "step"),
            "gradient_steps": 1,
            "net_arch": [256, 256, 128]
        }
    
    training_system.logger.info(f"Hyperparameters: {hyperparams}")
    
    def make_env(rank):
        """Create a single environment - this runs in separate process"""
        def _init():
            env = SoccerEnv(
                difficulty="easy", 
                config_path=training_system.config_path,
                render_mode=None  # IMPORTANT: No rendering in parallel envs
            )
            env = Monitor(env)
            return env
        return _init
    
    try:
        # Create environments
        # Create parallel training environments
        train_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
        
        # Single evaluation environment (non-parallel for consistency)
        eval_env = Monitor(SoccerEnv(
            difficulty="easy", 
            config_path=training_system.config_path,
            render_mode=None
        ))  

        # Action noise for exploration
        n_actions = train_env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=hyperparams["noise_sigma"] * np.ones(n_actions)
        )
        
        # Create brand new DDPG model
        model = DDPG(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=hyperparams["learning_rate"],
            buffer_size=hyperparams["buffer_size"],
            learning_starts=hyperparams["learning_starts"],
            batch_size=hyperparams["batch_size"],
            tau=hyperparams["tau"],
            gamma=hyperparams["gamma"],
            action_noise=action_noise,
            train_freq=hyperparams["train_freq"],
            gradient_steps=hyperparams["gradient_steps"],
            policy_kwargs=dict(
                net_arch=hyperparams["net_arch"],
                activation_fn=torch.nn.ReLU
            ),
            verbose=1,
            device="cpu",
            tensorboard_log=f"{training_system.output_dir}/tensorboard_logs"
        )
        
        training_system.logger.info("New DDPG model created successfully!")
        
        # Setup callbacks
        model_tracker = ModelTracker("DDPG_Parallel", training_system, run_name, verbose=1)
        model_tracker.set_model(model)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{training_system.output_dir}/models/ddpg/",
            log_path=f"{training_system.output_dir}/logs/",
            eval_freq=15000,  # Evaluate every 15k steps
            deterministic=True,
            render=False,
            n_eval_episodes=5,
            verbose=1
        )
        
        # Train model from scratch
        training_system.logger.info(f"Training new {run_name} for {total_timesteps:,} timesteps...")
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=[model_tracker, eval_callback],
            tb_log_name=run_name,  # Custom tensorboard log name
            progress_bar=True,
            reset_num_timesteps=True  # Start from timestep 0
        )
        
        training_time = time.time() - start_time
        training_system.logger.info(f"{run_name} training completed in {training_time/3600:.2f} hours")
        
        # Final evaluation and save
        final_score = model_tracker._comprehensive_evaluation()
        final_model_path = f"{training_system.output_dir}/models/ddpg/new_{run_name}_{training_system.timestamp}"
        model.save(final_model_path)
        
        # Save training information
        new_model_info = {
            'run_name': run_name,
            'hyperparameters': hyperparams,
            'total_timesteps': total_timesteps,
            'training_time_hours': training_time/3600,
            'final_score': final_score,
            'model_path': final_model_path,
            'timestamp': training_system.timestamp,
            'training_type': 'parallel',
            'device': "cpu"
        }
        
        info_file = f"{training_system.output_dir}/summaries/new_{run_name}_info.json"
        with open(info_file, 'w') as f:
            json.dump(new_model_info, f, indent=2)
        
        # Save hyperparameters separately for easy access
        hyperparam_file = f"{training_system.output_dir}/hyperparameters/new_{run_name}_hyperparams.json"
        with open(hyperparam_file, 'w') as f:
            json.dump({k: (v if not isinstance(v, np.ndarray) else v.tolist()) 
                      for k, v in hyperparams.items()}, f, indent=2)
        
        training_system.logger.info(f"New model saved: {final_model_path}")
        training_system.logger.info(f"Final score: {final_score:.2f}")
        
        return final_model_path, final_score, run_name
        
    except Exception as e:
        training_system.logger.error(f"New model training {run_name} failed: {e}")
        return None, -np.inf, run_name
        
    finally:
        train_env.close()
        eval_env.close()