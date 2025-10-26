Report Structure Guidelines (40% of the course grade)

1) Title (1 page): A concise and descriptive title of the thesis.
    Author: Full name of the student.
    Degree: Submitted in partial fulfilment of the requirements for the degree of Bachelor of Software Engineering (Honours).
    Affiliation: School of Information and Physical Sciences, The University of Newcastle, Callaghan, NSW 2308, Australia
    Date: Month and year of submission.
    You will generally also want to include the name and (brief) affiliation of your supervisor on the title page.

2) Abstract (1 page)
    Length: 250-300 words.
    A brief summary of the research problem, methodology, key findings, and significance of the study.

3) Table of Contents
    List of chapters, sections, and subsections with page numbers.

4) List of Figures, Tables and/or Algorithms (Optional)
    Titles and page numbers of all figures, tables, and or algorithms included in the report. Each of these lists, if included, should be separate.

5) Introduction (~ 5-7 pages)
    Background: Introduction to the topic and its importance in the field of software engineering.
    Problem Statement: Clear statement of the research problem or question.
    Objectives: Specific aims and objectives of the research.
    Scope: Discussion of the research scope, including any specific inclusions or exclusions.
    Significance: Importance and potential impact of the research, including its context within the state of the field.
    Outline: Briefly describe the structure of the report.

6) Background and Literature Review (~ 5-10 pages)
    Literature Review: Comprehensive and critical analysis of existing literature relevant to the topic.
    Scope: Cover the theoretical foundations and related work.
    Current Solutions: Discuss existing methodologies, technologies, and solutions.
    Critical Analysis: Evaluate the strengths, weaknesses, and gaps in the literature.
    Summary: Synthesise the findings to highlight the research gap your report will address.


    Trust Region Policy Optimisation (TRPO), the predecessor to PPO developed by Schulman et al. (2015), provides theoretical guarantees for monotonic policy improvement but at the cost of implementation and computational complexity. Some research suggests that TRPO may offer more robust performance in certain multi-agent
    scenarios where stability is crucial. However, due to PPO showing similar performance with simpler implementation, this algorithm was rejected. Modern deep reinforcement learning algorithms have significantly advanced the field's capabilities for handling complex, high-dimensional state and action spaces like those found in robot soccer.
    Proximal Policy Optimisation (PPO), introduced by Schulman et al. (2017), offers a balance of sample efficiency, implementation simplicity, and performance that makes it particularly suitable for robotics applications in both environments with discrete or continuous action spaces. 
    The algorithm implements a trust region constraint through a clipped surrogate objective function that limits policy changes at each update. This clipping mechanism prevents the destructive updates that plagued earlier policy gradient methods, enabling more stable learning with less hyperparameter sensitivity while allowing for continuous policy improvement. Deep Q-Networks (DQN) works well with environments with discrete actions spaces but does not do well with environments with continuous actions spaces which is where DDPG comes in since it combines Q-learning with a deterministic policy gradient to work well with continuous environments. Deep Deterministic Policy Gradient (DDPG) by Lillicrap et al. (2015) provides an actor-critic approach designed for continuous control which uses 2 neural networks: one to select an action and the other to evaluate the action based on value functions. DDPG's ability to handle continuous action spaces makes it potentially suitable for generating precise commands to the Motion layer for the NUbots system. 
    The algorithm maintains four neural networks: an actor that outputs actions, a critic that estimates action values, and target versions of each that stabilise learning. The actor is trained using the deterministic policy gradient, which provides an efficient mechanism for policy improvement in continuous spaces by following the gradient of the critic's value estimate with respect to actions.
The off-policy nature of DDPG enables substantial sample efficiency advantages over PPO. The algorithm maintains a replay buffer of past experiences and can learn from this data even when collected by previous policies. This capability is particularly valuable in domains where data collection is expensive, as the same experiences can be reused multiple times during training. However, DDPG exhibits higher sensitivity to hyperparameters compared to PPO, particularly learning rates and exploration noise parameters, sometimes requiring more extensive tuning to achieve stable learning.

    More recently, Soft Actor-Critic (SAC) by Haarnoja et al. (2018) has shown promising results in robot control tasks by incorporating entropy maximisation to encourage exploration while maintaining performance. SAC's sample efficiency and robustness to hyperparameter settings represent continued advancement in off-policy methods, though the current project focuses on the established implementations of the PPO and DDPG algorithms in the Stable-Baselines3 open source library due to time constraints.

    For this project, the focus narrows to PPO and DDPG as representative on-policy and off-policy algorithms, enabling controlled comparison of these algorithmic paradigms without expanding scope to include the full spectrum of contemporary RL methods. This selection provides sufficient coverage of major algorithmic approaches while maintaining project feasibility within resource and timeline constraints.

    ## Simulation Environments for Training
    Webots, developed by Cyberbotics (2004), provides a high-fidelity 3D physics simulation platform widely adopted for robotics research. The simulator incorporates realistic rigid-body dynamics, collision detection, sensor modeling (cameras, lidars, IMUs), and actuator characteristics (motor dynamics, joint limits). Webots supports importing complex robot models with accurate kinematic and dynamic parameters, enabling simulation of physical hardware behaviour. However, this simulation fidelity comes at substantial computational cost. The physics engine must solve contact dynamics, integrate equations of motion, and render 3D scenes at sufficient frequency for stable simulation (typically 50-100Hz). When training RL policies requiring millions of environmental interactions, this computational overhead becomes prohibitive. Training times extend to days even with GPU acceleration.

    MuJoCo (Multi-Joint dynamics with Contact) similarly provides advanced physics simulation optimised for robotics and biomechanics applications. The simulator implements fast and accurate contact dynamics using convex optimisation-based methods, enabling stable simulation of complex multi-body systems with numerous contacts. MuJoCo has become a standard benchmark platform for continuous control RL research, with extensive algorithm comparisons conducted in MuJoCo environments.
    Like Webots, MuJoCo's simulation realism brings computational overhead for training. Additionally, both simulators require extensive parameter tuning to achieve accurate modelling of specific robot hardware like calibrating friction coefficients, motor characteristics, sensor noise profiles, and other physical parameters to match real-world measurements. This calibration process itself represents a significant engineering investment before training can commence.

    Recent developments include RoboCup Gym Beukman et al. (2024), a Python-based RL environment for the RoboCup 3D simulation league that integrates with SimSpark and demonstrates successful training of NAO robots using PPO and SAC algorithms. However, RoboCup Gym's specific design for NAO robot morphology and SimSpark simulator would require extensive adaptation work to integrate with NUbots' different robot architecture and existing simulation infrastructure.

    Given the mentioned computational demands of high-fidelity simulation and project timeline constraints, this research elected to develop a custom 2D simulation environment (SoccerEnv) balancing simplicity with sufficient realism for proof-of-concept demonstration. This simplification reduces simulation complexity by eliminating balance control, vertical dynamics, and 3D perception, focusing learning only on strategic decision-making which is the main contribution.
The custom environment provides several advantages for rapid experimentation and iterative changes between training. Training times decrease by an order of magnitude compared to 3D simulation, enabling faster iteration on reward function design, hyperparameter tuning, and algorithm comparison. The simplified physics allows for easy understanding and debugging about learned behaviours, as the reduced state space complexity makes policies more interpretable. Development flexibility allows customisation of environment dynamics, opponent behaviours, and game mechanics to isolate specific learning challenges without navigating the complexity of general-purpose simulator APIs.
However, the 2D simplification introduces limitations towards physical deployment. The learned policies will not directly transfer to 3D humanoid robots, as critical aspects of soccer play (maintaining balance while moving, visual perception of 3D ball trajectory, coordinating multi-joint motor control) are abstracted away. The project acknowledges this limitation, positioning the work as a proof-of-concept for RL-based strategic decision-making rather than a complete robot soccer solution. The modular architecture separates strategic planning from low-level control, enabling future integration with 3D simulators and physical robots by replacing the abstract action interface with connections to existing motion controllers.

    TRANSFER FRMO SIMILATION TO REALITY
    A critical challenge for simulation-trained RL systems is the domain gap which is the discrepancy between simulated and real-world dynamics that can cause policies trained in simulation to fail when deployed on physical robots. This gap arises from multiple sources: unmodeled physics in simulation, sensor noise and delays, actuator imperfections, and environmental variations that are common in the real-world.
    
    Tobin et al. (2017) introduced domain randomisation as an approach to bridge the reality gap, randomly varying simulation parameters during training to create robust policies that can generalise to real-world conditions. Their method randomises visual appearance (textures, lighting, colours), object properties (masses, friction coefficients), and camera parameters (position, focal length) across training episodes. The resulting policy must perform successfully across this wide distribution of simulation conditions, implicitly learning features and strategies robust to the specific parameter variations encountered in reality.
Their experiments demonstrated successful transfer of visually-guided robotic manipulation policies from simulation to physical robots. Policies trained with domain randomisation generalised to real-world visual appearance without requiring any real-world training data, validating the hypothesis that sufficient simulation diversity can encompass real-world variation. However, the approach requires careful selection of which parameters to randomise and their ranges since too little randomisation fails to cover real-world variation, while excessive randomisation may prevent learning altogether by making the task impossible.

Desai et al. (2020) extended the Grounded Action Transformation (GAT) approach by introducing Stochastic Grounded Action Transformation (SGAT), which explicitly accounts for randomness in real-world environments during sim-to-real transfer. While GAT learns deterministic forward models predicting the next state given current state and action, SGAT models the forward dynamics as distributions over next states rather than single predictions, capturing the inherent randomness in physical systems. Deterministic models may predict the mean next state accurately but fail to capture the uncertainty range, leading to policies that work well for deterministic dynamics but fail when encountering variations. Stochastic models explicitly represent this uncertainty, enabling learning of policies robust to the range of possible outcomes rather than optimised for a single predicted trajectory.
In physical robot experiments with a NAO humanoid walking on uneven terrain, SGAT-trained policies completed the course 9 out of 10 times compared to GAT policies which fell down in every trial. This demonstrates the importance of modelling real-world stochasticity for successful sim-to-real transfer. The uneven terrain introduced random perturbations to robot dynamics that deterministic GAT models failed to capture, while SGAT's distributional modeling enabled robust policy learning.

7) Methodology and Design (~10-20 pages)
    You can consider splitting this into two sections, research methodology and software engineering methodology, if you find it flows better.
    Research Design: Description of the research approach (qualitative, quantitative, or mixed methods).
    Data Collection: Methods used for data collection (e.g., surveys, experiments, simulations).
    Data Analysis: Techniques and tools used for data analysis. Include details of any statistical hypothesis tests that will be conducted.
    Validation and Verification: Outline the methods for ensuring the validity and reliability of the research and methodology.
    Design Specifications: Provide detailed design specifications for the software solution, including architectural diagrams, UML diagrams, and/or system models, as appropriate.
    Software Development Methodology: Outline and contextualise the adopted software development methodology, particularly why it was selected and how it was adapted (if applicable) to the project at hand.
    Ethical Considerations: Ethical issues and how they were addressed, if applicable.


    \section{Research Design}
    This project employed an approach involving simulated experiments to test ideas and hypotheses to evaluate reinforcement learning algorithms for strategic behaviours based on quantitative methods. The experimental design compared three approaches across multiple training runs:
    \begin{itemise}
        \item DDPG-trained agent: Deep Deterministic Policy Gradient algorithm
        \item PPO-trained agent: Proximal Policy Optimisation algorithm
        \item Random baseline: Agent selecting random actions from action space
    \end{itemise}

    Each algorithm was trained for 2,500,000 timesteps in the custom SoccerEnv simulation environment, with evaluations being conducted every 10,000 timesteps using 20 independent episodes. This design allowed for fair comparisons between policies while isolating the effect of learning from within a simulation environment.
    The simulation-based approach was selected over testing on physical robots for rapid experimentation, algorithm comparison under identical conditions, and safe exploration of potentially damaging behaviours during early training. While this introduces sim-to-real transfer challenges, it provides a validated foundation for future physical deployment.

    Data collection was performed using automated logging during reinforcement learning training and evaluation episodes.
    Training Data Collection
    During training, the following data was automatically recorded for each episode:

    Episode return: Cumulative reward per episode
    Episode length: Number of timesteps before termination
    Success indicators: Goal scored (binary), ball possession achieved (binary)
    Policy metrics: Policy loss, value function loss, entropy coefficient (for PPO)
    Learning dynamics: Learning rate, gradient norms, KL divergence

    Evaluation Data Collection
    Systematic evaluation occurred every 10,000 training steps using a separate evaluation environment to prevent training data contamination. Each evaluation consisted of 20 independent episodes with the following metrics recorded:
    Performance metrics: Mean episode reward, standard deviation, success rate
    Behavioural metrics: Ball possession time (%), collision frequency, goal approach distance, ball out of bounds frequency
    Efficiency metrics: Average episode length, time to first goal contact
    The final evaluation for statistical comparison used 100 episodes per algorithm to ensure adequate statistical power for significance testing.

    Data was collected using the Stable-Baselines3 library's Monitor wrapper and custom callback functions, with outputs logged to timestamped directories organised by training run.

    Configuration and Reproducibility
    The simulations in SoccerEnv are random in nature since noise is incorporated into the simulations by randomising positions of the ball and the robots. Because of this, different runs of simulation with the same settings may generate different results. All training runs used version-controlled YAML configuration files specifying:
    Algorithm hyperparameters (learning rates, network architectures, batch sizes)
    Environment parameters (field dimensions, robot speed, reward weights)
    Training settings (total timesteps, evaluation frequency, checkpoint intervals)

    Test cases related to training runs used fixed random seeds (seed=42) for systematic validation enabled similar reproduction of experimental results and hyperparameter variation across experimental conditions.

    Data Analysis
    Quantitative Comparative Performance Analysis
    Algorithm performance was compared using independent samples t-tests to assess whether observed performance differences were statistically significant:

    Null hypothesis: μ_algorithm1 = μ_algorithm2 (no performance difference)
    Alternative hypothesis: μ_algorithm1 ≠ μ_algorithm2 (significant difference)
    Significance level: α = 0.05 (meaning 5%)

    Effect sizes were quantified using Cohen's d to measure practical significance beyond statistical significance:

    Small effect: |d| = 0.2
    Medium effect: |d| = 0.5
    Large effect: |d| = 0.8

    95% confidence intervals were calculated for mean performance metrics to estimate population-level performance ranges and support generalisation claims.

    Data collected during training and testing runs can differ from when visualised using the human render mode of the simulation. Such experiments can hide details for example, out of 100 episodes, a model can score a goal in 30 episodes indicating a 30% success rate however, during a simulation it can be visualised that the goal was either scored by the opponent or was scored by chance. Another example, is that during training, it could be observed that both DDPG and PPO models are achieving similar rewards, however when loaded and visualised during evaluation, the performance of the models can differ greatly such as one model exploiting the reward function ("reward hacking") while the other learned optimal behaviours. 
    Hence performance metrics such as goals scored, ball possession rate, collision frequency etc were collected to further evaluate the behaviours learned by the models.


Training Convergence Analysis
Training curves were plotted to assess learning stability and convergence:
Moving average smoothing: 100-episode window to reduce noise
Coefficient of variation: Standard deviation / mean < 0.20 indicates stable learning
Plateau detection: Performance improvement < 1% over 200,000 timesteps

Statistical Validation Results
Key statistical findings included:
DDPG vs Random: t=12.38, p<0.0001, d=5.3 (very large effect, 1,255% improvement)
PPO vs Random: t=0.72, p=0.24, d=0.27 (no significant difference)
DDPG vs PPO: t=-2.60, p=0.013, d=-0.82 (large effect favoruing DDPG)

These statistical tests validated that DDPG learned soccer strategies that significantly exceeded both random baseline and PPO performance.



Validation and Verification:
Multiple verification and validation methods were used to ensure robustness of the research methodology followed. Unit testing was carried out to validate the functionality of the individual units. These included environment physics validations: to check the observation space bounds, clipping of actions and deterministic state transitions such as ball and robot movements, collision detection; state transformations validated through checking the coordinate system used, angle wrapping; reward function tests to measure the weightage between the reward components, reward bounds validation. Integration tests were carried out to validate the training pipeline like PPO and DDPG training complete without errors, checkpoint creation, log and monitor generation during training, training difficulty transitions, training completion; ONNX conversion process, inference accuracy and determinism, along with output latency, metrics collection accuracy across multiple episodes, model persistence. Performance tests and model evaluation was carried out to compare trained models with random baseline (actions picked at random), validation that onnx and pytorch model outputs match within threshold value of 10^-5, and model prediction consistency. Performance benchmarks establised performance targets such as:
Ball Possession: DDPG 85%, PPO 25.3%, Random 1.78%
Collision Avoidance: DDPG 0%, PPO 3.2/episode, Random 8.5/episode
Success Rate: DDPG 68%, PPO 15.5%, Random 2%

Software Development Methodology
The project followed an modified version of the Agile development methodology adapted for machine learning research. It was selected for iterative development sprints, flexibility to adapt to results from experiments, and prioritising development over documentation. Due to project being based on research, it was necessary to conduct research on experiments simultaneously with development. Each sprint began with a meeting with a senior NUbots member to decide what could be incorporated into the current system and ideas to improve training results. This gave flexibility to the development sprints but to still keep track of the high level goal to be achieved.
Each sprint lasted for 2 week iterations which focused on experimental ideas rather than defined features. The end of each sprint checked current progress against success criteria. To ensure development was progressing in the correct direction testing was conducted via Pytest execution on each commit to guide implementation. It allowed for increments in complexity, for example, when progressing environment difficulty from a toy problem to the soccer environment (CartPole→ SoccerEnv) which helped validate implementations before domain-specific challenges. However the project scope focused on a single trained agent learning how to dribble the ball to score a goal without weaving other complex behaviours such as switching to defense in case the opponent robot had the ball. To facilitate this and the possibility that not all the project's milestones would be completed, a flexible methodology was used.
During the planning stages, other alternatives were discussed. A traditional waterfall methodology was rejected due to its rigid linear progression which would hinder any developments for reinforcement learning due to the constant need for adjustments to algorithm hyperparameters, reward functions and network architectures. In addition, testing would need to be conducted at the end rather than in conjunction with development. A pure scrum workflow was rejected due to the spirit of scrum being incremental developments when in reality progress in reinforcement learning is unpredictable as well as the project being undertaken by a single developer. 


Design Specifications:
System Architecture:
The system architecture used follows a pipeline-based architecture where components pass data through from the configuration loader to the training pipeline to the visualisation system and export via ONNX conversion pipeline. The architecture came from the sequential nature of RL training workflows.
The Configuration Loader loads values and parameters from YAML files and providing validated configuration objects to the simulation environment. The training pipeline orchestrates an end-to-end training using the implementations of the RL algorithms from the open source Stable-Baselines3 library, managing environment instances, RL agents and monitoring of training. The Visualisation system generates training plots and comparative analysis visualisations between the algorithms based on performance metrics. The Evaluation system consists of the test suite used for quality assurance of the whole system being developed. The ONNX conversion pipeline is responsible for converting models to the ONNX format which is the format required for the models to be integrated into the NUbots system.
The architecture makes use of the principle of separation of concerns as evidenced by the simulation environment which focuses on simulation physics, state observation with no knowledge of the training workflows. The training pipeline coordinates the high-level training flow without implementing RL algorithms directly since it uses the working SB3 implementations. RL agents from SB3 handle learning, while the visualisation and evaluation components operate independently used during training or model evaluation for result analysis. 
The architecture used has loose coupling between the components where environments expose only the Gymnasium interface while abstracting the implementation details. This allows for environment substitution without modifying the pipeline. Figure <Figure number> shows the major components and the interactions between them. The Command-Line Interface (CLI) is the system entry point which uses user specified arguments to execute the training flow. It support flags for algorithm selection, number of episodes to train for, configuration files to load and the help flag provides usage documentation and examples. This design allows for experimentation and automated traning runs via shell scripts. It supports multiple execution modes which include training mode to train models which can be done for new models or to continue training from a checkpoint of an existing model. Evaluation mode loads trained models and runs test episodes with the environment render set to human mode for visualisation. Compare mode runs multiple models (specified by the user) against standard benchmarks and generates a comparison between based on performance metrics. 
The Configuration Loader loads two main configuration files: one for environment parameters like field dimensions (9m × 6m matching RoboCup SPL specifications), robot physical properties, ball dynamics, opponent behaviour types, and rendering options; while the other defines the hyperparameters for the training algorithms like PPO (learning rates, batch sizes, etc.) and DDPG (replay buffer sizes, noise parameters, etc.). The configuration loader creates structured objects for other components to consume. 

The SoccerEnv simulation environment implements the core physics

Figure X shows the container-level architecture of the training system. The Training Pipeline container's internal workflow is detailed in Figure Y, which illustrates how it orchestrates the RL algorithms from Stable-Baselines3 with custom callbacks to implement the training loop and convergence monitoring described in the container responsibilities.



Ethical Considerations
While this research involved only simulated robots, several ethical considerations were addressed:
Data and Privacy: All data was synthetically generated in simulation. No human subjects were involved and no personally identifiable information was collected.
Computational Resources: Training consumed approximately 80-100 CPU-hours total. GPUs were not used although this was due to scheduling conflicts however renewable energy contributed to the system being used for training.
Future Safety: Although physical robot integration was not completed, reward functions were designed to discourage aggressive collision-inducing behaviours. ONNX conversion was rigorously validated to prevent deployment of incorrectly converted models especially in the case PPO models which were non-deterministic.
Reproducibility: All configurations, random seeds, and hyperparameters were version-controlled and documented to enable result reproduction and support academic integrity.
Public use: The repository for the project was made public under the MIT license like the open-source NUbots codebase for any developer to view and experiment with.




8) Implementation and Evaluation (~10-20 pages)
    Development Process: Describe the software development process, including technologies used, development environment, and coding practices.
    Implementation Details: Provide detailed descriptions of key modules, algorithms, and workflows.
    Challenges and Solutions: Discuss any challenges faced during implementation and how they were addressed.
    Evaluation Methods and Criteria: Outline the methods, criteria, and metrics used to evaluate the software solution, including testing strategies, user feedback, and performance metrics.
    Testing: Describe the testing methodology and results.

## Implementation continued:
### This section for the simulatioon environment
The simualtion environment extends the Gymnasium interface and exposes functions such as:
1) step() : function for the model to take the next action
2) reset() : function to reset the environment back to the initial state
3) calculate_reward: function to give rewards and penalties depending on the agent's current state
4) render() : function to provide a visualisation for human eyes which can provide ease in debugging when compared to viewing the rewards trends alone
5) and so on?

The step function was broken down into multiple helper functions such as the:
apply_robot_action :function which handled updating the robot's position and velocoty based on the chosen action and the state before it takes the action.
update_ball_physics: functoin which handled updating the ball's position, velocity and states
update_opponent: function to update the opponent AI with different behavioural configs such as the agressive, defensive and balanced behaviours which differ in whether it seeks ball possession from the robot, or ensures that it comes between the robot and its goal or a hybrid behaviour between the 2 depending on a distance threshold value of (TODO: Mention how it was implemented)
update_ball_possession_flags: function to update boolean flags for which robot has possession of the ball or neither 
calculate_reward function to calculate rewards and penalties and returns a final value
check_terminated: function to check whether any of the consitions for an episide to be considered as completed (success or fialure)
These conditions include, whether either robot scored a goal, if the 2 robots collided, if the ball went out of bounds


TODO: Define timestep, episode and other terms in the Background section
A timestep is an individual action that the agent can take.
An episode is a collection of timesteps until an action was taken that causes the end of a simulation scenario such as goals scored, ball out of bounds, etc.

A separate training script was developed to create a PPO or DDOG model, where at the begining og each episode, the positions of the robots and the ball was randomised for the models to be able to generalise to new unseen scenarios. the number of timesteps could be specified. It was noticed that DDPG would a longer time duration to train than PPO.

To view training logs and the metrics such as mean episode length, or mean episode reward, tensorboard logs were used to visualise the training trends in real-time. This was invaluabe in determining any areas or situations that needed patching wheyher the agent found a way to exploit the reward system by technically not doing the wrong thing but not doing the right thing either








#TODO: Look at old version of the repo for the old reward function to describe the approach taken when training models
USing a model trained on the previously incomplete soccerenv, the performance of the model was observed to be unsatisfactory given that the model would struggle to get ball possession, or would collide with the opponent, or spin around in a pth repeatedly. After fine-tuning it to be trained on the newer completed environment, the model perfirmance was observed to be grestly improved showing a 100% success rate in seeking ball possession, and maneouvering the ball towards the opponent's goal, however due to the reward function used which penalised the agent if the opponent was too close, the agent would back-away if it could not dribble the ball past the pponent. After updating the reward function to be more lenient with the penalties and adding new rewards to encourage a more 'aggressive' dribbling behaviour, the agent seemed to be getting closer towards the goal than previously observed, however still could not score, due to the difficulty of the opponent AI's behaviour and the opponent being too fast and easily catching up with the robot. After experimenting with different environment dynamics such as reducing the friction value, it was observed that the robot could move faster and push the ball and scored goals in 3/5 testing episodes which is a 60% success rate.
It was also observed that the dynamics related to the robot's movement was implemented from the robot's perspective relative to the soccer field which made it difficult to validate whthe accuracy of the environment with the real robots would use as the model was trained in the soccerenv environment and since it differed from the real scenario, the model would not transfer well to the real world.
Using a script to map the robot's actions to keyboard keys, the movements of the robot were found to inconsistent with what was expected. To work around this, the environment dyanmics were changed to use the world coordinate system where the action space used x,y,theta (x -> velocity along the x-axis relative to the world coordinate system)
(y -> velocity along the y-axis relative to the world coordinate system)
(theta -> TODO: Is this correct:  : :: : angular velocity in the counter-clockwise direction relative to the world coordinate system)

After validation of the robot's movements, new models were trained and it was observed that .....
The rewards for the goal progress component of the movements were contributing to approx ~ 85-98% of the total rewards. This illustrates that the agent learnt to optimise the goal distance but ignores ball control, positioning which were expected to be learnt. 




9) Results and Discussion (~10-15 pages)
    Presentation of Data: Detailed presentation of the research findings, including tables, graphs, and figures.
    Analysis and Interpretation: In-depth analysis and interpretation of the results and their implications in relation to the research objectives.
    Contributions: Highlight the contributions of the research to the field of software engineering.
    Comparison with Literature: Comparison of findings with existing literature.
    Limitations: Discussion of the limitations of the study.

    # VALIDATE:
    The DDPG model achieved exceptional cumulative reward performance (18,109 ± 2,812) compared to random baseline (993 ± 1,521), representing a 1,724% improvement (p < 0.0001). However, ball possession rates remained low (0.10% ± 0.23%), indicating the learned policy prioritizes goal-scoring behaviors and strategic positioning over continuous ball control. This demonstrates the multi-objective trade-off inherent in the reward function design, where the agent optimizes for episode success (goals scored, game won) rather than intermediate metrics (ball possession). This finding aligns with research on sparse reward learning, where agents learn to achieve terminal objectives without necessarily maximizing all intermediate performance indicators.

10) Conclusions (~3 pages)
    Summary: Summary of the main findings.
    Implications: Practical and theoretical implications of the research.
    Recommendations: Suggestions for future research or practice.

11) References
    Comprehensive list of all sources cited in the report, formatted according to a standard citation style (e.g., IEEE).

12) Appendices (Optional)
    Supplementary materials such as raw data, detailed calculations, questionnaires, or additional figures and tables.