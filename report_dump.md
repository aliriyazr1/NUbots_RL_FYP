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

7) Methodology and Design (~10-20 pages)
    You can consider splitting this into two sections, research methodology and software engineering methodology, if you find it flows better.
    Research Design: Description of the research approach (qualitative, quantitative, or mixed methods).
    Data Collection: Methods used for data collection (e.g., surveys, experiments, simulations).
    Data Analysis: Techniques and tools used for data analysis. Include details of any statistical hypothesis tests that will be conducted.
    Validation and Verification: Outline the methods for ensuring the validity and reliability of the research and methodology.
    Design Specifications: Provide detailed design specifications for the software solution, including architectural diagrams, UML diagrams, and/or system models, as appropriate.
    Software Development Methodology: Outline and contextualise the adopted software development methodology, particularly why it was selected and how it was adapted (if applicable) to the project at hand.
    Ethical Considerations: Ethical issues and how they were addressed, if applicable.

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

10) Conclusions (~3 pages)
    Summary: Summary of the main findings.
    Implications: Practical and theoretical implications of the research.
    Recommendations: Suggestions for future research or practice.

11) References
    Comprehensive list of all sources cited in the report, formatted according to a standard citation style (e.g., IEEE).

12) Appendices (Optional)
    Supplementary materials such as raw data, detailed calculations, questionnaires, or additional figures and tables.