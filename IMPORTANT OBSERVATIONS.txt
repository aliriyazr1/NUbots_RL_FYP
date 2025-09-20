###############################################################################################
(REMEMBER TO KEEP TRACK OF WHAT HYPERPARAMETERS GET USED AND THEIR VALUES FOR THE THESIS STUPID ALI)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
###############################################################################################

BTW np.clip function given a_min and a_max replaces any value less than a_min with a_min itself and any value greater than a_max with a_max itself
(BTW soccer_rl_ddpg_final.zip and soccer_rl_ddpg_less_camping_4Aug_2PM are the same in this branch. Same for PPO)

ALSO MAKE SURE THAT THE CLIPPING OF VELOCITIES DOESN'T CHANGE THE DIRECTION OF THE VELOCITY VECTOR
YOU'D PROBABLY WANT IT TO GO SLOWER THAN IN THE WRONG DIRECTION

Some observations for the final research report so far (Currently Saturday of Week 3):
- In Week 1 and 2, First had issues with the robot struggling to find the ball. It would move in a circular manner
or move along the edges of the field. This was probably because the rewards it was gaining were still 
"technically optimal" since the reward for scoring a goal was too high making it difficult to pursue.
- After giving rewards for getting closer to the ball and to start getting to closer to the goal with the ball
in possession, it started moving towards the ball but still struggled to gain possession definitively.
This continued despite adding penalties for moving away from the ball.
- After penalising moving away from both the goal and the ball and also penalising as time went on in the simulation
, it started to seek ball possession with avg rewards becoming greater than 100 during evaluation of the models
- To avoid the issue of moving along the edges or camping in corners or camp by either side of the goalpost, 
the reward function was modified to give penalties by checking its position after every action was taken in an episode.
On further changing the environment to incorporate curriculum learning where the training starts with an easy
environment, then progresses to medium and then finally to hard each varying the distance either robot had to get
to possess the ball, opponent robot's speed and behaviour (such as to position itself between the robot and the goal
when the other primary robot had ball possession) and the randomised posistions of the robots and the ball moving the
ball closer towards the opponent robot in higher difficulties, the DDPG model finally had the robot seeking ball
possession but this was not enough to score a goal or move towards the goal.
Further prioritising reaching the position in front of the goal which included aligning itself with the goal (not above or below)
and giving rewards based on how close they were to the center of the goal and whether they were aligned with the goal,
the DDPG model (trained on 250,000 episodes) showed results such as 4/5 trials of "medium" difficulty where the robot would get the ball and 
move itself into an optimal shooting zone defined as 
optimal_shooting_zone = (robot_x > 290 and robot_x < 340 and robot_y > 170 and robot_y < 230)  # Directly in front of goal
where the field was initially defined to have 400x400 (in pixel dimensions)
while the PPO model seemed to get the ball in 2/5 iterations while in the others, it struggled to go after the ball, going 
in seemingly random directions although moving extremely small distances initially.




WEEK 5 WHEN WORKIN GOUT THE CODE FOR ONNX CONVERSION, PPO would output raw neural network values whereas I have been working with scaled values
or normalised values in a certain range (FIND THE RANGE FROM THE CODE ALI!!!) 
ALso PPO would output different action outputs for the same inputs as it is non-deterministic or stochastic in nature while DDPG is 
deterministic
2 ideas: 1) use a seed for the random actions but this would give different actions for different seeds 
(ACCORDING TO AI, maybe don't say this justification for not using a seed)
2) Use a deterministic wrapper which does the same as PPO's internal actions except for the stochastic actions and this uses the trained model's policy
but with the same action outputs


TODO LIST:
- Fix the curriculum learning to progress between Easy, Medium and Hard difficulties
- FIx the metric collection and plots for the graphs
- Fix the reward function for training models
- Start work on testing plan and Progress Presentation and reseracgh
- Start work on the RL inference module for NUClear
- Week 12 PRESENTATION !!!!!!!!
- THESIS REPORT !!!!!!!!!!!!!!!!!!!



TODO: 11/09/2025::
1) Train without opponent
2) Tune reward function to work without opponent for a bit????
3) Focus on getting a good policy first more than deployment
4) Test that the actions or velocity is in the correct direction
5) Somehow smooth out the movement of the robots and maybe the ball
Right now it keeps teleporting bit by bit.

Also somehow reducing the friction by increasing the friction value to seems to get the robot zooming to score



# Changes:
# apply_robot_action
self.robot_angle += rotation * self.robot_rotation_speed #NOTE:1 Should this be -= rotation???
self.robot_angle = self.robot_angle % (2 * np.pi) # Normalise angle to [0, 2π)

strafe_x = -strafe_vel * np.sin(self.robot_angle)
strafe_y = strafe_vel * np.cos(self.robot_angle)

forward_x = forward_vel * np.cos(self.robot_angle) #NOTE:1 Changed from sin(angle + π/2)
forward_y = forward_vel * np.sin(self.robot_angle) #NOTE:1 New version removed minus sign and changed from cos(angle + π/2)

In render():
direction_end_y = robot_y + direction_length * np.sin(self.robot_angle) #NOTE:1 Claude AI flips to + in the 'correct' version??



