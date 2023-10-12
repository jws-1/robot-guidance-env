import gym
from robot_guidance_env.robot_guidance_env import RobotGuidanceEnv
gym.register(id='Robot-Guidance-Env-v0', entry_point='robot_guidance_env:RobotGuidanceEnv', kwargs={'width' : 10.0, 'height': 10.0, 'max_robot_vel': 1.0, 'max_person_vel': 1.0, 'preferred_following_distance' : 2.0, 'max_following_distance' : 5.0, 'start_state' : None, 'goal_state': None, 'seed' : 1032})

