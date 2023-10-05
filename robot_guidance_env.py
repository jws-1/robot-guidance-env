import gym
from gym.spaces import Box
import numpy as np

class RobotGuidanceEnv(gym.Env):

    def __init__(self, width, height, max_robot_vel, max_person_vel, preferred_following_distance, max_following_distance, start_state = None, goal_state = None, seed=1032):
        np.random.seed(seed)
        
        self.state_space = Box(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), np.array([width, height, width, height], dtype=np.float32))
        self.action_space = Box(-max_robot_vel, +max_robot_vel)
        
        if start_state is None:
            start_state = self.state_space.sample()
        if goal_state is None:
            goal_state = self.state_space.sample()
        self.start_state = start_state
        self.state = self.start_state
        self.goal_state = goal_state


        self.max_robot_vel = max_robot_vel
        self.max_person_vel = max_person_vel
        self.preferred_following_distance = preferred_following_distance
        self.max_following_distance = max_following_distance

    def step(self, robot_vels):
        state = self.state
        state += np.array([0.0, 0.0, robot_vels[0], robot_vels[1]])
        if not self.state_space.contains(state):
            raise Exception("Velocity applied takes robot out of state space")

        diff = state[2:] - state[:2]

        person_vel = np.array([0.0, 0.0])
        
        if np.any(np.abs(diff) < self.max_following_distance):



            req_vel = self.preferred_following_distance - diff
            
            if req_vel[0] > self.max_person_vel:
                person_vel[0] = self.max_person_vel
            else:
                person_vel[0] = req_vel[0]

            if req_vel[1] > self.max_person_vel:
                person_vel[1] = self.max_person_vel
            else:
                person_vel[1] = req_vel[1]

            state += np.array([person_vel[0], person_vel[1], 0.0, 0.0])

        self.state = state

        return self.state, -np.linalg.norm(self.state, self.goal_state), self.state == self.goal_state

