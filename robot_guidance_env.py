import gym
from gym.spaces import Box
import numpy as np

class RobotGuidanceEnv(gym.Env):

    def __init__(self, width, height, max_robot_vel, max_person_vel, preferred_following_distance, max_following_distance, start_state = None, goal_state = None, seed=1032):
        np.random.seed(seed)

        self.width = width
        self.height = height
        self.max_robot_vel = max_robot_vel
        self.max_person_vel = max_person_vel
        self.preferred_following_distance = preferred_following_distance
        self.max_following_distance = max_following_distance
        self.start_state = start_state
        self.goal_state = goal_state

        self.person_x, self.person_y, self.robot_x, self.robot_y = self.start_state


        self.observation_space = Box(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), np.array([width, height, width, height], dtype=np.float32))
        self.action_space = Box(low=-max_robot_vel, high=max_robot_vel, shape=(2,), dtype=np.float32)

    def reset(self):
        self.person_x, self.person_y, self.robot_x, self.robot_y = self.start_state
        observation = np.array([self.person_x, self.person_y, self.robot_x, self.robot_y])
        return np.array(observation, dtype=np.float32), {}


    def step(self, action):
        self.robot_x += action[0]
        self.robot_y += action[1]
        delta_x = self.robot_x - self.person_x
        delta_y = self.robot_y - self.person_y
        distance_to_robot = np.sqrt((self.person_x - self.robot_x) ** 2 + (self.person_y - self.robot_y) ** 2)
        print(f"The robot is {distance_to_robot}m from the person")
        print(f"Action: {action}")

        if distance_to_robot <= self.max_following_distance:
            print(f"Within max_following_distance, so person will try and follow. ({distance_to_robot} <= {self.max_following_distance})")
            
            desired_distance = self.preferred_following_distance

            if abs(distance_to_robot) >= self.preferred_following_distance:
                print(f"Far enough away from the preferred_following_distance ({distance_to_robot} >= {self.preferred_following_distance})")
                delta_x *= (desired_distance / distance_to_robot)
                delta_y *= (desired_distance / distance_to_robot)
                print(f"Desired move: {delta_x} {delta_y}")
                delta_x = np.clip(delta_x, -self.max_person_vel, +self.max_person_vel)
                delta_y = np.clip(delta_y, -self.max_person_vel, +self.max_person_vel)

                self.person_x += delta_x
                self.person_y += delta_y
                print(f"Person moves by {delta_x} {delta_y}")
            else:
                print(f"The person is already close to the robot, so not moving.")

        # Clip positions within the environment boundaries
        self.person_x = np.clip(self.person_x, 0, self.width)
        self.person_y = np.clip(self.person_y, 0, self.height)
        self.robot_x = np.clip(self.robot_x, 0, self.width)
        self.robot_y = np.clip(self.robot_y, 0, self.height)

        reward = -abs(np.sqrt((self.person_x - self.goal_state[0]) ** 2 + (self.person_y - self.goal_state[1]) ** 2))

        observation = np.array([self.person_x, self.person_y, self.robot_x, self.robot_y], dtype=np.float32)
        print(observation)
        return observation, reward, self.person_x == self.goal_state[0] and self.person_y == self.goal_state[1], False, {}


""" 
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
"""
