import gym
from gym.spaces import Box
import numpy as np
import pygame
import math


class RobotGuidanceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4, "scaling_factor" : 100.0, "draw_grid_lines" : False}

    def __init__(self, width, height, max_robot_vel, max_person_vel, preferred_following_distance, max_following_distance, start_state = None, goal_state = None, seed=1032, render_mode=None):
        np.random.seed(seed)
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.width = width
        self.height = height
        self.max_robot_vel = max_robot_vel
        self.max_person_vel = max_person_vel
        self.preferred_following_distance = preferred_following_distance
        self.max_following_distance = max_following_distance
        self.start_state = start_state
        self.goal_state = goal_state

        self.observation_space = Box(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), np.array([width, height, width, height], dtype=np.float32))
        self.action_space = Box(low=-max_robot_vel, high=max_robot_vel, shape=(2,), dtype=np.float32)

        if self.start_state is None:
            self.start_state = self.observation_space.sample()
        if self.goal_state is None:
            self.goal_state = self.observation_space.sample()

        self.person_x, self.person_y, self.robot_x, self.robot_y = self.start_state

        if render_mode == "human":

            # Calculate the window size based on the scaling factor and the aspect ratio
            scaling_factor = self.metadata["scaling_factor"]
            window_width = int(width * scaling_factor)
            window_height = int(height * scaling_factor)

            pygame.init()
            self.window = pygame.display.set_mode((window_width, window_height))
            pygame.display.set_caption("Robot Guidance Environment")
            self.clock = pygame.time.Clock()

    def reset(self):
        self.person_x, self.person_y, self.robot_x, self.robot_y = self.start_state
        observation = np.array([self.person_x, self.person_y, self.robot_x, self.robot_y])
        self.render()
        return np.array(observation, dtype=np.float32), {}
    
    def step(self, action):
        # Apply the robot's velocities based on the action
        self.robot_x += action[0]
        self.robot_y += action[1]
        print(f"The robot moves by {action[0]}m {action[1]}m")
        # Calculate the distance between the human and the robot
        distance_to_robot = np.sqrt((self.person_x - self.robot_x) ** 2 + (self.person_y - self.robot_y) ** 2)
        print(f"The robot is {distance_to_robot}m from the person")

        if distance_to_robot <= self.preferred_following_distance:
            print(f"Within the preferred_following_distance, so the person does not move.")
        elif distance_to_robot <= self.max_following_distance:
            print(f"Inside of max_following_distance, so person will try and follow.")

            # Calculate the desired movement to reach the preferred_following_distance
            # Use trig because it's useful
            x = math.atan2(self.robot_y - self.person_y, self.robot_x - self.person_x)
            z = math.pi - math.pi/2.0 - x
            delta_x = (distance_to_robot-self.preferred_following_distance / math.sin(math.pi/2.0)) * math.sin(z)
            delta_y = (distance_to_robot-self.preferred_following_distance / math.sin(math.pi/2.0)) * math.sin(x)

            # Clip the movement based on max_person_vel
            delta_x = np.clip(delta_x, -self.max_person_vel, self.max_person_vel)
            delta_y = np.clip(delta_y, -self.max_person_vel, self.max_person_vel)

            # Update the human's position
            self.person_x += delta_x
            self.person_y += delta_y
            print(f"Person moves by {delta_x}m {delta_y}m, new distance is {np.sqrt((self.person_x - self.robot_x) ** 2 + (self.person_y - self.robot_y) ** 2)}")

        # Clip positions within the environment boundaries
        self.person_x = np.clip(self.person_x, 0, self.width)
        self.person_y = np.clip(self.person_y, 0, self.height)
        self.robot_x = np.clip(self.robot_x, 0, self.width)
        self.robot_y = np.clip(self.robot_y, 0, self.height)

        # Calculate the reward based on the distance to the goal state
        reward = -abs(np.sqrt((self.person_x - self.goal_state[0]) ** 2 + (self.person_y - self.goal_state[1]) ** 2))

        observation = np.array([self.person_x, self.person_y, self.robot_x, self.robot_y], dtype=np.float32)
        print(observation)

        self.render()

        # Check if the person has reached the goal state
        done = (self.person_x == self.goal_state[0] and self.person_y == self.goal_state[1])

        return observation, reward, done, False, {}


        
    def render(self):

        if self.render_mode == "human":
            self.window.fill((255, 255, 255))  # Clear the window
            
            if self.metadata["draw_grid_lines"]:
                # Draw the grid

                cell_width = int(self.window.get_width() / self.width)
                cell_height = int(self.window.get_height() / self.height)

                for x in range(0, self.window.get_width(), cell_width):
                    for y in range(0, self.window.get_height(), cell_height):
                        pygame.draw.rect(self.window, (0,0,0), (x, y, cell_width, cell_height), 1)

            # Draw the goal state as a green circle
            goal_x, goal_y = self.convert_coordinates(self.goal_state[:2])
            pygame.draw.circle(self.window, (0, 255, 0), (int(goal_x), int(goal_y)), 10)

            # Draw the human as a blue circle
            human_x, human_y = self.convert_coordinates((self.person_x, self.person_y))
            pygame.draw.circle(self.window, (0, 0, 255), (int(human_x), int(human_y)), 10)

            # Draw the robot as a red circle
            robot_x, robot_y = self.convert_coordinates((self.robot_x, self.robot_y))
            pygame.draw.circle(self.window, (255, 0, 0), (int(robot_x), int(robot_y)), 10)

            # Draw a line between the human and the robot
            pygame.draw.line(self.window, (255, 0, 0), (int(human_x), int(human_y)), (int(robot_x), int(robot_y)))

            # Calculate the midpoint of the line
            midpoint_x = (human_x + robot_x) / 2
            midpoint_y = (human_y + robot_y) / 2

            # Calculate the distance between the human and the robot
            distance = np.sqrt((self.person_x - self.robot_x) ** 2 + (self.person_y - self.robot_y) ** 2)
            distance_text = f"Distance: {distance:.2f} m"

            # Render the distance text on the screen with a smaller font
            font = pygame.font.Font(None, 24)  # Use a smaller font size
            text_surface = font.render(distance_text, True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=(midpoint_x, midpoint_y))
            self.window.blit(text_surface, text_rect)

            pygame.display.flip()
            self.clock.tick(4)  # Control rendering FPS


    def convert_coordinates(self, coordinates):
        # Convert observation space coordinates to Pygame window space
        x, y = coordinates
        pygame_x = x * (self.window.get_width() / self.width)
        pygame_y = y * (self.window.get_height() / self.height)
        return pygame_x, pygame_y