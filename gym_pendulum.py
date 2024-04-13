import gymnasium as gym
import pyautogui
env = gym.make("Pendulum-v1", render_mode="human")
observation, info = env.reset()

def check_mouse_position():
    # Get the screen size
    screen_width, _ = pyautogui.size()
    
    # Get the current mouse position
    mouse_x, _ = pyautogui.position()

    # Convert the mouse position to a value between -2 and 2
    return (mouse_x / (screen_width - 1)) * 4 - 2



for _ in range(100000):
    # action = env.action_space.sample()  # agent policy that uses the observation and info
    action = [check_mouse_position()]
    observation, reward, terminated, truncated, info = env.step(action)

    # if terminated or truncated:
    #     observation, info = env.reset()

env.close()