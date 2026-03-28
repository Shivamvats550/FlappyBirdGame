import gymnasium
import flappy_bird_gymnasium
import time

# Create the environment
env = gymnasium.make("FlappyBird-v0", render_mode="human")

# Reset the environment
obs, _ = env.reset()

while True:
    # Select a random action (0: do nothing, 1: flap)
    action = env.action_space.sample()
    
    # Take the action
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Rendering is handled by render_mode="human"
    time.sleep(1/30) # Optional: control frame rate
    
    # Check if the bird died
    if terminated:
        obs, _ = env.reset()

env.close()
