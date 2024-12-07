import argparse
import os
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np

import ale_py

from agent import DDQNAgent  
from utils import stack_preprocess_frames  

# def stack_preprocess_frames(frames, device="cpu", mode="grayscale"):
#     """
#     Example frame preprocessing. Adjust as per your needs.
#     For Atari, often frames are resized to (84, 84) and converted to grayscale.
#     This is a placeholder; you should use your actual preprocessing function.
#     """
#     import torchvision.transforms as T
#     from PIL import Image

#     if mode == "grayscale":
#         transform = T.Compose([
#             T.ToPILImage(),
#             T.Resize((84, 84)),
#             T.Grayscale(),
#             T.ToTensor()
#         ])
#     else:
#         transform = T.Compose([
#             T.ToPILImage(),
#             T.Resize((84, 84)),
#             T.ToTensor()
#         ])

#     # If frames is a single frame, wrap it as a list for consistency
#     if not isinstance(frames, list):
#         frames = [frames]

#     processed_frames = [transform(frame) for frame in frames]
#     # Stack frames along the channel dimension
#     stacked = torch.cat(processed_frames, dim=0).unsqueeze(0).to(device)
#     return stacked

# def evaluate_agent(env, agent, num_episodes=1, input_mode="grayscale", device="cpu", max_episode_steps=3000):
#     total_rewards = []
#     step =  0
#     for _ in range(num_episodes):
#         state, _ = env.reset()
#         done = False
#         total_reward = 0

#         # Preprocess initial observation
#         state_stack = stack_preprocess_frames([state], device=device, mode=input_mode)

#         while not done and step < max_episode_steps:
#             step += 1
#             # Select action with epsilon=0.0 (pure greedy)
#             action = agent.select_action(state_stack, epsilon=0.0)[0]

#             next_state, reward, terminated, truncated, info = env.step(action)
#             done = terminated or truncated
#             total_reward += reward

#             # Preprocess next state
#             state_stack = stack_preprocess_frames([next_state], device=device, mode=input_mode)

#         total_rewards.append(total_reward)
#     return np.mean(total_rewards)


def evaluate_agent(env, agent, num_episodes=5, epsilon=0.00, input_mode="grayscale", device="cpu"):
    """
    Evaluates the agent over a specified number of episodes.
    """
    total_rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        terminated = truncated = False
        total_reward = 0

        # Preprocess initial state
        state_stack = stack_preprocess_frames([state], device=device, mode=input_mode)[0]  # Shape: [1, 84, 84]
        state_stack = state_stack.unsqueeze(0)  # Shape: [1, 1, 84, 84]

        while not terminated and not truncated:
            # Agent selects an action
            action = agent.select_action(state_stack, epsilon)
            action = action.item()  # Convert action array to scalar

            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            # Preprocess next state
            next_state_stack = stack_preprocess_frames([next_state], device=device, mode=input_mode)[0]
            state_stack = next_state_stack.unsqueeze(0)

        total_rewards.append(total_reward)

    average_reward = sum(total_rewards) / num_episodes
    return average_reward


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DDQN agent on an Atari environment.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file of the model.")
    parser.add_argument("--architecture", type=str, required=True, choices=["DDQNAgent"], help="Agent architecture type. (Currently only DDQNAgent supported in this example)")
    parser.add_argument("--env_name", type=str, required=True, help="Name of the OpenAI Gym environment.")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to record.")
    parser.add_argument("--video_dir", type=str, default="evaluation_videos", help="Directory to save evaluation videos.")
    parser.add_argument("--input_mode", type=str, default="grayscale", choices=["grayscale", "rgb"], help="Input mode for frame preprocessing.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use for inference.")
    args = parser.parse_args()

    # Create environment
    # For Atari, typically you'd use something like:
    # env = gym.make(args.env_name, render_mode='rgb_array')
    # If using Atari, ensure you have 'frameskip' and 'max_episode_steps' if needed, or any wrappers.
    env = gym.make(args.env_name, render_mode='rgb_array')

    # RecordVideo wrapper will automatically record episodes
    os.makedirs(args.video_dir, exist_ok=True)
    env = RecordVideo(env, video_folder=args.video_dir, episode_trigger=lambda episode_id: True)

    # Fetch state and action space details to initialize the agent
    # Adjust as needed based on your environment and agent
    dummy_obs = env.reset()[0]
    if args.input_mode == "grayscale":
        channels = 1
    else:
        channels = 3
    state_shape = (channels, 84, 84)  # Assuming this from your training setup
    action_space = env.action_space.n
    
    
    # Example default values for arguments that are not relevant in evaluation mode
    DUMMY_BATCH_SIZE = 32
    DUMMY_LR = 0.0001
    DUMMY_GAMMA = 0.99
    DUMMY_TARGET_UPDATE_FREQ = 1000
    DUMMY_MEMORY_SIZE = 10000

    # Initialize the agent with the required arguments
    agent = DDQNAgent(
        state_shape=state_shape,
        n_actions=action_space,
        batch_size=DUMMY_BATCH_SIZE,
        lr=DUMMY_LR,
        gamma=DUMMY_GAMMA,
        target_update_freq=DUMMY_TARGET_UPDATE_FREQ,
        memory_size=DUMMY_MEMORY_SIZE,
        device=args.device,
    )

    # # Initialize agent
    # if args.architecture == "DDQNAgent":
    #     agent = DDQNAgent(state_shape=state_shape, n_actions=action_space, device=args.device)
    # else:
    #     raise ValueError("Unsupported architecture provided.")

    # Load agent checkpoint
    agent.load_checkpoint(args.checkpoint_path)

    # Evaluate agent
    avg_reward = evaluate_agent(env, agent, num_episodes=args.num_episodes, input_mode=args.input_mode, device=args.device)
    print(f"Average reward over {args.num_episodes} episodes: {avg_reward}")

    env.close()
    print(f"Videos saved to {args.video_dir}")

if __name__ == "__main__":
    main()
