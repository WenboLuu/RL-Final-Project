import argparse
import os
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import ale_py

from agent import DDQNAgent  
from utils import stack_preprocess_frames  

def evaluate_agent(env, agent, num_episodes=5, epsilon=0.00, input_mode="grayscale", device="cpu"):
    """
    Evaluates the agent over a specified number of episodes.
    
    Args:
        env (gym.Env): The environment to evaluate the agent in.
        agent (DDQNAgent): The trained agent to evaluate.
        num_episodes (int): Number of episodes for evaluation.
        epsilon (float): Exploration rate (0 for fully greedy evaluation).
        input_mode (str): Preprocessing mode for input frames ("grayscale" or "rgb").
        device (str): Device to use for computations ("cpu" or "cuda").

    Returns:
        float: Average reward over the evaluation episodes.
    """
    total_rewards = []

    for _ in range(num_episodes):
        # Reset environment and initialize variables
        state, _ = env.reset()
        terminated = truncated = False
        total_reward = 0

        # Preprocess the initial state
        state_stack = stack_preprocess_frames([state], device=device, mode=input_mode)[0]
        state_stack = state_stack.unsqueeze(0)  # Add batch dimension

        while not terminated and not truncated:
            # Select action using the agent's policy
            action = agent.select_action(state_stack, epsilon)
            action = action.item()  # Convert action to scalar

            # Step in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            # Preprocess the next state
            next_state_stack = stack_preprocess_frames([next_state], device=device, mode=input_mode)[0]
            state_stack = next_state_stack.unsqueeze(0)

        total_rewards.append(total_reward)

    # Calculate and return average reward
    average_reward = sum(total_rewards) / num_episodes
    return average_reward


def main():
    """
    Main function to evaluate a trained DDQN agent on a specified environment.
    Parses command-line arguments, loads the agent, and evaluates it.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained DDQN agent on an Atari environment.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file of the model.")
    parser.add_argument("--architecture", type=str, required=True, choices=["DDQNAgent"], help="Agent architecture type. (Currently only DDQNAgent supported in this example)")
    parser.add_argument("--env_name", type=str, required=True, help="Name of the OpenAI Gym environment.")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to record.")
    parser.add_argument("--video_dir", type=str, default="evaluation_videos", help="Directory to save evaluation videos.")
    parser.add_argument("--input_mode", type=str, default="grayscale", choices=["grayscale", "rgb"], help="Input mode for frame preprocessing.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use for inference.")
    args = parser.parse_args()

    # Create the environment with rendering for video recording
    env = gym.make(args.env_name, render_mode='rgb_array')

    # Add video recording wrapper
    os.makedirs(args.video_dir, exist_ok=True)
    env = RecordVideo(env, video_folder=args.video_dir, episode_trigger=lambda episode_id: True)

    # Fetch state and action space details to initialize the agent
    dummy_obs = env.reset()[0]
    channels = 1 if args.input_mode == "grayscale" else 3  # Determine input channels
    state_shape = (channels, 84, 84)  # Assumed state shape based on preprocessing
    action_space = env.action_space.n

    # Default values for agent initialization in evaluation mode
    DUMMY_BATCH_SIZE = 32
    DUMMY_LR = 0.0001
    DUMMY_GAMMA = 0.99
    DUMMY_TARGET_UPDATE_FREQ = 1000
    DUMMY_MEMORY_SIZE = 10000

    # Initialize the agent
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

    # Load the model checkpoint into the agent
    agent.load_checkpoint(args.checkpoint_path)

    # Evaluate the agent in the environment
    avg_reward = evaluate_agent(env, agent, num_episodes=args.num_episodes, input_mode=args.input_mode, device=args.device)
    print(f"Average reward over {args.num_episodes} episodes: {avg_reward}")

    # Close the environment and display the video directory
    env.close()
    print(f"Videos saved to {args.video_dir}")


if __name__ == "__main__":
    main()
