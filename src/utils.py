import cv2
import matplotlib.pyplot as plt
import torch
from IPython import display
from constants import DTYPE_STATE


def stack_preprocess_frames(frames, device="cpu", mode="grayscale"):
    frames = [torch.tensor(preprocess_frame(frame, mode=mode), device=device, dtype=DTYPE_STATE) for frame in frames]
    stacked_frames_tensor = torch.stack(frames)
    if mode == "grayscale":
        stacked_frames_tensor = stacked_frames_tensor.unsqueeze(1)  # Add channel dimension for grayscale
    elif mode == "rgb":
        stacked_frames_tensor = stacked_frames_tensor.permute(0, 3, 1, 2).contiguous()  # Rearrange dimensions for RGB
    else:
        raise ValueError("Invalid mode: choose 'grayscale' or 'rgb'")
    return stacked_frames_tensor


def preprocess_frame(frame, mode="grayscale"):
    """
    Preprocesses a given frame by converting it to the specified mode and resizing it to 84x84.

    Parameters:
    - frame: The input frame in RGB format.
    - mode: 'grayscale' or 'rgb'.

    Returns:
    - The preprocessed frame.
    """
    if mode == "grayscale":
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))
    elif mode == "rgb":
        frame = cv2.resize(frame, (84, 84))
    else:
        raise ValueError("Invalid mode: choose 'grayscale' or 'rgb'")
    return frame


def play_and_render_episode(env, agent):
    """
    Plays an episode using the provided agent in the given environment,
    renders each frame, and displays the animation with the accumulated reward.

    Parameters:
    - env: The environment instance (should be created with a render mode that returns frames).
    - agent: The agent instance with a select_action method.

    Usage:
    >>> import gymnasium as gym
    >>> from visualize_agent import play_and_render_episode
    >>> env = gym.make('LunarLander-v3', render_mode='rgb_array')
    >>> agent = AnAgentClass()
    >>> play_and_render_episode(env, agent)
    """

    # Initial setup
    state, _ = env.reset()
    terminated = truncated = False
    total_reward = 0

    # Precompute all frames and rewards
    frames = []
    rewards = []

    while not terminated and not truncated:
        # Render the environment and store the frame
        frame = env.render()
        frames.append(frame)

        rewards.append(total_reward)

        # Agent selects an action
        action = agent.select_action(state, eval=True)
        observation, reward, terminated, truncated, info = env.step(action)
        state = observation
        total_reward += reward  # Accumulate the total reward

    fig, ax = plt.subplots()
    img = ax.imshow(frames[0])
    ax.axis("off")

    text = ax.text(
        5,
        5,  # Coordinates (adjust as needed)
        f"Total Reward: {rewards[0]:.2f}",
        color="white",
        fontsize=12,
        backgroundcolor="black",
        ha="left",
        va="top",
    )

    # Display precomputed frames
    for i, frame in enumerate(frames):
        img.set_data(frame)  # Update the image data
        text.set_text(f"Total Reward: {rewards[i]:.2f}")  # Update the text

        display.display(fig)
        display.clear_output(wait=True)

    print(f"Total Reward: {total_reward}")
    env.close()


def evaluate_agent(env, agent, num_episodes=5, epsilon=0.05, input_mode="grayscale", device="cpu"):
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
            next_state_stack = stack_preprocess_frames([state], device=device, mode=input_mode)[0]
            state_stack = next_state_stack.unsqueeze(0)

        total_rewards.append(total_reward)

    average_reward = sum(total_rewards) / num_episodes
    return average_reward
