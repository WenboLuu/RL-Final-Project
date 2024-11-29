import cv2
import matplotlib.pyplot as plt
import torch
from IPython import display
from constants import DTYPE_STATE


def to_tensor_preprocess_frames(frames, device="cpu", mode="grayscale"):
    """
    Preprocesses a list of frames by converting each frame to the specified mode,
    resizing it to 84x84, and converting it to a PyTorch tensor.

    Parameters:
    - frames: A list of input frames in RGB format.
    - mode: 'grayscale' or 'rgb' for preprocessing.

    Returns:
    - A list of preprocessed frames as PyTorch tensors.
    """
    preprocessed_tensor_frames = [
        torch.tensor(preprocess_frame(frame, mode=mode), device=device, dtype=DTYPE_STATE) for frame in frames
    ]
    return preprocessed_tensor_frames


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
        # Ensure the frame has the correct channel order (if needed)
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


def evaluate_agent(env, agent, num_episodes=5, epsilon=0.05, device="cpu"):
    """
    Evaluates the agent over a specified number of episodes.
    """
    total_rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        terminated = truncated = False
        total_reward = 0

        # Preprocess initial state
        state_frame = to_tensor_preprocess_frames([state], device=device)[0]  # Shape: [84, 84]
        state_stack = state_frame.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 84, 84]

        while not terminated and not truncated:
            # Agent selects an action
            action = agent.select_action(state_stack, epsilon)
            action = action.item()  # Convert action array to scalar

            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            # Preprocess next state
            next_state_frame = to_tensor_preprocess_frames([next_state], device=device)[0]
            state_stack = next_state_frame.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 84, 84]

        total_rewards.append(total_reward)

    average_reward = sum(total_rewards) / num_episodes
    return average_reward
