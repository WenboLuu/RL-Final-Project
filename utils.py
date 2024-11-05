# ╦ ╦┌─┐┌┐┌┌┐ ┌─┐  ╦  ┬ ┬
# ║║║├┤ │││├┴┐│ │  ║  │ │
# ╚╩╝└─┘┘└┘└─┘└─┘  ╩═╝└─┘

import matplotlib.pyplot as plt
from IPython import display


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
