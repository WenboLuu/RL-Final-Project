import gymnasium as gym
import ale_py
import torch
from agent import DDQNAgent
from utils import to_tensor_preprocess_frames, evaluate_agent
from wandb_logger import initialize_wandb, log_metrics, finalize_wandb  # Import wandb_logger
import wandb

# Register ALE (Atari Learning Environment) environments
gym.register_envs(ale_py)


def main():
    # Initialize Weights & Biases using wandb_logger
    config = wandb.config

    ENV_NAME = config.env_name
    MAX_EPISODE_STEPS = 1000
    NUM_EPISODES = config.num_episodes
    NUM_ENVS = config.num_envs
    BATCH_SIZE = config.batch_size
    LEARNING_RATE = config.learning_rate
    GAMMA = config.gamma
    TARGET_UPDATE_FREQ = config.target_update_freq
    MEMORY_SIZE = config.memory_size
    MIN_REPLAY_SIZE = config.memory_size
    EPS_START = config.epsilon_start
    EPS_END = config.epsilon_end
    EPS_DECAY = config.epsilon_decay
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FRAME_SKIP = config.frame_skip
    EVAL_INTERVAL = config.eval_interval
    NUM_EVAL_EPISODES = config.num_eval_episodes
    EPS_EVAL = config.eps_eval

    initialize_wandb(project_name="RL-Final-Project", config=config)

    # Create vectorized environments
    envs = gym.vector.AsyncVectorEnv(
        [
            lambda: gym.make(ENV_NAME, max_episode_steps=MAX_EPISODE_STEPS, frameskip=FRAME_SKIP)
            for _ in range(NUM_ENVS)
        ]
    )
    obs_space = envs.single_observation_space
    action_space = envs.single_action_space.n

    agent = DDQNAgent(
        state_shape=(1, 84, 84),
        n_actions=action_space,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        target_update_freq=TARGET_UPDATE_FREQ,
        memory_size=MEMORY_SIZE,
        device=DEVICE,
    )

    global_step = 0
    epsilon = EPS_START

    # Create evaluation environment
    eval_env = gym.make(ENV_NAME)

    # Initialize replay memory
    init_states = envs.reset(seed=42)[0]
    init_frames = to_tensor_preprocess_frames(init_states, device=DEVICE)
    state_stack = torch.stack(init_frames).unsqueeze(1)

    episode_count = 0  # Initialize episode counter
    episode_reward = torch.zeros(NUM_ENVS, device=DEVICE)

    while episode_count < NUM_EPISODES:
        # Epsilon-greedy action selection
        actions = agent.select_action(state_stack, epsilon)
        next_states, rewards, terminated, truncated, infos = envs.step(actions)
        dones = [t or tr for t, tr in zip(terminated, truncated)]

        # Preprocess frames
        next_frames = to_tensor_preprocess_frames(next_states, device=DEVICE)
        next_state_stack = torch.stack(next_frames).unsqueeze(1)

        # Store transitions in replay buffer without for loop
        agent.replay_buffer.push(state_stack, actions, rewards, next_state_stack, dones)

        state_stack = next_state_stack
        episode_reward += torch.tensor(rewards, device=DEVICE)
        global_step += NUM_ENVS

        # Update epsilon
        epsilon = max(EPS_END, EPS_START - global_step * EPS_DECAY)

        # Train the agent
        if len(agent.replay_buffer) > MIN_REPLAY_SIZE:
            loss = agent.update()
            # Log training loss
            log_metrics({"Loss/Train": loss}, step=global_step)

        # Update target network
        if global_step % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        # Evaluate the agent at intervals
        if global_step % EVAL_INTERVAL == 0:
            avg_reward = evaluate_agent(
                eval_env,
                agent,
                num_episodes=NUM_EVAL_EPISODES,
                epsilon=EPS_EVAL,
                device=DEVICE,
            )
            log_metrics({"Reward/Eval": avg_reward}, step=global_step)
            print(f"Evaluation at step {global_step}: Average Reward: {avg_reward}")

        for idx in range(NUM_ENVS):
            if dones[idx]:
                # Log episode reward for completed episodes
                log_metrics({"Reward/Train": episode_reward[idx].item()}, step=global_step)
                if episode_count % 100 == 0:
                    print(f"Episode {episode_count} Reward: {episode_reward[idx].item()}")
                # Increment episode count and reset the reward for this environment
                episode_count += 1
                episode_reward[idx] = 0

    envs.close()
    eval_env.close()
    finalize_wandb()  # Finalize Weights & Biases run


if __name__ == "__main__":
    main()
