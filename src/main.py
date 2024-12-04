import os
import time
import torch

import random
import numpy as np

import gymnasium as gym
import ale_py
import wandb

from agent import DQNAgent
from utils import stack_preprocess_frames, evaluate_agent

from wandb_logger import initialize_wandb, log_metrics, finalize_wandb  # Import wandb_logger

import hashlib  # Add this import if not already present

# Register ALE (Atari Learning Environment) environments
gym.register_envs(ale_py)

def generate_group_name(config):
    # Exclude parameters that vary between runs, like 'seed'
    hyperparams = [(key, config[key]) for key in sorted(config.keys()) if key != 'seed']
    hyperparams_str = str(hyperparams)
    # Generate a hash to create a unique group name
    group_name = hashlib.md5(hyperparams_str.encode('utf-8')).hexdigest()
    return group_name

def main(seed):
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # If using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Initialize Weights & Biases using wandb_logger
    config = wandb.config
    initialize_wandb(project_name="RL-Final-Project", config=config)

    # Generate group_name and add it as a tag
    group_name = generate_group_name(config)
    wandb.run.tags = wandb.run.tags + (group_name,)

    # Optionally, set a run name that includes the seed
    wandb.run.name = f"run_{wandb.run.id}_seed_{seed}"

    # Rest of your code
    # Extract configuration parameters
    ENV_NAME = config.env_name
    MAX_TRAIN_EPISODE_STEPS = config.max_episode_steps
    NUM_TRAINING_STEPS = config.num_training_steps
    NUM_ENVS = config.num_envs
    BATCH_SIZE = config.batch_size
    LEARNING_RATE = config.learning_rate
    GAMMA = config.gamma
    TARGET_UPDATE_FREQ = config.target_update_freq
    MEMORY_SIZE = config.memory_size
    MIN_REPLAY_SIZE = int(0.1 * MEMORY_SIZE)
    EPS_START = config.epsilon_start
    EPS_END = config.epsilon_end
    EPS_DECAY = config.epsilon_decay
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    FRAME_SKIP = config.frame_skip
    EVAL_INTERVAL = config.eval_interval
    NUM_EVAL_EPISODES = config.num_eval_episodes
    EPS_EVAL = config.eps_eval
    INPUT_MODE = config.input_mode  # Add input_mode to configurations
    EVAL_MAX_EPISODE_STEPS = config.eval_max_episode_steps

    checkpoint_path = f"../checkpoints/{ENV_NAME.replace('/', '-')}"
    os.makedirs(checkpoint_path, exist_ok=True)

    # Create vectorized environments
    envs = gym.vector.AsyncVectorEnv(
        [
            lambda: gym.make(ENV_NAME, max_episode_steps=MAX_TRAIN_EPISODE_STEPS, frameskip=FRAME_SKIP)
            for _ in range(NUM_ENVS)
        ]
    )
    obs_space = envs.single_observation_space
    action_space = envs.single_action_space.n

    # Adjust state shape based on input mode
    if INPUT_MODE == "grayscale":
        state_channels = 1
    elif INPUT_MODE == "rgb":
        state_channels = 3
    else:
        raise ValueError("Invalid input_mode: choose 'grayscale' or 'rgb'")

    agent = DQNAgent(
        state_shape=(state_channels, 84, 84),
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
    if EVAL_MAX_EPISODE_STEPS is None:
        eval_env = gym.make(ENV_NAME, frameskip=FRAME_SKIP)
    else:
        eval_env = gym.make(ENV_NAME, max_episode_steps=EVAL_MAX_EPISODE_STEPS, frameskip=FRAME_SKIP)

    # Initialize replay memory
    init_states = envs.reset(seed=seed)[0]
    state_stack = stack_preprocess_frames(init_states, device=DEVICE, mode=INPUT_MODE)

    episode_count = 0  # Initialize episode counter
    episode_reward = torch.zeros(NUM_ENVS, device=DEVICE)

    while global_step < NUM_TRAINING_STEPS:
        # Epsilon-greedy action selection
        actions = agent.select_action(state_stack, epsilon)
        next_states, rewards, terminated, truncated, infos = envs.step(actions)
        dones = [t or tr for t, tr in zip(terminated, truncated)]

        # Preprocess frames
        next_state_stack = stack_preprocess_frames(next_states, device=DEVICE, mode=INPUT_MODE)

        # Store transitions in replay buffer
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
            # print(f"Target network updated at step {global_step}")

        # Evaluate the agent at intervals
        if global_step % EVAL_INTERVAL == 0:
            avg_reward = evaluate_agent(
                eval_env,
                agent,
                num_episodes=NUM_EVAL_EPISODES,
                epsilon=EPS_EVAL,
                input_mode=INPUT_MODE,
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
