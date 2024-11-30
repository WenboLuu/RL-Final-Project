import os
import time
import torch
import torch.profiler
import yaml

import gymnasium as gym
import ale_py

from agent import DDQNAgent
from utils import stack_preprocess_frames, evaluate_agent

# Register ALE (Atari Learning Environment) environments
gym.register_envs(ale_py)


def main():
    # Load the configuration from the YAML file
    with open("../run_config.yaml") as file:
        params = yaml.safe_load(file)

    ENV_NAME = params["env_name"]
    MAX_EPISODE_STEPS = params["max_episode_steps"]
    NUM_TRAINING_STEPS = params["num_training_steps"]
    NUM_ENVS = params["num_envs"]
    BATCH_SIZE = params["batch_size"]
    LEARNING_RATE = params["learning_rate"]
    GAMMA = params["gamma"]
    TARGET_UPDATE_FREQ = params["target_update_freq"]
    MEMORY_SIZE = params["memory_size"]
    MIN_REPLAY_SIZE = int(0.1 * MEMORY_SIZE)
    EPS_START = params["epsilon_start"]
    EPS_END = params["epsilon_end"]
    EPS_DECAY = params["epsilon_decay"]
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    FRAME_SKIP = params["frame_skip"]
    EVAL_INTERVAL = params["eval_interval"]
    NUM_EVAL_EPISODES = params["num_eval_episodes"]
    EPS_EVAL = params["eps_eval"]
    INPUT_MODE = params["input_mode"]  # Add input_mode to configurations

    checkpoint_path = f"../checkpoints/{ENV_NAME.replace('/', '-')}"
    os.makedirs(checkpoint_path, exist_ok=True)

    # Create vectorized environments
    envs = gym.vector.AsyncVectorEnv(
        [
            lambda: gym.make(ENV_NAME, max_episode_steps=MAX_EPISODE_STEPS, frameskip=FRAME_SKIP)
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

    agent = DDQNAgent(
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
    eval_env = gym.make(ENV_NAME)

    # Initialize replay memory
    init_states = envs.reset(seed=42)[0]
    state_stack = stack_preprocess_frames(init_states, device=DEVICE, mode=INPUT_MODE)

    episode_count = 0  # Initialize episode counter
    episode_reward = torch.zeros(NUM_ENVS, device=DEVICE)

    WAIT = 1
    WARMUP = 500
    ACTIVE = 10
    REPEAT = 3
    
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=WAIT, warmup=WARMUP, active=ACTIVE, repeat=REPEAT),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("../log/profiler"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        while global_step < NUM_TRAINING_STEPS:
            with torch.profiler.record_function("Epsilon-greedy action selection"):
                actions = agent.select_action(state_stack, epsilon)
            with torch.profiler.record_function("Environment step"):
                next_states, rewards, terminated, truncated, infos = envs.step(actions)
            dones = [t or tr for t, tr in zip(terminated, truncated)]

            with torch.profiler.record_function("Preprocess frames"):
                next_state_stack = stack_preprocess_frames(next_states, device=DEVICE, mode=INPUT_MODE)

            with torch.profiler.record_function("Store transitions in replay buffer"):
                agent.replay_buffer.push(state_stack, actions, rewards, next_state_stack, dones)

            state_stack = next_state_stack
            episode_reward += torch.tensor(rewards, device=DEVICE)
            global_step += NUM_ENVS

            # Update epsilon
            epsilon = max(EPS_END, EPS_START - global_step * EPS_DECAY)

            # Train the agent
            if len(agent.replay_buffer) > MIN_REPLAY_SIZE:
                with torch.profiler.record_function("Agent update"):
                    loss = agent.update()

            # Update target network
            if global_step % TARGET_UPDATE_FREQ == 0:
                with torch.profiler.record_function("Update target network"):
                    agent.update_target_network()

            # Evaluate the agent at intervals
            if global_step % EVAL_INTERVAL == 0:
                with torch.profiler.record_function("Evaluate agent"):
                    avg_reward = evaluate_agent(
                        eval_env,
                        agent,
                        num_episodes=NUM_EVAL_EPISODES,
                        epsilon=EPS_EVAL,
                        input_mode=INPUT_MODE,
                        device=DEVICE,
                    )
                    print(f"Evaluation at step {global_step}: Average Reward: {avg_reward}")

            for idx in range(NUM_ENVS):
                if dones[idx]:
                    if episode_count % 100 == 0:
                        print(f"Episode {episode_count} Reward: {episode_reward[idx].item()}")
                    episode_count += 1
                    episode_reward[idx] = 0

            prof.step()  # Step the profiler at the end of each iteration
            
            if prof.step_num >= (WAIT + WARMUP + ACTIVE) * REPEAT:
                break

    envs.close()
    eval_env.close()


if __name__ == "__main__":
    main()
