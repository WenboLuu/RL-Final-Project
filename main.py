import gymnasium as gym
import ale_py
import torch
from torch.utils.tensorboard import SummaryWriter
from agent import DDQNAgent
from utils import to_tensor_preprocess_frames
from tqdm import tqdm
# import multiprocessing

# Register ALE (Atari Learning Environment) environments
gym.register_envs(ale_py)

# Hyperparameters
# ENV_NAME = "ALE/BattleZone-v5"
ENV_NAME = "ALE/DemonAttack-v5"
MAX_EPISODE_STEPS = 1_000
NUM_ENVS = 16
NUM_EPISODES = 1_000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
GAMMA = 0.99
TARGET_UPDATE_FREQ = 5_000
MEMORY_SIZE = 200_000
MIN_REPLAY_SIZE = 1000
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = (EPS_START - EPS_END) / (NUM_EPISODES * MAX_EPISODE_STEPS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAME_SKIP = 4


def main():
    # Set multiprocessing start method to 'spawn'
    # multiprocessing.set_start_method('spawn')

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

    writer = SummaryWriter()
    global_step = 0
    epsilon = EPS_START

    # Initialize replay memory
    init_states = envs.reset(seed=42)[0]
    init_frames = to_tensor_preprocess_frames(init_states, device=DEVICE)
    state_stack = torch.stack(init_frames).unsqueeze(1)

    episode_count = 0  # Initialize episode counter
    episode_reward = torch.zeros(NUM_ENVS, device=DEVICE)

    import time
    start_time = time.time()

    while episode_count < NUM_EPISODES:
        # Epsilon-greedy action selection
        actions = agent.select_action(state_stack, epsilon)
        next_states, rewards, terminated, truncated, infos = envs.step(actions)
        dones = [t or tr for t, tr in zip(terminated, truncated)]

        # Preprocess frames
        next_frames = to_tensor_preprocess_frames(next_states, device=DEVICE)
        next_state_stack = torch.stack(next_frames).unsqueeze(1)

        # Store transitions in replay buffer without for loop
        agent.replay_buffer.push(
            state_stack,
            actions,
            rewards,
            next_state_stack,
            dones
        )

        state_stack = next_state_stack
        episode_reward += torch.tensor(rewards, device=DEVICE)
        global_step += 1

        # Update epsilon
        epsilon = max(EPS_END, EPS_START - global_step * EPS_DECAY)

        # Train the agent
        if len(agent.replay_buffer) > MIN_REPLAY_SIZE:
            agent.update()

        # Update target network
        if global_step % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
        
        for idx in range(NUM_ENVS):
            if dones[idx]:
                # Log episode reward for completed episodes
                writer.add_scalar("Reward/Episode", episode_reward[idx].item(), episode_count)
                print(f"Episode {episode_count} Reward: {episode_reward[idx].item()}")
                
                # Increment episode count and reset the reward for this environment
                episode_count += 1
                episode_reward[idx] = 0
        if global_step == 10_000:
            print(f"Time to reach 10,000 steps: {time.time() - start_time}")

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
