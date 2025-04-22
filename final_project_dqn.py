import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Neural Network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)
    
class DQNAgent:
    # learning rates attempted: 1e-3, 1e-4, 3e-4
    # target_update_freq  attempted: 5000, 1000
    # decay attempted: 0.995, 0.999
    # epsilon min attempted: 0.01, 0.05
    def __init__(self, env, gamma=0.99, lr=3e-4, batch_size=64, ep=1.0, ep_min=0.05, ep_decay=0.999, mem_size=100000, target_update_freq=1000):
        self.env = env
        self.gamma = gamma
        self.ep = ep
        self.ep_min = ep_min
        self.ep_decay = ep_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.n_actions = env.action_space.n
        self.memory = deque(maxlen=mem_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.warmup_steps = 1000  # Delay learning <- should improve overall learning as delays learning until more experiences are "collected"

        # Set up the Q-Network and Target Network
        input_dim = env.observation_space.shape[0]
        self.q_net = DQN(input_dim, self.n_actions).to(self.device)
        self.target_net = DQN(input_dim, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.step_count = 0

    def getAction(self, state, evaluation=False):
        if not evaluation and random.random() < self.ep:
            return self.env.action_space.sample()
        
        state_mean = np.array([-0.3, 0.0]) # -0.3 is center of -1.2,0.6 (position) and 0.0 is center of -0.07,0.07 (velocity)
        state_std = np.array([0.9,0.07]) # 0.9 = (0.6 - -1.2) / 2
        normalized_state = (state - state_mean) / state_std

        state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return int(torch.argmax(q_values).item())

    def storeTransition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < max(self.warmup_steps, self.batch_size):
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        state_mean = np.array([-0.3, 0.0], dtype=np.float32) # -0.3 is center of -1.2,0.6 (position) and 0.0 is center of -0.07,0.07 (velocity)
        state_std = np.array([0.9,0.07], dtype=np.float32) # 0.9 = (0.6 - -1.2) / 2

        states = (torch.FloatTensor(states).to(self.device) - state_mean) / state_std # normalized

        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        
        # normalized next_states
        next_states = (torch.FloatTensor(next_states).to(self.device) - state_mean) / state_std

        
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1, keepdim=True) # action selection
            max_next_q = self.target_net(next_states).gather(1, next_actions) # action evaluation
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = nn.SmoothL1Loss()(q_values, target_q) # help stabilize learning 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.ep = max(self.ep_min, self.ep * self.ep_decay)

def dqnLearning(agent, env, episodes=500):
    rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.getAction(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            velocity_bonus = 0.1 * abs(next_state[1])
            position_min = env.observation_space.low[0]
            position_max = env.observation_space.high[0] 
            position_reward = 0.3 * ((next_state[0] - position_min) / (position_max - position_min))

            new_reward = reward + velocity_bonus + position_reward

            if terminated:
                # bonuses attempted: 10.0, 100.0
                new_reward += 100.0 # Reached the flag

            clipped_rewards = np.clip(new_reward, -1.0, 1.0)
            agent.storeTransition(state, action, clipped_rewards, next_state, done)

            agent.update()
            state = next_state
            total_reward += new_reward

        rewards.append(total_reward)
        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.ep:.3f}")
    return rewards

# Note that there are no bonuses to the rewards in this function, they are raw
def evaluate(env, agent, num_episodes=10):
    total_rewards = []
    total_successes = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        success = False

        while not done:
            action = agent.getAction(state, evaluation=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            if terminated:
                success = True
        
        total_successes.append(success)
        total_rewards.append(episode_reward)
    return np.mean(total_rewards), np.std(total_rewards), sum(total_successes)

def main():
    env = gym.make("MountainCar-v0")
    agent = DQNAgent(env)
    # episodes attempted: 1000, 2000
    dqnLearning(agent, env, episodes=2000)

    # eval_env = gym.make("MountainCar-v0", render_mode="human")
    eval_env = gym.make("MountainCar-v0")
    
    mean_reward, std_reward, success_count = evaluate(eval_env, agent, num_episodes=10)

    print(f"Evaluation over 10 episodes: Average Reward = {mean_reward} +/- {std_reward} :: Number of Successes = {success_count}")    
    eval_env.close()

if __name__ == "__main__":
    main()
