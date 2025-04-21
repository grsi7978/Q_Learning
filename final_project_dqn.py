import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial 

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
    def __init__(self, env, gamma=0.99, lr=1e-3, batch_size=64, ep=1.0, ep_min=0.01, ep_decay=0.995, mem_size=100000, target_update_freq=5000):
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
        
        state_mean = np.array([-0.5, 0.0])
        state_std = np.array([0.7,0.07])
        normalized_state = (state - state_mean) / state_std

        state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(self.device)
        # state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return int(torch.argmax(q_values).item())

    def storeTransition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        state_mean = np.array([-0.5, 0.0], dtype=np.float32)
        state_std = np.array([0.7,0.07], dtype=np.float32)

        # states = torch.FloatTensor(states).to(self.device)
        states = (torch.FloatTensor(states).to(self.device) - state_mean) / state_std

        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        
        # next_states = torch.FloatTensor(next_states).to(self.device)
        next_states = (torch.FloatTensor(next_states).to(self.device) - state_mean) / state_std

        
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            # max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            # target_q = rewards + self.gamma * max_next_q * (1 - dones)
            next_actions = self.q_net(next_states).argmax(1, keepdim=True)
            max_next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        # loss = nn.MSELoss()(q_values, target_q)
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

            # velocity_bonus = 0.1 * abs(next_state[1])  # Encourage speed
            # forward_bonus = 0.5 * (next_state[0] - state[0])  # Encourage moving forward
            # shaped_reward = reward + velocity_bonus + forward_bonus
            # if terminated:
            #     shaped_reward += 10.0 # Reached the flag

            shaped_reward = reward + 0.1 * abs(next_state[1])

            agent.storeTransition(state, action, shaped_reward, next_state, done)
            # agent.storeTransition(state, action, reward, next_state, done)

            agent.update()
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.ep:.3f}")
    return rewards

def main():
    env = gym.make("MountainCar-v0")
    agent = DQNAgent(env)
    rewards = dqnLearning(agent, env, episodes=1000)

    # Plot learning curve
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.grid()
    plt.show()

    # eval_env = gym.make("MountainCar-v0", render_mode="human")
    eval_env = gym.make("MountainCar-v0")
    total_rewards = []
    
    for _ in range(10):
        state, _ = eval_env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.getAction(state, evaluation=True)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
        total_rewards.append(episode_reward)

    print(f"Eval Average Reward over 10 eps: {np.mean(total_rewards):.2f}")
    eval_env.close()

if __name__ == "__main__":
    main()
