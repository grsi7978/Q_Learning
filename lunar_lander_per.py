import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import imageio # for creating a gif

# Prioritized Replay Buffer
class PER:
    def __init__(self, capacity, alpha):
        self.capacity = capacity # max number of stored transitions
        self.alpha = alpha # exponent for prioritization
        self.buffer = [] # buffer for transitions
        self.priorities = np.zeros((capacity,), dtype=np.float32) # array of priorities
        self.pos = 0 # used for ciruclar tracking in the buffer

    # add new transition and priority to the buffer
    def add(self, transition, priority=1.0):
        # if there is still room in the buffer then add experience and priority
        if len(self.buffer) <self.capacity:
            self.buffer.append(transition)
        else:
            # buffer is full so overwrite the oldest item
            self.buffer[self.pos] = transition 
        self.priorities[self.pos] = min(priority, 1e3) # record the priority for transition
        self.pos = (self.pos + 1) % self.capacity # update the position for the next transition

    # sample batch of transistions based on priority
    def sample(self, batch_size, beta=0.4):
        # get the list of priorities
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]
        
        # determine probabilities by priority^alpha
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum() # normalize the probabilities

        # select the indexes of the samples
        indicies = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        # get the samples
        samples = [self.buffer[idx] for idx in indicies]

        total = len(self.buffer)
        weights = (total * probabilities[indicies]) ** -beta
        weights /= weights.max()  # Normalize for stability
        
        return samples, indicies, weights

    # Update the priorities of transitions
    def update_priorities(self, indicies, priorities):
        for idx, priority in zip(indicies, priorities):
            # update priority values
            self.priorities[idx] = min(priority, 1e3)
    
    # return the length of the buffer
    def __len__(self):
        return len(self.buffer)

# Neural Network
class DQN(nn.Module):
    # input_dim: input dimensions
    # output_dim: output dimensions    
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # sequential neural network using 3 layers
        # using ReLU for activation functions
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256), # attempted 128 previously
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    # forward pass through the network
    def forward(self, x):
        return self.model(x)
    
class DQNAgent:
    # learning rates attempted: 1e-3, 5e-4
    # target_update_freq  attempted: 500
    # decay attempted: 0.99, 0.995
    # epsilon min attempted:0.05, 0.01
    # target update frequency attempted: 500, 1000
    def __init__(self, env, gamma=0.99, lr=1e-3, batch_size=64, ep=1.0, ep_min=0.01, ep_decay=0.99, mem_size=100000, target_update_freq=500):
        self.env = env
        self.gamma = gamma # discount factor for future rewards
        self.ep = ep # epsilon for exploration
        self.ep_min = ep_min # minimum epsilon
        self.ep_decay = ep_decay # epsilon decay factor
        self.batch_size = batch_size # batch size for training
        self.target_update_freq = target_update_freq # target network update frequency
        self.n_actions = env.action_space.n # number of possilbe actions
        self.memory = PER(mem_size, alpha=0.5) # initialize replay buffer with alpha=0.5. Alphas attempted: 0.5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # warmup_steps attempted: 5000, 10000
        self.warmup_steps = 10000  # Delay learning <- should improve overall learning as delays learning until more experiences are "collected"

        # Set up the Q-Network and Target Network
        input_dim = env.observation_space.shape[0]
        self.q_net = DQN(input_dim, self.n_actions).to(self.device)
        self.target_net = DQN(input_dim, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict()) # synchronizes weights

        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.step_count = 0

        # sampling bias correction for PER
        self.beta = 0.4 # betas attempted: 0.4
        self.beta_increment = 5e-4 # increase beta. increments attempted: 1e-3, 5e-4

    # convert to tensor
    def toTensor(self, array):
        tensor = torch.as_tensor(array, dtype=torch.float32).to(self.device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    # greedy action selection
    def getAction(self, state, evaluation=False):
        if not evaluation and random.random() < self.ep:
            return self.env.action_space.sample() # random action exploration
        
        # evaluate q-vals for state and choose action with max 
        # state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        state_tensor = self.toTensor(state)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return int(torch.argmax(q_values).item())
    
    # store experience in the replay buffer
    def storeTransition(self, state, action, reward, next_state, done):
        # state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        state_tensor = self.toTensor(state)
        # next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
        next_state_tensor = self.toTensor(next_state)
        action_tensor = torch.tensor([[action]]).to(self.device)
        reward_tensor = torch.tensor([[reward]]).to(self.device)
        done_tensor = torch.tensor([[done]]).float().to(self.device)

        # without gradients to save memory and speed things up
        with torch.no_grad():
            q = self.q_net(state_tensor).gather(1, action_tensor) # current Q-Value

            # next Q-value
            next_a = self.q_net(next_state_tensor).argmax(1, keepdim=True)
            next_q = self.target_net(next_state_tensor).gather(1, next_a)
        
            target_q = reward_tensor + self.gamma * next_q * (1 - done_tensor) # target Q-Value
            td_error = torch.abs(target_q - q).detach().cpu().item()
        
        # add transition to memory with td_error as priority   
        self.memory.add((state, action, reward, next_state, done), td_error + 1e-5)

    def update(self):
        # skip the training if the replay buffer doesn't have enough samples
        if len(self.memory) < max(self.warmup_steps, self.batch_size):
            return

        # sample the experiences
        batch, indicies, weights = self.memory.sample(self.batch_size, self.beta)
        # increase beta over time
        self.beta = min(1.0, self.beta + self.beta_increment)

        # convert to tensors
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)        
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        # compute q-values for actions
        q_values = self.q_net(states).gather(1, actions)

        # compute target q-vals with target network
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1, keepdim=True) # action selection
            max_next_q = self.target_net(next_states).gather(1, next_actions) # action evaluation
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        # update priorities in replay buffer based on TD errors
        td_errors = (target_q - q_values).detach().cpu().numpy().squeeze()
        new_priorities = np.abs(td_errors) + 1e-5
        self.memory.update_priorities(indicies, new_priorities)

        # calculate loss
        loss = (weights * nn.SmoothL1Loss(reduction='none')(q_values, target_q)).mean()
        self.optimizer.zero_grad()
        loss.backward() # backpropogation
        self.optimizer.step() # update the parameters

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.ep = max(self.ep_min, self.ep * self.ep_decay)

def dqnLearning(agent, env, episodes=1000):
    rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # choose action
            action = agent.getAction(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.storeTransition(state, action, reward, next_state, done)

            # actual agent training
            agent.update()
            state = next_state
            total_reward += reward

        rewards.append(total_reward)  # track the episode rewards
        # log the status / rewards every 100 episodes
        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.ep:.3f}")
    return rewards

# actual evaluation of the agent / actions taken
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
            # set evaluation=True to prevent exploration
            action = agent.getAction(state, evaluation=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        # success condition per environment    
        if episode_reward > 200:
            success = True

        total_successes.append(success)
        total_rewards.append(episode_reward)
    return np.mean(total_rewards), np.std(total_rewards), sum(total_successes)

# for creating a gif of evaluation runs
def record(agent, env_name, filename="img/per_dqn_learning_lunar_lander.gif", episodes=10):
    frames = []
    env = gym.make(env_name, render_mode="rgb_array")

    for i in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            frame = env.render()
            frames.append(frame)
            action = agent.getAction(state, evaluation=True)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    env.close()

    imageio.mimsave(filename, frames, duration=1/30, loop=0)
    print(f"Saved gif to {filename}")

def main(seed=0):
    # set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    env = gym.make("LunarLander-v3")
    agent = DQNAgent(env)
    # episodes attempted: 1000, 2000
    rewards = dqnLearning(agent, env, episodes=2000)

    # plot training reward curve
    # plt.figure(figsize=(15,10))
    # plt.plot(rewards, label='Total Reward per Episode')
    # window=50
    # smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    # plt.plot(range(window-1, len(smoothed)+window-1), smoothed, label=f'{window}-Episode Moving Average', linewidth=2)
    # plt.xlabel('Episode')
    # plt.ylabel('Total Episode Reward')
    # plt.title('Training Reward over Time')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('img/per_dqn_lunar_lander_reward_time_01.png')
    # plt.close()

    # eval_env = gym.make("LunarLander-v3", render_mode="human")
    eval_env = gym.make("LunarLander-v3")
    
    mean_reward, std_reward, success_count = evaluate(eval_env, agent, num_episodes=10)

    print(f"Evaluation over 10 episodes: Average Reward = {mean_reward} +/- {std_reward} :: Number of Successes = {success_count}")    
    eval_env.close()

    if success_count == 10:
        plt.figure(figsize=(6, 6))
        plt.pie([success_count, 10 - success_count], labels=["Success", "Failure"], autopct='%1.1f%%', colors=['green', 'red'])
        plt.title('Success Rate')
        plt.tight_layout()
        plt.savefig('img/per_dqn_lunar_lander_success_rate_01.png')
        plt.close()

    if success_count == 10:
        record(agent, "LunarLander-v3", filename="img/per_dqn_learning_lunar_lander.gif", episodes=10)

if __name__ == "__main__":
    # seeds = [0, 23, 52, 123, 256, 999, 1233] # seeds for testing
    seeds = [0, 23, 52, 123, 256, 999, 1233] # seed 999 had the best success rate so will contiue testing with it
    for seed in seeds:
        print(f"\nrunning with seed {seed}")
        main(seed=seed)
