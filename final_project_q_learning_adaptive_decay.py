import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import imageio # for creating a gif

# Basic random agent used to test initial setup
class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    # Select action at random
    def getAction(self, observation, evaluation=False):
        return self.action_space.sample()
    
# Basic function to evaluate the agent
# Note that there are no bonuses to the rewards in this function, they are raw
def evaluate(env, agent, num_episodes=20000):
    total_rewards = []
    total_successes = []
    for episode in range(num_episodes):
        success = False
        observation, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.getAction(observation, evaluation=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            if terminated:
                success = True
        total_rewards.append(episode_reward)
        total_successes.append(success)
    return np.mean(total_rewards), np.std(total_rewards), sum(total_successes)

# Q-Learning Agent
class QAgent:
    # bucketcombos tried: (40,30), (24,18), (18,14), (20,15)*
    # ep_decay tried: 0.995, 0.99
    # ep_min tried: 0.01, 0.05
    def __init__(self, env, buckets=(20,15), alpha=0.1, gamma=0.99,ep=1.0, ep_min=0.01, ep_decay=0.995):
        self.env = env
        self.buckets = buckets
        self.alpha = alpha
        self.gamma = gamma
        self.ep = ep
        self.ep_min = ep_min
        self.ep_decay = ep_decay
        self.q_table = np.zeros(self.buckets + (env.action_space.n,))
        self.bounds = list(zip(env.observation_space.low, env.observation_space.high))

    # convert the continues state into indexes
    def mapPosVel(self, state):
        ratios = []
        for i in range(len(state)):
            low, high = self.bounds[i]
            # Calculate the ratio of the current state wihin its range [0,1]
            ratios.append((state[i] - low)/(high-low))
        new_state = []
        for i, ratio in enumerate(ratios):
            # scale the ratio to the # of buckets
            # round to the nearest bucket index
            new_state.append(int(round((self.buckets[i] - 1) * ratio)))
            # make sure that new_state[i] is within bounds
            new_state[i] = min(self.buckets[i] - 1, max(0,new_state[i]))
        return tuple(new_state)

    # greedy action selection    
    def getAction(self, state, evaluation=False):
        if not evaluation and np.random.random() < self.ep:
            # exploration
            return self.env.action_space.sample()
        else:
            # exploitation
            state_buckets = self.mapPosVel(state) # convert to buckets
            # Return action with highest qvalue
            return int(np.argmax(self.q_table[state_buckets]))
            
    def update(self, state, action, reward, next_state, done):
        # bucketize
        current_state = self.mapPosVel(state)
        next_state_bucket = self.mapPosVel(next_state)
        # get best possible future reward
        best_next_action = np.max(self.q_table[next_state_bucket])
        
        # Bellman equation for target q-value
        td_target = reward + self.gamma * best_next_action * (1 - done)
        # TD error: the temporal difference
        td_error = td_target - self.q_table[current_state + (action,)]
        # Update the q-table w/ error and learning rate
        self.q_table[current_state + (action,)] += self.alpha * td_error

    # Decay the epsilon adaptively. E.g. if the average reward is larger than
    # target then decay quickly. If reward is less than, then decay slowly.
    def decayEpAdapt(self, rewards, episode, target=-140, factor=0.95):
        wait = 500
        if episode < wait:
            self.ep = max(self.ep_min, self.ep * self.ep_decay)
            return

        avg_recent_rewards = np.mean(rewards[-50:]) # get the last 50 rewards only

        threshold = target + (episode / 20000) * (0- target)

        if avg_recent_rewards > threshold:
            self.ep = max(self.ep_min, self.ep * (self.ep_decay * factor))
        else:
            self.ep = max(self.ep_min, self.ep * (1 - (1 - self.ep_decay) / 2))

def qLearning(agent, env, num_episodes=20000, window=50):
    rewards =[]
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_over = False
        total_reward = 0
        while not episode_over:
            action = agent.getAction(state)
            # reward is a standard -1 for the cart game
            next_state, reward, terminated, truncated, _ = env.step(action)
            # states are [position, velocity] - want to reward for higher velocity meaning 
            # more likely that the car is going to make it over the hill
            velocity_reward = 0.1 * abs(next_state[1]) # reward higher velocity
            # want to reward being closer to the flag, position ranges from -1.2 -> 0.6
            # (https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/mountain_car.py)
            pos_min = env.observation_space.low[0] # -1.2
            pos_max = env.observation_space.high[0] #0.6
            # normalize so to only get 0->0.3 range bonus
            normalized_pos = (next_state[0] - pos_min) / (pos_max - pos_min) 
            position_reward = 0.3 * normalized_pos # reward for being closer to the flag
            reward += velocity_reward + position_reward

            # additionally there are 3 deterministic actions (see link above): 
            # - 0: accelerate to the left
            # - 1: Don't accelerate
            # - 2: Accelerate to the right
            # Additional rewards for proper actions
            if next_state[1] > 0 and action == 2: # accelerating right and moving right
                reward += 0.05
            elif next_state[1] < 0 and action == 0: # accelerating left and moving left
                reward += 0.05
            elif next_state[1] > 0 and action == 0: # accelerating left and moving right
                reward -= 0.05
            elif next_state[1] < 0 and action == 2: # accelerating right and moving left
                reward -= 0.05
            # if the cart is near the goal add a bonus
            if state[0] > 0.4:
                reward += 1.0
            if terminated: # reward for reaching the flag
                reward += 10.0

            episode_over = terminated or truncated

            agent.update(state, action, reward, next_state, episode_over)
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
        # boosts tried: 1.1, 1.2, 1.05
        agent.decayEpAdapt(rewards, episode)
        agent.alpha = max(0.01, agent.alpha * 0.9995) # decay alpha
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}: Total Reward = {total_reward}, Epsilon = {agent.ep:.3f}")         
    return rewards   

# for creating a gif of evaluation runs
def record(agent, env_name, filename="adaptive_q_learning.gif", episodes=10):
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

    imageio.mimsave(filename, frames, duration=1/30)
    print(f"Saved gif to {filename}")

def main():
    env = gym.make("MountainCar-v0")
    agent = QAgent(env)

    qLearning(agent, env, num_episodes=20000, window=50)

    # eval_env = gym.make("MountainCar-v0", render_mode="human")
    eval_env = gym.make("MountainCar-v0")    
    mean_reward, std_reward, success_count = evaluate(eval_env, agent, num_episodes=10)
    print(f"Evaluation over 10 episodes: Average Reward = {mean_reward} +/- {std_reward} :: Number of Successes = {success_count}")
    eval_env.close()

    record(agent, "MountainCar-v0", filename="adaptive_q_learning.gif", episodes=10)

if __name__ == "__main__":
    main()
