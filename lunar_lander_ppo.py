from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
import imageio # for creating a gif
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

class PPOAgent:
    def __init__(self, env="LunarLander-v3", device="cpu"):
        gym_env = gym.make(env)
        # wrap env and normalize obs and rewards (https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)
        self.env = DummyVecEnv([lambda: gym_env])
        self.env = VecNormalize(self.env)
        self.model = PPO(
            "MlpPolicy", # multilayer policy
            self.env, 
            verbose=1, # prints training logs for easy viewing
            device=device, # cpu
            n_steps=1024, # attempted: 2048, 1024: Number of steps collected
            batch_size=64, # size for updates
            n_epochs=10, 
            gamma=0.99, # discount factor
            gae_lambda=0.95, # for advantage estimation
            ent_coef=0.0, # 0 forced exploration
            learning_rate=1e-4, # attempted: 1e-4, 2.5e-4 
            clip_range=0.2, 
            policy_kwargs=dict(
                # (https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html)
                net_arch=[dict(pi=[256, 256], vf=[256, 256])] # two hidden layers of 256 units for actor and critic
            ),
        )

    # train the agent
    def train(self, timesteps=500_000):
        self.model.learn(total_timesteps=timesteps)

    # evaluate the agent
    def evaluate(self, episodes=10):
        self.env.training = False
        self.env.norm_reward = False
        total_rewards = []
        total_successes = []
        for _ in range(episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            success = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True) # deterministic means no randomness
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward[0]
                if done[0]:
                    break
            
            # Lunar Lander env success = score > 200
            if episode_reward > 200: 
                success = True

            total_successes.append(success)
            total_rewards.append(episode_reward)
                
        return np.mean(total_rewards), np.std(total_rewards), sum(total_successes)

    # save the model
    def save(self, path="ppo/ppo_model"):
        self.model.save(path)
        print(f"Model saved to {path}.zip")

    # load the model
    def load(self, path="ppo/ppo_model"):
        self.model = PPO.load(path, env=self.env)
        print(f"Model loaded from {path}.zip")

# for creating a gif of evaluation runs
def record(agent, env_name, filename="img/ppo_learning_lunar_lander.gif", episodes=10):
    frames = []
    env = gym.make(env_name, render_mode="rgb_array")

    for i in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            frame = env.render()
            frames.append(frame)
            action, _ = agent.predict(state, deterministic=True)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    env.close()

    imageio.mimsave(filename, frames, duration=1/30, loop=0)
    print(f"Saved gif to {filename}")

def main():
    agent = PPOAgent()
    agent.train(timesteps=1_500_000) # timesteps attempted: 500_000, 1_500_000
    
    # evaluate on 10 episodes
    mean_reward, std_reward, success_count = agent.evaluate(episodes=10)
    print(f"Evaluation over 10 episodes: Average Reward = {mean_reward} +/- {std_reward} :: Number of Successes = {success_count}")    

    # save and record if perfecet success
    if success_count == 10:
        agent.save("ppo/ppo_model")
        record(agent, "LunarLander-v3", filename="img/ppo_learning_lunar_lander.gif", episodes=10)

if __name__ == "__main__":
    main()