from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
import imageio # for creating a gif
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

class PPOAgent:
    def __init__(self, env="LunarLander-v3", device="cpu"):
        # wrap env and normalize obs and rewards (https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)
        env_vec = SubprocVecEnv([lambda: gym.make(env) for _ in range(4)])
        self.env = VecNormalize(env_vec)
        self.model = PPO(
            "MlpPolicy", # multilayer policy
            self.env, 
            verbose=1, # prints training logs for easy viewing
            device=device, # cpu
            n_steps=2048, # attempted: 2048, 1024: Number of steps collected
            batch_size=256, #atempted: 64, 256 size for updates
            n_epochs=10, 
            gamma=0.99, # discount factor
            gae_lambda=0.95, # for advantage estimation
            ent_coef=0.01, # attempted: 0.0, 0.1, 0.01
            learning_rate=lambda f: 2.5e-4 * f, # attempted: 1e-4, 2.5e-4 
            clip_range=0.1, # attempted: 0.1, 0.2 
            policy_kwargs=dict(
                # (https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html)
                net_arch=[dict(pi=[512, 256, 128], vf=[512, 256, 128])] # hidden layers
            ),
        )

    # train the agent
    def train(self, timesteps=500_000):
        self.model.learn(total_timesteps=timesteps)

    # evaluate the agent
    def evaluate(self, episodes=50):
        self.env.training = False
        self.env.norm_reward = False
        total_rewards = []
        total_successes = []
        for _ in range(episodes):
            obs = self.env.reset()
            done = [False]
            episode_reward = 0
            success = False

            while not done[0]:
                action, _ = self.model.predict(obs, deterministic=True) # deterministic means no randomness
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward[0]
            
            # Lunar Lander env success = score > 200
            if episode_reward > 200: 
                success = True

            total_successes.append(success)
            total_rewards.append(episode_reward)
                
        return np.mean(total_rewards), np.std(total_rewards), sum(total_successes)

# for creating a gif of evaluation runs
def record(agent, env_name, filename="img/ppo_learning_lunar_lander.gif", episodes=10):
    frames = []
    env = gym.make(env_name, render_mode="rgb_array")
    # Load the normalization stats for the environment that were saved before
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load("ppo/vecnormalize.pkl", env)
    env.training = False
    env.norm_reward = False

    frame_idx = 0

    for i in range(episodes):
        state = env.reset()
        done = False

        while not done:
            if frame_idx % 2 == 0: # reduce the number of frames captured due to RAM 
                frame = env.render()
                frames.append(frame)

            action, _ = agent.model.predict(state, deterministic=True)
            state, reward, done, info = env.step(action)
            frame_idx += 1

    env.close()

    imageio.mimsave(filename, frames, duration=1/30, loop=0)
    print(f"Saved gif to {filename}")

def main():
    agent = PPOAgent()
    agent.train(timesteps=2_000_000) # timesteps attempted: 500_000, 1_500_000, 2_000_000, 5_000_000
    agent.env.save("ppo/vecnormalize.pkl") # saving for use in the record function to make gif

    # evaluate on 50 episodes
    mean_reward, std_reward, success_count = agent.evaluate(episodes=50)
    print(f"Evaluation over 50 episodes: Average Reward = {mean_reward} +/- {std_reward} :: Number of Successes = {success_count}")    

    record(agent, "LunarLander-v3", filename="img/ppo_learning_lunar_lander.gif", episodes=10)
        
if __name__ == "__main__":
    main()