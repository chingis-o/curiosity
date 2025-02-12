import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import gym
import numpy as np

# Hyperparameters
learning_rate = 3e-4
gamma = 0.99
clip_epsilon = 0.2
ppo_epochs = 10
batch_size = 64
num_steps = 2048
max_training_steps = 100000

# Environment setup
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Policy Network (Actor)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

# Value Network (Critic)
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# PPO Agent
class PPO:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def compute_returns(self, rewards, dones, next_state, gamma=0.99):
        returns = []
        R = self.critic(torch.FloatTensor(next_state)).detach().numpy()[0]
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * (1 - dones[step])
            returns.insert(0, R)
        return torch.tensor(returns)

    def update(self, states, actions, log_probs_old, returns, advantages):
        states = torch.FloatTensor(states)
        actions = torch.tensor(actions, dtype=torch.int64)
        log_probs_old = torch.stack(log_probs_old).detach()
        returns = torch.tensor(returns)
        advantages = torch.tensor(advantages)

        # Optimize policy for K epochs
        for _ in range(ppo_epochs):
            # Evaluate actions and values
            probs = self.actor(states)
            dist = Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            # Compute ratio (pi_theta / pi_theta_old)
            ratio = torch.exp(log_probs - log_probs_old)

            # Surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
            actor_loss = (-torch.min(surr1, surr2) - 0.01 * entropy).mean()

            # Critic loss
            values = self.critic(states).squeeze()
            critic_loss = F.mse_loss(values, returns)

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

# Training Loop
def train():
    agent = PPO(state_dim, action_dim)
    state = env.reset()
    done = False
    total_rewards = []

    for step in range(max_training_steps):
        states, actions, log_probs_old, rewards, dones, next_state = [], [], [], [], [], None

        for t in range(num_steps):
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs_old.append(log_prob)
            rewards.append(reward)
            dones.append(done)

            state = next_state
            if done:
                state = env.reset()

        # Compute returns and advantages
        returns = agent.compute_returns(rewards, dones, next_state, gamma)
        values = agent.critic(torch.FloatTensor(states)).squeeze().detach().numpy()
        advantages = returns - values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update the agent
        agent.update(states, actions, log_probs_old, returns, advantages)

        # Logging
        total_rewards.append(sum(rewards))
        if step % 10 == 0:
            print(f"Step: {step}, Avg Reward: {np.mean(total_rewards[-10:])}")

    env.close()

if __name__ == "__main__":
    train()