import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

# changes by Erfan


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)

        # Learned standard deviation for exploration at training time
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space) + init_sigma)

        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm
        # changes
        self.fc1_critic = torch.nn.Linear(self.state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_value = torch.nn.Linear(self.hidden, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        """
            Critic
        """
        # TASK 3: forward in the critic network
        # changes
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        value = self.fc3_critic_value(x_critic)

        return normal_dist, value.squeeze(-1)


class Agent(object):
    def __init__(self, policy, device="cpu", use_critic=True):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        self.state_values = []  # added  in task 3

        # changes
        self.use_critic = use_critic  # <-- ADD THIS task 2

    def update_policy(self):
        action_log_probs = (
            torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        )
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = (
            torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        )
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)
        # added in task 3
        (
            self.states,
            self.next_states,
            self.action_log_probs,
            self.rewards,
            self.done,
            self.state_values,
        ) = ([], [], [], [], [], [])
        """ To calculate the advantage accurately,
          we need to keep the V(s) corresponding to each state."""
        """(
            self.states,
            self.next_states,
            self.action_log_probs,
            self.rewards,
            self.done,
        ) = ([], [], [], [], []) """

        values = torch.stack(self.state_values).to(self.train_device).squeeze(-1)

        # changes
        returns = discount_rewards(rewards, gamma=self.gamma)

        if self.use_critic:
            _, values = self.policy(states)
            advantages = returns - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            actor_loss = -(action_log_probs * advantages).sum()
            critic_loss = F.mse_loss(values, returns)

            loss = actor_loss + critic_loss
        else:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            loss = -(action_log_probs * returns).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return

        #
        # TASK 3:
        #   - compute boostrapped discounted return estimates
        #   - compute advantage terms
        #   - compute actor loss and critic loss
        #   - compute gradients and step the optimizer

    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)

        # normal_dist, _ = self.policy(x) # task 3
        normal_dist, value = self.policy(x)
        """  Task 3 :
        Reason: In Actor-Critic, we need to store the value 
        (i.e. the estimate of V(s)) to be used in calculating
          advantage and critic loss. """

        if evaluation:  # Return mean
            # changes
            action = torch.argmax(normal_dist.mean).item()
            # return action, None  # task3
            return action, None, value

        else:  # Sample from the distribution
            action = torch.argmax(normal_dist.sample()).item()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = torch.log_softmax(normal_dist.sample(), dim=0)[action]

            # return action, action_log_prob  # task 3
            return action, action_log_prob, value

    # def store_outcome(self, state, next_state, action_log_prob, reward, done):  # task 3
    def store_outcome(self, state, next_state, action_log_prob, reward, done, value):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
        self.state_values.append(value)  # added in task 3
