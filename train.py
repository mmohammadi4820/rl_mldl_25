"""Train an RL agent on the OpenAI Gym Hopper environment using
REINFORCE and Actor-critic algorithms
"""

import argparse

import torch
import gym
import pickle
from env.custom_hopper import *
from agent import Agent, Policy

# changed by Erfan


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-episodes", default=500, type=int, help="Number of training episodes"
    )
    parser.add_argument(
        "--print-every", default=10, type=int, help="Print info every <> episodes"
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="network device [cpu, cuda]"
    )

    return parser.parse_args()


args = parse_args()


def main():
    ########################################## For draw charts to different bt ACT & REIN  ########################################
    episode_rewards = []
    ########################################## For draw charts to different bt ACT & REIN  ############

    env = gym.make("CartPole-v1")  # 1t_change
    # env = gym.make('CustomHopper-target-v0')

    print("Action space:", env.action_space)
    print("State space:", env.observation_space)

    # print('Dynamics parameters:', env.get_parameters())

    """
		Training
	"""
    observation_space_dim = env.observation_space.shape[-1]

    # action_space_dim = env.action_space.shape[-1]
    action_space_dim = env.action_space.n

    policy = Policy(observation_space_dim, action_space_dim)
    # agent = Agent(policy, device=args.device)
    ########################################## For draw charts to different bt ACT & REIN  ########################################
    agent = Agent(
        policy, device=args.device, use_critic=True
    )  # [ use_critic--> False ]  means agent= REINFORCE

    ########################################## For draw charts to different bt ACT & REIN  ########################################
    #
    # TASK 2 and 3: interleave data collection to policy updates
    #

    for episode in range(args.n_episodes):
        done = False
        train_reward = 0
        state = env.reset()  # -----change from state, _  , 4th
        # Reset the environment and observe the initial state

    while not done:
        # changes to
        # action, action_probabilities = agent.get_action(state) comment in task 3
        action, action_probabilities, value = agent.get_action(state)

        # because its for gym 0.26 +   ,comment in  4th
        # next_state, reward, terminated, truncated, info = env.step(action)  # <-- Corrected here
        # done = terminated or truncated
        next_state, reward, done, info = env.step(action)
        # comment in task 3
        # agent.store_outcome(state, next_state, action_probabilities, reward, done)
        agent.store_outcome(
            state, next_state, action_probabilities, reward, done, value
        )
        """ Reason: We need to get the value from get_action
          and give it to store_outcome to use later in update_policy.
        """

        train_reward += reward

        # primary code :
        # action, action_probabilities = agent.get_action(state)
        # previous_state = state

        # state, reward, done, info = env.step(action.detach().cpu().numpy())

        # agent.store_outcome(previous_state, state, action_probabilities, reward, done)

    ########################################## For draw charts to different bt ACT & REIN  ########################################
    episode_rewards.append(train_reward)
    ########################################## For draw charts to different bt ACT & REIN  ########################################
    if (episode + 1) % args.print_every == 0:
        print("Training episode:", episode)
        print("Episode return:", train_reward)

    torch.save(agent.policy.state_dict(), "model.mdl")

    ########################################## For draw charts to different bt ACT & REIN  ########################################
    with open("rewards_actor_critic.pkl", "wb") as f:  # rewards_actor_critic.pkl
        pickle.dump(episode_rewards, f)


########################################## For draw charts to different bt ACT & REIN  ########################################

if __name__ == "__main__":
    main()
