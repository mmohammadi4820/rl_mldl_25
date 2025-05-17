import pickle
import matplotlib.pyplot as plt

# Load reward data
with open("rewards_actor_critic.pkl", "rb") as f:
    rewards_ac = pickle.load(f)

with open("rewards_reinforce.pkl", "rb") as f:
    rewards_reinforce = pickle.load(f)


# Moving average for smoothing
def moving_avg(data, window=10):
    return [
        sum(data[max(0, i - window) : i + 1]) / len(data[max(0, i - window) : i + 1])
        for i in range(len(data))
    ]


plt.figure(figsize=(12, 6))
plt.plot(moving_avg(rewards_ac), label="Actor-Critic")
plt.plot(moving_avg(rewards_reinforce), label="REINFORCE")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("REINFORCE vs Actor-Critic")
plt.legend()
plt.grid(True)
plt.show()
