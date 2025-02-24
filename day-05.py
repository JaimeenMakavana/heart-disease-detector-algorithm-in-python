# import numpy as np
# # problem : push notification optimization
# class MultiArmedBandit:
#     def __init__(self, n_arms):
#         self.n_arms = n_arms
#         self.counts = np.zeros(n_arms)
#         self.values = np.zeros(n_arms)
    
#     def select_arm(self):
#         if 0 in self.counts:
#             return np.argmin(self.counts)
        
#         ucb_values = self.values + np.sqrt(2 * np.log(np.sum(self.counts)) / self.counts)
#         return np.argmax(ucb_values)
    
#     def update(self, chosen_arm, reward):
#         self.counts[chosen_arm] += 1
#         n = self.counts[chosen_arm]
#         self.values[chosen_arm] = (self.values[chosen_arm] * (n - 1) + reward) / n

# arms = ["Discount", "New Connection", "Trending Post"]
# bandit = MultiArmedBandit(len(arms))

# true_click_rates = [0.05, 0.10, 0.30]

# for _ in range(1000):
#     chosen_arm = bandit.select_arm()
#     reward = np.random.binomial(1,true_click_rates[chosen_arm])
#     bandit.update(chosen_arm, reward)

# print("final estimated rewards:", bandit.values)
# print("Notification selection counts:", bandit.counts)
import numpy as np
import matplotlib.pyplot as plt

# Multi-Armed Bandit with UCB1
class MultiArmedBandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
    
    def select_arm(self):
        if 0 in self.counts:
            return np.argmin(self.counts)
        
        ucb_values = self.values + np.sqrt(2 * np.log(np.sum(self.counts)) / self.counts)
        return np.argmax(ucb_values)
    
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        self.values[chosen_arm] = (self.values[chosen_arm] * (n - 1) + reward) / n

# Arms and true click rates
arms = ["Discount", "New Connection", "Trending Post"]
true_click_rates = [0.05, 0.10, 0.30]

bandit = MultiArmedBandit(len(arms))
iterations = 1000

# Track estimated click rates over time
estimated_rewards = np.zeros((len(arms), iterations))

for i in range(iterations):
    chosen_arm = bandit.select_arm()
    reward = np.random.binomial(1, true_click_rates[chosen_arm])
    bandit.update(chosen_arm, reward)

    # Store estimated rewards at each step
    estimated_rewards[:, i] = bandit.values

# Plotting the estimated rewards over time
plt.figure(figsize=(10, 5))
for i, arm in enumerate(arms):
    plt.plot(range(iterations), estimated_rewards[i], label=arm)

plt.axhline(y=true_click_rates[0], color='r', linestyle='dashed', alpha=0.5, label="True Click Rate (Discount)")
plt.axhline(y=true_click_rates[1], color='g', linestyle='dashed', alpha=0.5, label="True Click Rate (New Connection)")
plt.axhline(y=true_click_rates[2], color='b', linestyle='dashed', alpha=0.5, label="True Click Rate (Trending Post)")

plt.xlabel("Iterations")
plt.ylabel("Estimated Click Rate")
plt.title("Convergence of Estimated Click Rates in UCB1")
plt.legend()
plt.show()
