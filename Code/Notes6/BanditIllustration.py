# Run using python 3.12.7

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Switch to show or save files
dpl = False  # True = display plots, False = save plots

# Functions to make up data and implement bandit algorithms
class Bandit:
    """Class representing a 10-armed bandit with fixed stochastic rewards"""
    def __init__(self, k=10, steps=1000, seed=None):
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        self.k = k
        self.q_true = np.random.normal(0, 1, k)  # True action values
        self.optimal_action = np.argmax(self.q_true)

        # Precompute all rewards for each action and step
        self.rewards = np.random.normal(self.q_true[:, None], 1, (k, steps))

    def get_reward(self, action, step):
        """Returns the precomputed reward for a given action and step"""
        return self.rewards[action, step]

class EpsilonGreedyAgent:
    """Epsilon-Greedy agent that ensures each arm is pulled at least once."""
    def __init__(self, k=10, epsilon=0.1, initial_draws=False):
        self.k = k
        self.epsilon = epsilon
        self.q_est = np.zeros(k)  # Estimated values
        self.action_counts = np.zeros(k)  # Count of times each action is taken
        self.initial_draws = initial_draws  # Whether to force each arm to be drawn once
        self.initial_pulls = set() if initial_draws else None # Track which actions have been taken

    def select_action(self):
        """Selects an action using epsilon-greedy policy, ensuring each arm is pulled at least once."""
        # Force each arm to be taken at least once
        if self.initial_draws and len(self.initial_pulls) < self.k:
            action = len(self.initial_pulls)  # Pick the next untried arm
            self.initial_pulls.add(action)
            return action

        # Follow epsilon-greedy policy after each arm has been pulled at least once
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.k)
        return np.argmax(self.q_est)

    def update(self, action, reward):
        """Updates estimates using incremental formula"""
        self.action_counts[action] += 1
        self.q_est[action] += (reward - self.q_est[action]) / self.action_counts[action]

class UCB_Agent:
    """UCB agent requiring at least two pulls per action and incorporating sample variance."""

    def __init__(self, k=10, c=2):
        self.k = k
        self.c = c
        self.q_est = np.zeros(k)  # Sample means
        self.action_counts = np.zeros(k, dtype=int)  # Number of pulls per action
        self.t = 0  # Time step

        # Track rewards for sample variance calculation
        self.sum_rewards = np.zeros(k, dtype=np.float64)
        self.squared_rewards = np.zeros(k, dtype=np.float64)

    def select_action(self):
        """Selects an action using UCB with sample variance adjustment."""
        self.t += 1

        # Force each action to be taken at least twice before applying UCB formula
        if np.min(self.action_counts) < 2:
            return np.argmin(self.action_counts)  # Pick the least tried action

        # Compute sample means and sample variances
        sample_means = self.q_est
        sample_variances = np.zeros(self.k)

        for action in range(self.k):
            if self.action_counts[action] > 1:
                mean = self.q_est[action]
                squared_mean = self.squared_rewards[action] / self.action_counts[action]
                variance = max(squared_mean - mean**2, 1e-6)  # Ensure nonzero variance
            else:
                variance = 1  # Default to 1 if we don't have enough samples

            sample_variances[action] = variance

        # Compute UCB values incorporating sample variance
        ucb_values = sample_means + self.c * np.sqrt(sample_variances * np.log(self.t) / self.action_counts)

        return np.argmax(ucb_values)

    def update(self, action, reward):
        """Updates sample mean and variance estimates."""
        self.action_counts[action] += 1
        n = self.action_counts[action]

        self.sum_rewards[action] += reward
        self.squared_rewards[action] += reward**2

        # Update sample mean
        self.q_est[action] += (reward - self.q_est[action]) / n

class ThompsonSamplingAgent:
    """Thompson Sampling agent using Normal-Inverse-Gamma (NIG) conjugate priors."""

    def __init__(self, k=10, mu_0=0, kappa_0=1, alpha_0=1, beta_0=1):
        self.k = k  # Number of arms

        # Prior parameters
        self.mu_0 = np.full(k, mu_0, dtype=np.float64)  # Mean prior
        self.kappa_0 = kappa_0  # Strength of prior on mean
        self.alpha_0 = np.full(k, alpha_0, dtype=np.float64)  # Shape parameter (prior)
        self.beta_0 = np.full(k, beta_0, dtype=np.float64)  # Scale parameter (prior)

        # Posterior parameters (initialize with priors)
        self.kappa_n = np.full(k, kappa_0, dtype=np.float64)
        self.mu_n = np.full(k, mu_0, dtype=np.float64)
        self.alpha_n = np.full(k, alpha_0, dtype=np.float64)
        self.beta_n = np.full(k, beta_0, dtype=np.float64)

        # Track rewards for updating
        self.action_counts = np.zeros(k, dtype=int)  # Number of pulls per arm
        self.sum_rewards = np.zeros(k, dtype=np.float64)  # Sum of observed rewards
        self.squared_rewards = np.zeros(k, dtype=np.float64)  # Sum of squared rewards

    def select_action(self):
        """Selects an action by drawing samples from the posterior distribution."""
        sampled_means = np.zeros(self.k)

        for action in range(self.k):
            sampled_variance = 1 / np.random.gamma(self.alpha_n[action], 1 / self.beta_n[action])
            sampled_means[action] = np.random.normal(self.mu_n[action], np.sqrt(sampled_variance / self.kappa_n[action]))

        return np.argmax(sampled_means)

    def update(self, action, reward):
        """Updates the posterior parameters using conjugate Normal-Inverse-Gamma updates."""
        self.action_counts[action] += 1
        n = self.action_counts[action]

        self.sum_rewards[action] += reward
        self.squared_rewards[action] += reward ** 2

        sample_mean = self.sum_rewards[action] / n

        # Update parameters
        self.kappa_n[action] = self.kappa_0 + n
        self.mu_n[action] = (self.kappa_0 * self.mu_0[action] + n * sample_mean) / self.kappa_n[action]
        self.alpha_n[action] = self.alpha_0[action] + n / 2

        # Compute new beta_n using observed variance
        sum_sq_diff = self.squared_rewards[action] - n * sample_mean**2
        self.beta_n[action] = self.beta_0[action] + 0.5 * (sum_sq_diff + (self.kappa_0 * n / self.kappa_n[action]) * (sample_mean - self.mu_0[action])**2)

def run_experiment(bandit, agent, steps=1000):
    rewards = np.zeros(steps)
    optimal_action_count = np.zeros(steps)
    actions_taken = np.zeros(steps, dtype=int)

    for t in range(steps):
        action = agent.select_action()
        reward = bandit.get_reward(action, t)  # Ensure consistent rewards
        agent.update(action, reward)

        rewards[t] = reward
        actions_taken[t] = action
        if action == bandit.optimal_action:
            optimal_action_count[t] = 1

    if isinstance(agent, ThompsonSamplingAgent):
      estimated_q = agent.mu_n
    else:
      estimated_q = agent.q_est

    return rewards, optimal_action_count, actions_taken, estimated_q


def plot_results(agents, bandit, steps=1000, runs=2000, dpl=dpl, dir=script_dir):
    avg_rewards = {name: np.zeros(steps) for name in agents}
    optimal_actions = {name: np.zeros(steps) for name in agents}

    # Storage for MSE calculations
    mse_results = {name: {"mse_optimal": [], "mse_overall": []} for name in agents}

    for _ in range(runs):
        new_bandit = Bandit(steps=steps)
        true_q = new_bandit.q_true
        optimal_action = new_bandit.optimal_action

        for name, agent_factory in agents.items():
            agent = agent_factory()
            rewards, opt_actions, _, estimated_q = run_experiment(new_bandit, agent, steps)

            # Store rewards and optimal action selection counts
            avg_rewards[name] += rewards
            optimal_actions[name] += opt_actions

            # Compute MSE
            mse_optimal = (estimated_q[optimal_action] - true_q[optimal_action]) ** 2
            mse_overall = np.mean((estimated_q - true_q) ** 2)

            mse_results[name]["mse_optimal"].append(mse_optimal)
            mse_results[name]["mse_overall"].append(mse_overall)

    # Average rewards and optimal action percentages over all runs
    for name in agents:
        avg_rewards[name] /= runs
        optimal_actions[name] /= runs

    # Plot Average Rewards
    plt.figure(figsize=(12, 5))
    for name, rewards in avg_rewards.items():
        plt.plot(rewards, label=name)
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.title("10-Armed Bandit: Average Reward")
    if dpl:
        plt.show()
    else:
        plt.savefig(os.path.join(dir,"bandit_performance.png"))
        plt.close()

    # Plot Optimal Action Percentage
    plt.figure(figsize=(12, 5))
    for name, opt_action in optimal_actions.items():
        plt.plot(opt_action * 100, label=name)
    plt.xlabel("Steps")
    plt.ylabel("Optimal Action %")
    plt.legend()
    plt.title("10-Armed Bandit: Optimal Action %")
    if dpl:
        plt.show()
    else:
        plt.savefig(os.path.join(dir,"bandit_action.png"))
        plt.close()

    # Compute final MSE values
    mse_summary = {"Agent": [], "Avg MSE Optimal Action": [], "Avg MSE Overall": []}
    for name in agents:
        mse_summary["Agent"].append(name)
        mse_summary["Avg MSE Optimal Action"].append(np.mean(mse_results[name]["mse_optimal"]))
        mse_summary["Avg MSE Overall"].append(np.mean(mse_results[name]["mse_overall"]))

    # Convert to DataFrame for display
    df_mse_summary = pd.DataFrame(mse_summary)

    print(df_mse_summary)

#############################################################
# Run experiments

# Initialize agents
agents = {
    "Completely Random (ε=1)": lambda: EpsilonGreedyAgent(epsilon=1),
    "Epsilon-Greedy (ε=0.1)": lambda: EpsilonGreedyAgent(epsilon=0.1),
    "Completely Greedy (ε=0)": lambda: EpsilonGreedyAgent(epsilon=0),
    "UCB (c=2)": lambda: UCB_Agent(),
    "Thompson Sampling": lambda: ThompsonSamplingAgent()
}

# Look at outcome of one individual run
bandit0 = Bandit(seed = 25)

estimated_q_values = {}
reward_values = {}
opt_action_values = {}

# Run the experiment and tabulate results
for name, agent_factory in agents.items():
    agent = agent_factory()
    rewards, opt_actions, a_t, estimated_q = run_experiment(bandit0, agent, steps=1000)
    estimated_q_values[name] = estimated_q
    reward_values[name] = np.mean(rewards)
    opt_action_values[name] = np.sum(opt_actions)

df_summary = pd.DataFrame([reward_values, opt_action_values], index=["reward_values", "opt_action_values"])
print(df_summary)

df_results = pd.DataFrame(estimated_q_values)
df_results.insert(0, "True Q", bandit0.q_true)

print(df_results)


# Now run 2000 experiments and plot results
plot_results(agents, Bandit(seed = 2025), steps=1000, runs=2000, dpl=dpl, dir=script_dir)