# Adapted from example code in Chapter 14 of:
# Machine Learning for Business Analytics: Concepts, Techniques, and Applications
# in Python (Second Edition), Shmueli et al. (© 2025 John Wiley & Sons, Inc.)
# Modified for instructional use.

import mlba
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
from contextualbandits.online import LinUCB
from sklearn.linear_model import LogisticRegression
import os

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Switch to show or save files
dpl = False  # True = display plots, False = save plots

def load_and_process_movies():
    movies = mlba.load_data('MovieLensMovies.csv.gz')
    # convert |-separated genres into individual columns
    genres = movies['genres'].str.split('|', expand=True).stack()
    genres = pd.get_dummies(genres.reset_index(level=1, drop=True))
    genres = genres.groupby(level=0).sum().astype(int)
    return pd.concat([movies, genres], axis=1).drop(columns=['genres', '(no genres listed)'])

# load and combine ratings with movie information
movies = load_and_process_movies()
ratings = mlba.load_data('MovieLensRatings.csv.gz')
ratings['reward'] = [1 if rating >= 4.5 else 0 for rating in ratings['rating']]
ratings = ratings.sort_values('timestamp')
all_movies = ratings.merge(movies, on='movieId')

# find the top-50 most frequently rated movies and subset all_movies
top_50 = set(all_movies.value_counts('movieId').head(50).index)
top_50_movies = all_movies[all_movies['movieId'].isin(top_50)]

# create profile of genres for each movie in the top-50 (arm_features)
arm_features = (top_50_movies.drop(columns=['userId', 'reward', 'rating', 'timestamp', 'title'])
                .drop_duplicates(subset='movieId'))
arm_features = arm_features.set_index('movieId')

# for each user, create their profile of genre preferences based on
# their viewed movies that are not in the top-50 (user_features)
user_features = (all_movies[~all_movies['movieId'].isin(top_50)]
                 .drop(columns=['movieId', 'reward', 'rating', 'timestamp', 'title'])
                 .groupby('userId').sum())  # sum genres across movies
user_features = user_features.div(user_features.sum(axis=1), axis=0)  # normalize

# keep only users who rated top-50 movies
top_50_raters = top_50_movies['userId'].unique()
user_features = user_features[user_features.index.isin(top_50_raters)]

# create the observed reactions for each user-movie pair
reactions = all_movies[['userId', 'movieId', 'reward']]
reactions = reactions[reactions['movieId'].isin(top_50)]
reactions = reactions[reactions['userId'].isin(top_50_raters)]

print(arm_features)
print(user_features)

random.seed(1234)

nchoices = arm_features.shape[1]
base_algorithm = LogisticRegression(solver='lbfgs')
bandit = LinUCB(nchoices=nchoices, random_state=5555)

def get_rewards(actions, arms, arm_features):
    """ determine the rewards for the taken actions (action matches the genre of a movie) """
    rewards = []
    for arm, action in zip(arms, actions, strict=True):
        features = arm_features.loc[arm]
        rewards.append(features.iloc[action])
    return rewards

# we process the data in batches of 50 records and refit the model after each batch
batch_size = 50

# initialize the simulation with a random selection of actions
batch = reactions.iloc[:batch_size, :]
userIds = batch['userId']
actions = random.choices(range(nchoices), k=batch_size)
rewards = get_rewards(actions, batch['movieId'], arm_features)
bandit.fit(X=user_features.loc[userIds], a=np.array(actions), r=np.array(rewards))

# collect results
results = {
    'random': {'actions': list(actions), 'rewards': list(rewards)},
    'bandit': {'actions': list(actions), 'rewards': list(rewards)},
}

# iterate through the rest of the data in batches
start = batch_size
while start < reactions.shape[0]:
    end = min(start + batch_size, reactions.shape[0])
    batch = reactions.iloc[start:end, :]
    batch_userIds = reactions.iloc[start:end]['userId']
    # obtain the actions for this batch, using the previously trained model and determine rewards
    batch_actions = bandit.predict(X=user_features.loc[batch_userIds])
    batch_rewards = get_rewards(batch_actions, batch['movieId'], arm_features)

    # feed these back to the algorithm
    results['bandit']['actions'].extend(batch_actions)
    results['bandit']['rewards'].extend(batch_rewards)
    userIds = reactions.iloc[:end]['userId']
    bandit.fit(X=user_features.loc[userIds],
               a=np.array(results['bandit']['actions']),
               r=np.array(results['bandit']['rewards']))

    # collect random actions and rewards for comparison
    random_actions = np.array(random.choices(range(nchoices), k=len(batch['movieId'])))
    random_rewards = get_rewards(random_actions, batch['movieId'], arm_features)
    results['random']['actions'].extend(random_actions)
    results['random']['rewards'].extend(random_rewards)
    start = end

def get_mean_reward(rewards):
    return np.cumsum(rewards) / np.arange(1, len(rewards)+1)

fig, ax = plt.subplots()
df = pd.DataFrame({
    'LinUCB': get_mean_reward(results['bandit']['rewards']),
    'Random': get_mean_reward(results['random']['rewards']),
})
df.plot(ax=ax)
ax.set_xlabel('Number of records seen')
ax.set_ylabel('Cumulative reward rate')
plt.tight_layout()
if dpl:
    plt.show()
else:
    plt.savefig(os.path.join(script_dir, 'linucb_movie_recommendation.png'))
    plt.close()

#####################################################################################
# Some interpretation 

# Map action index -> genre name (columns of arm_features are genres)
genre_names = list(arm_features.columns)

# 1) What genres does the learned policy choose overall (across records)?
bandit_action_counts = (
    pd.Series(results['bandit']['actions'], name="action_idx")
      .value_counts()
      .sort_index()
)
bandit_action_counts.index = [genre_names[i] for i in bandit_action_counts.index]
bandit_action_share = (bandit_action_counts / bandit_action_counts.sum()).rename("share").to_frame()

random_action_counts = (
    pd.Series(results['random']['actions'], name="action_idx")
      .value_counts()
      .sort_index()
)
random_action_counts.index = [genre_names[i] for i in random_action_counts.index]
random_action_share = (random_action_counts / random_action_counts.sum()).rename("share").to_frame()

policy_action_summary = bandit_action_share.join(
    random_action_share, how="outer", lsuffix="_LinUCB", rsuffix="_Random"
).fillna(0).sort_values("share_LinUCB", ascending=False)

print("\n=== Genre choices: LinUCB vs Random (share of records) ===")
print(policy_action_summary)

# 2) Expected reward by chosen genre (empirical, from realized stream)
#    (This is just the average of realized rewards among records where a genre was chosen.)
bandit_rewards = np.asarray(results['bandit']['rewards'], dtype=float)
bandit_actions = np.asarray(results['bandit']['actions'], dtype=int)

avg_reward_by_genre = (
    pd.DataFrame({"genre": [genre_names[i] for i in bandit_actions],
                  "reward": bandit_rewards})
      .groupby("genre")["reward"]
      .mean()
      .sort_values(ascending=False)
      .rename("avg_reward")
      .to_frame()
)

print("\n=== Average realized reward by chosen genre (LinUCB stream) ===")
print(avg_reward_by_genre)


# 3) Policy at the USER level: what genre would LinUCB choose for each user context?
#    (One prediction per user)
X_users = user_features.values
user_level_actions = bandit.predict(X_users)

user_policy = (
    pd.Series(user_level_actions, index=user_features.index, name="action_idx")
      .map(lambda i: genre_names[i])
      .rename("chosen_genre")
      .to_frame()
)

print("\n=== Most common chosen genre at USER level ===")
print(user_policy["chosen_genre"].value_counts())

# Look at a few users: their top genres vs chosen genre
top_k = 5
user_top_genres = (
    user_features.apply(lambda row: list(row.sort_values(ascending=False).head(top_k).index), axis=1)
                 .rename("top_genres")
                 .to_frame()
)
user_inspect = user_policy.join(user_top_genres)
print("\n=== Sample users: chosen genre vs their top-5 historical genres ===")
print(user_inspect.sample(10, random_state=0))

# 4) Model coefficients for each genre (action)
ovr = bandit._oracles
nchoices = bandit.nchoices

# per-arm models
models = [ovr.algos[a].model for a in range(bandit.nchoices)]

# stack coefs (nchoices x 20)
coef = np.vstack([m.coef_.ravel() for m in models])

# name rows (actions) and columns (features + intercept)
action_names = getattr(bandit, "choice_names", None)
if action_names is None or len(action_names) != nchoices:
    action_names = list(arm_features.columns)   # in your code: genres/actions

feat_names = list(user_features.columns) + ["intercept"]

theta_df = pd.DataFrame(coef, index=action_names, columns=feat_names)
print("\n=== Model coefficients (theta) for each genre (action) ===")
print(theta_df)

# Visualize the coefficients as a heatmap
theta_viz = theta_df.drop(columns=['intercept'])
feat_viz_names = theta_viz.columns

fig, ax = plt.subplots(figsize=(10, 6))
cax = ax.matshow(theta_viz, cmap='coolwarm')
fig.colorbar(cax)
ax.set_xticks(np.arange(len(feat_viz_names)))
ax.set_yticks(np.arange(len(action_names)))
ax.set_xticklabels(feat_viz_names, rotation=90)
ax.set_yticklabels(action_names)
ax.set_title('LinUCB Model Coefficients by Genre and Feature')
plt.tight_layout()
if dpl:
    plt.show()
else:
    plt.savefig(os.path.join(script_dir, 'linucb_movie_coefficients_heatmap.png'))
    plt.close()

# 5) Sensitivity analysis: how does changing each user feature affect the action scores?
X = user_features.values
baseline = bandit.decision_function(X).mean(axis=0)

eps = 0.05
sens = {}

for j, feat in enumerate(user_features.columns):
    Xp = X.copy()
    Xp[:, j] += eps
    delta = bandit.decision_function(Xp).mean(axis=0) - baseline
    sens[feat] = delta

sens_df = pd.DataFrame(sens, index=action_names)

for a in action_names:
    print(f"\n=== Action: {a} ===")
    print(sens_df.loc[a].sort_values(ascending=False))

# Visualize the sensitivity as a heatmap
fig, ax = plt.subplots(figsize=(10, 6))
cax = ax.matshow(sens_df, cmap='bwr')
fig.colorbar(cax)
ax.set_xticks(np.arange(len(user_features.columns)))
ax.set_yticks(np.arange(len(action_names)))
ax.set_xticklabels(user_features.columns, rotation=90)
ax.set_yticklabels(action_names)
ax.set_title('LinUCB Sensitivity Analysis by Genre and Feature')
plt.tight_layout()
if dpl:
    plt.show()
else:
    plt.savefig(os.path.join(script_dir, 'linucb_movie_sensitivity_heatmap.png'))
    plt.close()
