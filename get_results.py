import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

with open("sample_data_1000vertices_10.00radius.pickle", "rb") as f:
    data = pickle.load(f)
    
error_dists = data['path_lengths'][0]
astar_dists = data['path_lengths'][1]

error_covs = data['error_covs'][0]
astar_covs = data['error_covs'][1]

error_dist_mean = np.mean(error_dists)
astar_dist_mean = np.mean(astar_dists)

error_cov_mean = np.mean(error_covs)
astar_cov_mean = np.mean(astar_covs)

print("Mean path lengths: (A*, EPA*): (%.2f, %.2f)" % (astar_dist_mean, error_dist_mean))
print("Difference in path length from A* to EPA*: %.2f%%\n" % ((error_dist_mean - astar_dist_mean) * 100 / astar_dist_mean))

print("Mean error covariance traces: (A*, EPA*): (%.4f, %.4f)" % (np.mean(astar_covs), np.mean(error_covs)))
print("Difference in error covariance trace from A* to EPA*: %.2f%%\n" % ((error_cov_mean - astar_cov_mean) * 100 / astar_cov_mean))

sns.set()
sns.set_style('whitegrid')
sns.set_palette('Set3')

df_length = pd.DataFrame({
    'A*': astar_dists,
    'EPA*': error_dists
})

df_length = pd.melt(df_length)

fig = plt.figure()
sns.boxplot(x='variable', y='value', data=df_length)
sns.stripplot(x='variable', y='value', data=df_length, jitter=True, color='black')
plt.xlabel(' ')
plt.ylabel('path lengths')
plt.savefig('fig/boxplot_dist.png')
plt.show()

df_covs = pd.DataFrame({
    'A*': astar_covs,
    'EPA*': error_covs
})

df_covs = pd.melt(df_covs)

fig = plt.figure()
sns.boxplot(x='variable', y='value', data=df_covs)
sns.stripplot(x='variable', y='value', data=df_covs, jitter=True, color='black')
plt.xlabel(' ')
plt.ylabel('error covariance traces')
plt.savefig('fig/boxplot_cov.png')
plt.show()