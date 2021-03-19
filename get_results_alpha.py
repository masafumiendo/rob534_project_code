import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

with open("sample_data_alpha_1000vertices_10.00radius.pickle", "rb") as f:
    data = pickle.load(f)

error_dists10 = data['path_lengths'][0]
error_dists100 = data['path_lengths'][1]
error_dists1000 = data['path_lengths'][2]
astar_dists = data['path_lengths'][3]

error_covs10 = data['error_covs'][0]
error_covs100 = data['error_covs'][1]
error_covs1000 = data['error_covs'][2]
astar_covs = data['error_covs'][3]

error_dist_mean10 = np.mean(error_dists10)
error_dist_mean100 = np.mean(error_dists100)
error_dist_mean1000 = np.mean(error_dists1000)
astar_dist_mean = np.mean(astar_dists)

error_cov_mean10 = np.mean(error_covs10)
error_cov_mean100 = np.mean(error_covs100)
error_cov_mean1000 = np.mean(error_covs1000)
astar_cov_mean = np.mean(astar_covs)

print("Mean path lengths: (A*, EPA*): (%.2f, %.2f)" % (astar_dist_mean, error_dist_mean10))
print("Difference in path length from A* to EPA*: %.2f%%\n" % ((error_dist_mean10 - astar_dist_mean) * 100 / astar_dist_mean))

print("Mean error covariance traces: (A*, EPA*): (%.4f, %.4f)" % (np.mean(astar_covs), np.mean(error_covs10)))
print("Difference in error covariance trace from A* to EPA*: %.2f%%\n" % ((error_cov_mean10 - astar_cov_mean) * 100 / astar_cov_mean))

sns.set()
sns.set_style('whitegrid')
sns.set_palette('Set3')

df_length = pd.DataFrame({
    'A*': astar_dists,
    'alpha=10': error_dists10,
    'alpha=100': error_dists100,
    'alpha=1000': error_dists1000,
})

df_length = pd.melt(df_length)

fig = plt.figure()
sns.boxplot(x='variable', y='value', data=df_length)
sns.stripplot(x='variable', y='value', data=df_length, jitter=True, color='black')
plt.xlabel(' ')
plt.ylabel('path lengths (m)')
plt.savefig('fig/boxplot_dist.png')
plt.show()

df_covs = pd.DataFrame({
    'A*': astar_covs,
    'alpha=10': error_covs10,
    'alpha=100': error_covs100,
    'alpha=1000': error_covs1000,
})

df_covs = pd.melt(df_covs)

fig = plt.figure()
sns.boxplot(x='variable', y='value', data=df_covs)
sns.stripplot(x='variable', y='value', data=df_covs, jitter=True, color='black')
plt.xlabel(' ')
plt.ylabel('error covariance traces (m^2)')
plt.savefig('fig/boxplot_cov.png')
plt.show()