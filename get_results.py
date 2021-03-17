import numpy as np
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