# write a script that reads two csv files and computes the element-wise ratio of the two files
import numpy as np

# read the two csv files as numpy arrays (since they're headerless)
arr1 = np.loadtxt('results/run_2025422_22929/cross_label_distribution/cross_label_dist_00_round_4.csv', delimiter=',')
arr2 = np.loadtxt('results/run_2025422_22929/cross_label_distribution/cross_label_dist_10_round_4.csv', delimiter=',')

# print the dimensions of the two arrays
print(f"Shape of first array: {arr1.shape}")
print(f"Shape of second array: {arr2.shape}")

# compute the element-wise ratio
# add a small epsilon to avoid division by zero
epsilon = 1e-10
ratio = arr1 / (arr2 + epsilon)

# save the ratio to a new csv file
output_path = 'results/run_2025422_22929/cross_label_distribution/ratio_round_4.csv'
np.savetxt(output_path, ratio, delimiter=',')
print(f"Ratio saved to: {output_path}")


