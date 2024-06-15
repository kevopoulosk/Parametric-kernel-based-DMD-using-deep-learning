### This file generates the parametric samples for the heat problem.
### The samples are collcted from a latin hypercube


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import os


def LatinHypercube(dim_sample, low_bounds, upp_bounds, num_samples):
    """
    Function that is used to sample the parameters from a latin hypercube.
    Later, the active learning/adaptive sampling technique will be used instead.
    :param dim_sample: The dimension that we sample
    :param low_bounds: lower bound of the sampling interval
    :param upp_bounds: upper bound of the sampling interval
    :param num_samples: number of desired samples
    :return:
    """
    sampler = qmc.LatinHypercube(d=dim_sample)
    sample = sampler.random(n=num_samples)

    l_bounds = low_bounds
    u_bounds = upp_bounds
    sample_params = qmc.scale(sample, l_bounds, u_bounds)
    return sample_params


### Generate the parametric samples for the heat problem.
directory_data = '/Users/konstantinoskevopoulos/Documents/Heat_Eq_Thesis/SnapshotData/'
directory_samples = '/Users/konstantinoskevopoulos/Documents/Heat_Eq_Thesis/Parameter_Samples/'


samples_heat_eq = LatinHypercube(dim_sample=2, low_bounds=[0.5, 5], upp_bounds=[1, 10], num_samples=300)

filename_train = 'samples_heat_eq.txt'

file_path_data = os.path.join(directory_samples, filename_train)

np.savetxt(file_path_data, samples_heat_eq)

for i in range(samples_heat_eq.shape[0]):
    folder_name = f'mu{i}'

    # Construct the full path to the new folder
    new_folder_path = os.path.join(directory_data, folder_name)

    # Check if the directory already exists
    if not os.path.exists(new_folder_path):
        # Create the directory if it doesn't exist
        os.makedirs(new_folder_path)
        print(f"Folder '{folder_name}' created successfully.")
    else:
        print(f"Folder '{folder_name}' already exists.")


plt.plot(samples_heat_eq[:, 0], samples_heat_eq[:, 1], 'o')
plt.xlabel(r"$\mu_1$")
plt.ylabel(r"$\mu_2$")
plt.title("Parameter space of the heat equation")
plt.show()

# # run the FreeFem file for data generation
# os.system("FreeFem++ Heat_problem_thesis.edp -v 0")

