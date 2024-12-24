### This file generates the parametric samples for the Allen Cahn equation.
### The samples are collcted via Latin Hypercube sampling

from scipy.stats import qmc
import os
from tqdm import tqdm
from Allen_Cahn_equation import *
import time


def LatinHypercube(dim_sample, low_bounds, upp_bounds, num_samples):
    """
    Function that is used to sample the parameters from a latin hypercube.
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


def GenerateSamples(num_samples, plot=False):

    directory_samples = '/Users/konstantinoskevopoulos/Documents/Allen_Cahn_Thesis/Parameter_Samples/'
    directory = '/Users/konstantinoskevopoulos/Documents/Allen_Cahn_Thesis/'

    os.makedirs(directory_samples, exist_ok=True)

    directory_data = directory + 'SnapshotData/'
    os.makedirs(directory_data, exist_ok=True)

    samples = LatinHypercube(dim_sample=2, low_bounds=[0.0001, 0.5], upp_bounds=[0.001, 4], num_samples=num_samples)

    filename_train = 'samples_allen_cahn.txt'

    file_path_data = os.path.join(directory_samples, filename_train)

    np.savetxt(file_path_data, samples)

    pbar = tqdm(total=samples.shape[0], desc="Data Generation...")
    exec_times = []
    for i in range(samples.shape[0]):
        folder_name = f'mu{i}'

        # Construct the full path to the new folder
        new_folder_path = os.path.join(directory_data, folder_name)

        start_time = time.time()
        X = Allen_Cahn_eq(D=samples[i][0], a=samples[i][1])
        end_time = time.time()
        exec_time = end_time - start_time
        exec_times.append(exec_time)
        np.save(directory_data+f"sample{i}", X)
        pbar.update()
    pbar.close()
    print(f"Mean execution time per sample: {np.mean(exec_times)}")

    if plot:
        plt.plot(samples[:, 0], samples[:, 1], 'o')
        plt.xlabel(r"$\mu_1= D$")
        plt.ylabel(r"$\mu_2=\alpha$")
        plt.title("Parameter space of the Allen-Cahn equation")
        plt.show()


GenerateSamples(num_samples=1000)



