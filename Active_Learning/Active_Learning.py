import numpy as np
import matplotlib.pyplot as plt


def Failure(residual, threshold):
    return np.abs(residual) - threshold


def Failure_Prob(num_fail, N_train):
    return num_fail / N_train


# def (X):
#
#     scaledX = Scale(X)
#     Xperm, _ = Permute(X)


def GenerateSamples(failure_samples, num_new_samples, N_choose, train_dictionary):
    # Maybe only select a some of the samples failure for the new samples generation
    # Or choose this adaptively according to some annealing technique
    fail_samples = N_choose * len(failure_samples)








def FI_Sampling(pLANDO, iterations, L2_error, NN_error_threshold, epsilon_tol):
    for i in range(iterations):
        # Offline + Online_phase
        mu_samples = pLANDO.offline()  # TODO: Fix these to return the correct values
        errors = pLANDO.online()

        if L2_error < np.mean(errors):
            break

        failure_function = [Failure(error, NN_error_threshold) for error in errors]
        samples_failure = sorted([x for x in failure_function if x > 0], reverse=True)


        failure_prob = Failure_Prob(num_fail=len(samples_failure), N_train=len(errors))
        if failure_prob < epsilon_tol:
            break


        # TODO: Generate additional samples -> adaptive method.
