import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel


def GP_Regression(velocities, coefficient_matrix):
    # Sort the velocities and the rows of the coefficient matrix accordingly.
    # This is done, in order for the GPR to be applied correctly
    # Create a list of indices that represents the sorting order
    sorting_indices = np.argsort(velocities)
    # Use the sorting indices to reorder the rows of the matrix
    coefficient_matrix = coefficient_matrix[sorting_indices]
    velocities = np.array(sorted(velocities))
    velocities = velocities.reshape(-1, 1)
    training_ratio = 0.75
    sampled_velocities = coefficient_matrix.shape[0]
    rng = np.random.RandomState(7)
    # randomly choose some training indices
    training_indices = rng.choice(np.arange(velocities.size), size=int(training_ratio * sampled_velocities),
                                  replace=False)

    # built different regression for each output dimension. This means that we will have one regression per base.
    # As a result, we have single input-output cases
    X_train = velocities[training_indices]

    kernel = 1 * Matern(length_scale=0.2, nu=2.5) + 1 * WhiteKernel(noise_level=30)
    L2_error = []
    prediction = np.zeros((240, 11))
    # for each dimension
    for i in range(coefficient_matrix.shape[1]):
        Y_train = coefficient_matrix[training_indices, i]

        # What kernel should I use? The choice of the kernel plays an important role in the predictive behavior
        gpr = GaussianProcessRegressor(kernel=kernel)
        gpr.fit(X_train, Y_train)
        mean_prediction, std_prediction = gpr.predict(velocities, return_std=True)

        plt.scatter(X_train, Y_train, label="Observations-training samples")
        plt.plot(velocities, mean_prediction, label="Mean prediction")

        plt.fill_between(
            velocities.ravel(),
            mean_prediction - 1.96 * std_prediction,
            mean_prediction + 1.96 * std_prediction,
            alpha=0.5,
            label=r"95% confidence interval",
        )
        plt.legend()
        plt.xlabel("$x$, velocity samples")
        plt.ylabel(f"$f(x)$, RB coefficients of dimension number {i}")
        plt.title(f"Gaussian Process Regression of dimension number/base number {i}")
        plt.show()

        # These are the reduced basis coefficients per dimension for the test cases
        # as they are predicted by the mean of the GPR
        prediction[:, i] = mean_prediction
        error = np.sqrt(np.sum(coefficient_matrix[:, i] - prediction[:, i]) ** 2) / np.sum(coefficient_matrix[:, i]) ** 2
        L2_error.append(error)

    # We performed GPR for each base (dimension of the data).
    # Hence, we calculate the relative error, using the L2 norm for each one of the 11 bases (mean and std of the error)
    prediction_error_mean = np.mean(L2_error)
    prediction_error_std = np.std(L2_error)

    return prediction, prediction_error_mean, prediction_error_std


def read_velocities(directory_vel):
    """
    Function to read the velocities from .txt file to numpy array
    :param directory_vel:
    :return:
    """
    txt_vel = open(directory_vel + "/velocities_sampled.txt", "r")
    # reading the file
    data = txt_vel.read()

    velocities_input = data.strip().split()
    del velocities_input[0]
    txt_vel.close()
    velocities_input = np.array([float(value) for value in velocities_input])
    return velocities_input


directory_vel = "/Users/konstantinoskevopoulos/Documents/SnapshotData/flow_around_cylinder/velocities"
directory_rom = "/Users/konstantinoskevopoulos/Documents/SnapshotData/flow_around_cylinder/ROM"
attribute = "u"

output = np.load(directory_rom + '/ReducedFlow_' + attribute + '.npy')
velocity_samples = read_velocities(directory_vel)
coeffs_pred, mean, std = GP_Regression(velocities=velocity_samples, coefficient_matrix=output)
print(f"The mean of the prediction error is {mean} and the st.dev of the error is  {std}")


