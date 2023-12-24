import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


def NN_Regression(velocities, coefficient_matrix):
    # Sort the velocities and the rows of the coefficient matrix accordingly.
    # This is done, in order for the GPR to be applied correctly
    # Create a list of indices that represents the sorting order
    sorting_indices = np.argsort(velocities)
    # Use the sorting indices to reorder the rows of the matrix
    coefficient_matrix = coefficient_matrix[sorting_indices]
    velocities = np.array(sorted(velocities))
    velocities = velocities.reshape(-1, 1)
    # NOTE: The x_train and y_train must correspond to each other.
    # The randomly sampled velocities for X_train should be the same velocities
    # (same rows sampled from the coefficient matrix)
    # 75% training set and 25% test set.
    X_train, X_test, y_train, y_test = train_test_split(velocities, coefficient_matrix, test_size=0.25, random_state=1)

    # print(f"the y_train(coefficient matrix) is {y_train}, with shape {y_train.shape}")
    # print(f"the X_train(velocity samples) is {X_train}, with shape {X_train.shape}")
    regr = MLPRegressor(activation='relu')
    regr.fit(X_train, y_train)
    result = regr.predict(velocities)
    # print(f"the prediction from the NN regression is {result}, with shape {result.shape}")
    # print(f"the score is {regr.score(X_test, y_test)}")

    L2_error = []
    prediction = np.zeros((240, 11))
    # calculate the L2 error for each one of the dominant bases (dimensions)
    for i in range(coefficient_matrix.shape[1]):
        prediction[:, i] = result[:, i]
        error = np.sqrt(np.sum(coefficient_matrix[:, i] - prediction[:, i]) ** 2) / np.sum(
            coefficient_matrix[:, i]) ** 2
        L2_error.append(error)
    # Compute the L2 error for each base i.e for each one of the 11 bases.
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

# reduced basis coefficients
output = np.load(directory_rom + '/ReducedFlow_' + attribute + '.npy')
velocity_samples = read_velocities(directory_vel)
prediction, mean, std = NN_Regression(velocities=velocity_samples, coefficient_matrix=output)
print(f"The mean of the prediction error is {mean} with st.dev {std}")
