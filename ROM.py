import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from scipy.interpolate import RBFInterpolator



def vertices_num(directory_to_txt):
    TargetMSH = open(directory_to_txt, "r+")

    # Read the total number of vertices(points) from .msh file
    line = TargetMSH.readline()
    words = line.strip().split()
    v_num = int(words[0])
    TargetMSH.close()
    return v_num


def extract_vertices(directory_to_txt):
    TargetMSH = open(directory_to_txt, "r+")

    # Read the total number of vertices(points) from .msh file
    line = TargetMSH.readline()
    words = line.strip().split()
    v_num = int(words[0])
    l_num = int(np.ceil(v_num / 5))
    TargetMSH.close()
    # print('TOTAL VERTICES:',v_num)

    # Create a list and copy out all the boundary vertices
    VerticeList = []

    for i in range(1, l_num + 1):
        with open(directory_to_txt, 'r') as txt:

            text = txt.readlines()
            currentline = text[i]
            coordinates = currentline.strip().split()

            for j in range(len(coordinates)):
                VerticeList.append(np.float64(coordinates[j]))
    return np.expand_dims(np.asarray(VerticeList), axis=1)


def NumComponents(directory, attribute):
    """
    Function to find out how many components do we need for PCA
    :param directory:
    :param attribute:
    :return:
    """
    snapshot_matrix = np.load(directory + "POD" + attribute + ".npy")
    sampled_velocities = snapshot_matrix.shape[1]
    training_ratio = 0.75
    training_samples = int(training_ratio * sampled_velocities)

    # compute and subtract the mean (mean centered data/everything to the center of the mean)
    mean_X = np.expand_dims(np.mean(snapshot_matrix, axis=1), axis=1)
    X_hat = snapshot_matrix - mean_X
    # split in training, validation and test cases
    X_train = X_hat[:, :training_samples]
    pca_all = PCA(svd_solver='full')
    # perform the PCA (it is needed to transpose the X snapshot matrix)
    pca_all.fit(X_train.T)
    all_singular_values = pca_all.singular_values_

    desired_components = 1
    pca = PCA(n_components=desired_components, svd_solver='full')
    pca.fit(X_train.T)
    system_energy = np.sum(pca.explained_variance_ratio_)
    while system_energy < 0.9999:
        pca = PCA(n_components=desired_components, svd_solver='full')
        pca.fit(X_train.T)

        system_energy = np.sum(pca.explained_variance_ratio_)
        if system_energy < 0.9999:
            desired_components += 1
        else:
            pass
    # this is the error of the rank-r SVD approximation.
    rank_r_SVD_error = np.sum(all_singular_values[desired_components:] ** 2) / np.linalg.norm(snapshot_matrix,
                                                                                              ord='fro') ** 2
    POD_projection_error = 1 - np.sum(all_singular_values[:desired_components] ** 2) / np.sum(all_singular_values ** 2)

    return desired_components, rank_r_SVD_error, POD_projection_error


def SVD(directory, attribute, components, error_rank_r, projection_error):
    print(f"Initializing SVD for attribute {attribute}")
    snapshot_matrix = np.load(directory + "POD" + attribute + ".npy")
    sampled_velocities = snapshot_matrix.shape[1]
    # print(f"the snapshot matrix is{snapshot_matrix}")
    training_ratio = 0.75
    training_samples = int(training_ratio * sampled_velocities)

    # compute and subtract the mean (mean centered data/everything to the center of the mean)
    mean_X = np.expand_dims(np.mean(snapshot_matrix, axis=1), axis=1)
    X_hat = snapshot_matrix - mean_X
    # split in training, validation and test cases
    X_train = X_hat[:, :training_samples]
    X_test = X_hat[:, training_samples:]
    pca = PCA(n_components=components, svd_solver='full')
    # perform the PCA (it is needed to transpose the X snapshot matrix)
    pca.fit(X_train.T)  # could also be not the inverse

    system_energy = np.sum(pca.explained_variance_ratio_)

    print(f"SVD completed with {system_energy} % of the system energy explained")
    print(f"The most informative singular values are {pca.singular_values_}")
    print(f"The number of singular values used (truncation threshold) is {components}")
    # plot of the explained variance of the system
    plt.figure()
    plt.plot(pca.explained_variance_ratio_, "-o", label="Singular Values")
    plt.ylabel("Explained variance ratio")
    plt.xlabel("Singular values")
    plt.legend()
    plt.title(f"Explained variance of the system, attribute {attribute}")
    plt.show()
    # plot the singular values
    plt.figure()
    plt.plot(pca.singular_values_, "-o", label='Singular Values')
    plt.ylabel("Value of each singular value")
    plt.xlabel("Singular values")
    plt.legend()
    plt.title(f"Singular values, attribute {attribute}")
    plt.show()
    print(f"SVD for attribute {attribute} finished")
    print(f"components are {pca.components_} with shape {pca.components_.shape}")
    # compute the "alpha" projection coefficient of the dataset
    alpha_train = pca.transform(X_train.T)
    alpha_test = pca.transform(X_test.T)  # dot product of X_test with the POD modes

    # Print the LS relative error of rank-r SVD approximation. This error is related to the fraction of kinetic energy
    # that is missing in the approximation of the snapshot matrix
    print(f"The relative rank-r SVD approximation error for the {attribute} attribute is {error_rank_r * 100}%")
    print(f"The relative error of POD projection for the {attribute} attribute is {projection_error * 100}%\nThis is "
          f"the cumulative energy not captured by the projection")

    reducedbasis_coeff = np.vstack((alpha_train, alpha_test))
    print(f'Training output of {attribute}, shape: {reducedbasis_coeff.shape} \n')

    np.save(directory + '/ReducedFlow_' + attribute + '.npy', reducedbasis_coeff)  # reduced basis coefficients
    np.save(directory + '/Components_' + attribute + '.npy', pca.components_)  # rows of V.T/baserd
    np.save(directory + '/Mean_' + attribute + '.npy', mean_X)  # mean of the snapshot matrix

    # Compute the relative error/relative difference between 2 flow fields, for eachsampled velocity value
    relative_error_flow_fields = []
    for i in range(reducedbasis_coeff.shape[0]):
        X_i = reducedbasis_coeff[i, :].reshape(-1, 1) * pca.components_ + mean_X.T
        error = np.abs(snapshot_matrix[:, i] - X_i) / np.max(snapshot_matrix[:, i])
        relative_error_flow_fields.append(error)

    relative_error_flow_fields = np.column_stack(relative_error_flow_fields)
    print(
        f"the relative error of flow fields is {relative_error_flow_fields}, with type {type(relative_error_flow_fields)} and shape {len(relative_error_flow_fields)}")

    return pca.transform(X_train.T), X_train, alpha_test


def RBF_Interpolate(directory_vel, directory_rom, attribute, true_rb_coeff):
    txt_vel = open(directory_vel + "/velocities_sampled.txt", "r")
    # reading the file
    data = txt_vel.read()

    velocities_input = data.strip().split()
    del velocities_input[0]
    txt_vel.close()
    velocities_input = np.array([float(value) for value in velocities_input]).reshape(-1, 1)

    output = np.load(directory_rom + '/ReducedFlow_' + attribute + '.npy')
    # preprocess/scale the input data such that larger magnitudes do not dominate the interpolation process
    scaler = preprocessing.MinMaxScaler()
    scaled_input = scaler.fit_transform(velocities_input)

    # Actual RBF Interpolation
    rbf_function = RBFInterpolator(scaled_input[:180, :], output[:180, :], kernel='cubic', degree=1)
    prediction = rbf_function(scaled_input[180:, :])

    np.save(directory_rom + '/Reduced_Coeff_prediction_' + attribute + '.npy', prediction)

    # print(f"The true reduced basis coefficients for the last 60 values of velocity are {true_rb_coeff}")
    # print(f"The predicted reduced basis coefficients for the last 60 values of velocity are {prediction}")
    for sample in range(prediction.shape[0]):
        x = np.linspace(true_rb_coeff.min(), true_rb_coeff.max())
        plt.scatter(true_rb_coeff[sample, :], prediction[sample, :])
        plt.plot(x, x, color='red', label="y=x")
        plt.ylabel("predicted coefficients")
        plt.xlabel("expected coefficients")
        plt.title(f"Velocity sample {sample}")
        plt.xlim((-90, 90))
        plt.ylim((-90, 90))
        plt.legend(loc="best")
        # plt.show()


def Reconstruction(directory, attribute):
    # load the alpha coefficients
    alpha = np.load(directory + '/ROM/Reduced_Coeff_prediction_' + attribute + '.npy')
    # load the pca components(modes)
    pca_components = np.load(directory + '/ROM/Components_' + attribute + '.npy')
    # load the mean of the snapshot matrix
    mean = np.load(directory + '/ROM/Mean_' + attribute + '.npy')
    # load the snapshot matrix
    ground_truth = np.load(directory + "/ROM/POD" + attribute + ".npy")
    print(f"ground truth shape {ground_truth.shape}")

    # Set test/validation cases
    StartNumTest = 180
    NumTest = 60
    TestCase = ground_truth[:, StartNumTest:(StartNumTest + NumTest)]

    # Reconstruct the variable field (coefficients*modes)
    Prediction_no_mean = alpha @ pca_components
    # add the previously subtracted mean
    Prediction = Prediction_no_mean.T + mean
    print(f"Prediction shape {Prediction.shape}")

    # Compute the mean and std of the difference between the FOM and the ROM: L2 norm(||u - Phi*alpha||)/L2norm(||u||)
    # Phi denotes the matrix with the orthonormal reduced bases
    # u denotes the ground truth flow field
    # alpha denotes the reduced basis coefficients
    # Compute the difference for each column (for all the flow fields in the snapshot matrix)
    # After that, calculate the mean and std of these errors
    mean_std = []
    for i in range(Prediction.shape[1]):
        error_FOM_ROM = np.sum((TestCase[:, i] - Prediction[:, i])**2)/np.sum(TestCase[:, i]**2)
        mean_std.append(error_FOM_ROM)
    print(f"The mean of the error between the FOM and ROM is {np.mean(mean_std)}, and the error list is {mean_std}\nThis can also be thought as the interpolation error")
    print(f"The std of the error between the FOM and ROM is {np.std(mean_std)}")


    # Next, we calculate the projection error introduced by POD
    # This is: L2norm(u-Phi*Phi.T*u_hat)/L2norm(u)
    # u = the ground truth flow field, Phi = the matrix with the orthonormal reduced bases
    # u_hat = the predicted flow field, as it is constructed above
    # calculate the error for each column of the whole snapshot matrix
    # then print the mean and std of that error
    mean_std_POD = []
    for i in range(Prediction.shape[1]):
        error_POD = np.sum((ground_truth[:, i] - pca_components.T@pca_components@ground_truth[:, i])**2)/np.sum(ground_truth[:, i]**2)
        mean_std_POD.append(error_POD)
    print(
        f"The mean of the error of the POD projection is {np.mean(mean_std_POD)}, and the error list is {mean_std_POD}")
    print(f"The std of the error between the POD projection is {np.std(mean_std_POD)}")

    return Prediction, np.mean(mean_std), np.std(mean_std), np.mean(mean_std_POD), np.std(mean_std_POD)





generated_data_directory = "/Users/konstantinoskevopoulos/Documents/SnapshotData/flow_around_cylinder/"
# number of sampled velocities
num_samples = 240

# Read number of vertices and create empty matrix for data storage
u_num = vertices_num(generated_data_directory + 'u/sample_0_u.txt')
v_num = vertices_num(generated_data_directory + 'v/sample_0_v.txt')
p_num = vertices_num(generated_data_directory + 'p/sample_0_p.txt')

ulist = np.empty((u_num, 0))
vlist = np.empty((v_num, 0))
plist = np.empty((p_num, 0))

# Collect all the snapshots and concatenate them in the "A" matrix
for samplenum in range(num_samples):
    print('Processing sample:', samplenum)
    u = extract_vertices(generated_data_directory + 'u/sample_' + str(samplenum) + '_u.txt')
    v = extract_vertices(generated_data_directory + 'v/sample_' + str(samplenum) + '_v.txt')
    p = extract_vertices(generated_data_directory + 'p/sample_' + str(samplenum) + '_p.txt')

    ulist = np.hstack((ulist, u))
    vlist = np.hstack((vlist, v))
    plist = np.hstack((plist, p))

# Save snapshot matrix (v_num,num_sample)
np.save(generated_data_directory + 'ROM/PODu.npy', ulist)
np.save(generated_data_directory + 'ROM/PODv.npy', vlist)
np.save(generated_data_directory + 'ROM/PODp.npy', plist)
print('Snapshot matrix shape: ', ulist.shape)


def Dim_Reduction(directory, attr):
    num_components, rank_r_error, projection_error = NumComponents(directory=directory, attribute=attr)
    res, train, rb_coeff = SVD(directory=directory, attribute=attr, components=num_components,
                               error_rank_r=rank_r_error, projection_error=projection_error)
    plt.figure(figsize=(10, 7))
    plt.imshow(train)
    plt.colorbar()
    plt.xlabel("Number of snapshots-samples")
    plt.ylabel("Degrees of freedom of the mesh")
    plt.title(f"Collection of snapshots of {attr}")
    plt.show()

    return rb_coeff


rb_coeff_u = Dim_Reduction(directory="/Users/konstantinoskevopoulos/Documents/SnapshotData/flow_around_cylinder/ROM/",
                         attr="u")
rb_coeff_v = Dim_Reduction(directory="/Users/konstantinoskevopoulos/Documents/SnapshotData/flow_around_cylinder/ROM/", attr="v")
rb_coeff_p = Dim_Reduction(directory="/Users/konstantinoskevopoulos/Documents/SnapshotData/flow_around_cylinder/ROM/", attr="p")

RBF_Interpolate(directory_vel="/Users/konstantinoskevopoulos/Documents/SnapshotData/flow_around_cylinder/velocities",
                directory_rom="/Users/konstantinoskevopoulos/Documents/SnapshotData/flow_around_cylinder/ROM",
                attribute="u", true_rb_coeff=rb_coeff_u)

# RBF_Interpolate(directory_vel="/Users/konstantinoskevopoulos/Documents/SnapshotData/flow_around_cylinder/velocities",
#                 directory_rom="/Users/konstantinoskevopoulos/Documents/SnapshotData/flow_around_cylinder/ROM",
#                 attribute="u", true_rb_coeff=rb_coeff_v)
#
# RBF_Interpolate(directory_vel="/Users/konstantinoskevopoulos/Documents/SnapshotData/flow_around_cylinder/velocities",
#                 directory_rom="/Users/konstantinoskevopoulos/Documents/SnapshotData/flow_around_cylinder/ROM",
#                 attribute="u", true_rb_coeff=rb_coeff_p)


Reconstruction(directory="/Users/konstantinoskevopoulos/Documents/SnapshotData/flow_around_cylinder", attribute="u")
# Reconstruction(directory="/Users/konstantinoskevopoulos/Documents/SnapshotData/flow_around_cylinder", attribute="v")
# Reconstruction(directory="/Users/konstantinoskevopoulos/Documents/SnapshotData/flow_around_cylinder", attribute="p")
