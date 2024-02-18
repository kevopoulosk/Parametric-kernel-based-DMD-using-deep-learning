import numpy as np
import torch
from torch import nn
from Sparse_Dictionary_Learning import *
from scipy.stats import qmc
import itertools
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data as data
from scipy import interpolate
from sklearn import preprocessing


class BranchNet(nn.Module):

    def __init__(self, num_input, depth, width):
        super().__init__()
        # Number of hidden layers of the network
        self.Depth = depth
        # Number of nodes in each hidden layer
        self.Width = width
        # Number of input nodes for the network
        self.NumInput = num_input

        layers = []
        # Start with the input layer
        layers.append(nn.Linear(in_features=self.NumInput, out_features=self.Width))
        layers.append(nn.ReLU())

        for i in range(self.Depth):
            # Add hidden layers
            layers.append(nn.Linear(in_features=self.Width, out_features=self.Width))
            # Do not apply ReLU activation to the output layer
            if i < (self.Depth - 1):
                layers.append(nn.ReLU())

        self.fnn_stack = nn.Sequential(*layers)

    def forward(self, x):
        fnn_output = self.fnn_stack(x)

        return fnn_output


class TrunkNet(nn.Module):

    def __init__(self, num_input, depth, width, features_len=0):
        super().__init__()
        # Number of hidden layers of the network
        self.Depth = depth
        # Number of nodes in each hidden layer
        self.Width = width
        # Number of input nodes for the network
        self.NumInput = num_input
        # Whether or not to perform a feature expansion (for physics-informed network)
        self.features_len = features_len

        layers = []
        if self.features_len > 0:
            # Input layer with feature expansion
            layers.append(nn.Linear(in_features=self.features_len, out_features=self.Width, bias=False))
            layers.append(nn.ReLU())
        else:
            # Input layer without feature expansion
            layers.append(nn.Linear(in_features=self.NumInput, out_features=self.Width, bias=False))
            layers.append(nn.ReLU())

        for i in range(self.Depth):
            # Hidden layers
            layers.append(nn.Linear(in_features=self.Width, out_features=self.Width, bias=False))
            # Apply ReLU to the output layer for the trunk net
            layers.append(nn.ReLU())

        self.fnn_stack = nn.Sequential(*layers)

    def forward(self, x):
        fnn_output = self.fnn_stack(x)

        return fnn_output


class MIONet(nn.Module):

    def __init__(self, num_branch, kernel_vector_input, trunk_input, depth, width, feature_expansion=False,
                 num_features=0):
        super().__init__()
        # Number of branch networks
        self.NumBranch = num_branch
        # Number of input nodes in each branch net. It is the number of m_sensors.
        self.InputBranchNodes = kernel_vector_input
        if feature_expansion:
            # Change the number of input nodes in the trunk network if feature expansion is performed
            self.InputTrunkNodes = num_features
        else:
            # Number of input nodes in the trunk net. Can be 1 or 2
            self.InputTrunkNodes = trunk_input
        # Depth of networks. The trunk and all the branch networks have the same depth and width.
        # Refers to the number of hidden layers
        self.Depth = depth
        # Width of networks. Number of nodes in each layer
        self.Width = width

        # Initialize the branch network
        self.branch_nets = [BranchNet(num_input=self.InputBranchNodes, depth=self.Depth, width=self.Width)
                            for _ in range(self.NumBranch)]

        # Initialize the trunk network
        self.trunk_net = TrunkNet(num_input=self.InputTrunkNodes, depth=self.Depth, width=self.Width,
                                  features_len=num_features)
        # Add bias
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))

    def forward(self, x_train):

        # Define the input of trunk net
        trunk_input = x_train[:, -1][:, None]

        # Feature expansion option
        if self.InputTrunkNodes > 2:
            features = [trunk_input, torch.sin(trunk_input), torch.cos(trunk_input)]

            trunk_input = torch.cat(features, dim=1)

        # Define the inputs of all branch nets
        branch_inputs = []
        # For all the branch nets, define their inputs by decomposing each column of the X_train matrix.
        # The columns denote the number of features. So I decompose them to pass the training data in the networks
        for i in range(0, x_train.size(1) - 1, self.InputBranchNodes):
            branch_inputs.append(x_train[:, i:(i + self.InputBranchNodes)])

        # Compute the branch output for the given inputs
        branch_outputs = [branch_net(branch_input) for branch_net, branch_input in zip(self.branch_nets, branch_inputs)]
        # Compute the trunk output for the given input
        trunk_output = self.trunk_net(trunk_input)

        # Now element-wise multiplication is performed between all the outputs of the branch nets

        element_wise_branch = branch_outputs[0]  # Initialize result with the first tensor
        for tensor in branch_outputs[1:]:
            element_wise_branch = torch.mul(element_wise_branch, tensor)

        # Next, perform element-wise multiplication between the output of the trunk net
        # and the previously computed element-wise multiplied branch outputs
        element_wise_total = torch.mul(element_wise_branch, trunk_output)

        # Finally, sum all the elements of the "element_wise_total" tensor
        # The resulting quantity is the point prediction of MIONet
        # Point prediction is G_k_y: G is the operator that MIONet learns,
        # k is the kernel matrix for the specific "mu" value
        # y is the location (point) where the prediction is evaluated.

        G_k_y = torch.sum(element_wise_total, dim=1)[:, None]
        # Add bias
        G_k_y += self.b

        return G_k_y


class Data(data.Dataset):
    # Class that preprocesses the data that go into the MIONet architecture.
    # This class is needed for the Dataloader.
    def __init__(self, m_sensors, T_end, kernels, models):
        # number of sensors -> discretisation of the input functions
        self.m_sensors = m_sensors
        # time horizon of the simulation
        self.T_end = T_end
        # "n" kernel matrices for "n" parametric samples
        self.kernels = kernels
        # input of trunk network, encoding the locations where the output is evaluated
        self.trunk_input = torch.tensor(np.linspace(0, T_end, self.m_sensors), dtype=torch.float32)

        # cartesian product form of the data. Each kernel input is evaluated in all the points of the trunk input.
        # e.g. dataset of 10000 is comprised by 100 inputs and 100 points in the trunk net.
        cartesian_prod = list(itertools.product(self.kernels, self.trunk_input))
        # make the data in matrix form with dimensions (datapoints, number of features)
        self.X = np.vstack([np.hstack((matrix.flatten(), value)) for matrix, value in cartesian_prod])
        self.X = torch.tensor(self.X, dtype=torch.float32)

        # compute the ground truth y_true, denoted as G_k_y -> G(k)(y)
        self.G_k_y = []
        pbar = tqdm(total=len(self.kernels), position=0, leave=True, desc="Progress of Data Preprocessing...")
        for i in range(len(self.kernels)):
            for t in range(len(self.trunk_input)):
                self.G_k_y.append(models[i][0][t])
            pbar.update()
        pbar.close()
        self.y = np.array(self.G_k_y).reshape(-1, 1)

        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_point_x = self.X[idx]
        data_point_y = self.y[idx]
        return data_point_x, data_point_y


def LatinHypercube(NumSamples):
    """
    Function used in order to take parametric samples with the Latin Hypercube method
    :param NumSamples:
    :return:
    """
    sampler = qmc.LatinHypercube(d=4)
    sample = sampler.random(n=NumSamples)

    l_bounds = [0.01, 0.001, 0.02, 0.0015]
    u_bounds = [0.14, 0.0025, 0.3, 0.0025]
    sample_params = qmc.scale(sample, l_bounds, u_bounds)

    return sample_params


def LANDO(kernel_choice, T_end, tol, num_samples, train_test):
    """
    Function that generates the data from kernel DMD
    :param kernel_choice:
    :param T_end:
    :param tol:
    :param num_samples:
    :param train_test:
    :return:
    """
    param_samples = LatinHypercube(num_samples)

    sparse_dicts_all = []
    scaled_X_all = []
    X_perm_all = []
    lando_data = []
    Y_vals = []
    Y_perm_all = []
    X_all = []
    pbar = tqdm(total=num_samples, desc=f"Generation of {train_test} data...")
    # Generate the sparse dictionaries for all the parametric samples
    for val, param_sample in enumerate(param_samples):
        X, _ = Lotka_Volterra_Snapshot(params=param_sample, T=T_end)
        Y = Lotka_Volterra_Deriv(X, *param_sample)

        scaledX = Scale(X)

        Xperm, perm = Permute(X)
        Yperm = Y[:, perm]

        SparseDict, SnapMat, C = SparseDictionary(Xperm, scaledX, kernel_choice, tolerance=tol, pbar_bool=False)
        sparse_dicts_all.append(SparseDict)

        lando_data.append([SparseDict, SnapMat, C])
        scaled_X_all.append(scaledX)
        X_perm_all.append(Xperm)
        Y_perm_all.append(Yperm)
        X_all.append(X)
        Y_vals.append(Y)
        pbar.update()
    pbar.close()

    # Ensure that all the kernels have the same number of columns.
    num_cols_all = [lando_data[val][0].shape[1] for val in range(len(lando_data))]
    max_cols = np.max(num_cols_all)
    SparseDicts_Consistent = ColumnConsistency(sparse_dicts_all, lando_data, kernel_choice, max_cols)

    # Compute W tilde and form the model for all the kernels .
    W_tildes = [y_perm @ np.linalg.pinv(kernel_choice(Sparse_Dict, scaled_X * x_perm))
                for y_perm, Sparse_Dict, scaled_X, x_perm in
                zip(Y_perm_all, SparseDicts_Consistent, scaled_X_all, X_perm_all)]

    kernels = [kernel_choice(Sparse_Dict, scaled_X * X_mat)
               for Sparse_Dict, scaled_X, X_mat in zip(SparseDicts_Consistent, scaled_X_all, X_all)]

    models = [W_tilde_mat @ kernel_mat for W_tilde_mat, kernel_mat in zip(W_tildes, kernels)]

    # sanity check -> compute reconstruction error to make sure its small
    reconstruction_relative_errors = [abs(Y_vals[i][0] - models[i][0]) / Y_vals[i][0] for i in range(len(Y_vals))]
    mean_reconstruction_relative_errors = [np.mean(list_err) for list_err in reconstruction_relative_errors]

    print(f"The mean relative reconstruction errors are: {mean_reconstruction_relative_errors}")

    return kernels, models, W_tildes


def ColumnConsistency(sparse_dicts_all, lando_data, kernel_choice, max_cols):
    """
    Function used to make sure that all the kernel matrices in the dataset have the same number of columns
    This function adds columns to the dictionaries if needed, but does so in an optimal way, augmenting the current
    dictionary wit the least linearly dependent sample
    :param sparse_dicts_all:
    :param lando_data:
    :param kernel_choice:
    :param max_cols:
    :return:
    """
    pbar = tqdm(total=len(sparse_dicts_all),
                desc="Ensuring data consistency -> same columns for all kernel matrices...")
    # Iterate through all the sparse dictionaries
    for i, mat in enumerate(sparse_dicts_all):
        if mat.shape[1] == max_cols:
            pass
        else:

            sparse_dict = lando_data[i][0]
            SnapMat = lando_data[i][1]
            C = lando_data[i][2]
            # This iteration deals with the case when the difference between col and max_col is more than 1
            # e.g. If max_col = 6 and col = 4, then this sparse dictionary needs to be "augmented" with 2 snapshots.
            m = C.shape[0]
            for _ in range(max_cols - mat.shape[1]):

                delta_vals_modif = np.zeros(SnapMat.shape[1])
                # Iterate through the snapshots of X, to find the snapshot that is the least linearly dependent with the
                # current version of the dictionary. Then update the dictionary
                k_tilde_prev_list = []
                ktt_list = []
                for j in range(1, SnapMat.shape[1]):
                    CandidateSample = SnapMat[:, j].reshape(-1, 1)
                    k_tilde_prev = kernel_choice(sparse_dict, CandidateSample).reshape(-1, 1)
                    # calculate pi with two back substitutions
                    y = np.linalg.solve(C, k_tilde_prev)
                    pi = np.linalg.solve(C.T, y)

                    k_tt = kernel_choice(CandidateSample, CandidateSample)
                    delta = k_tt - k_tilde_prev.T @ pi
                    delta_vals_modif[j] = delta[0]
                    k_tilde_prev_list.append(k_tilde_prev)
                    ktt_list.append(k_tt)

                least_linearly_dep = np.argmax(delta_vals_modif)
                k_tilde_prev_chosen = k_tilde_prev_list[least_linearly_dep]
                k_tt_chosen = ktt_list[least_linearly_dep]
                sparse_dict = np.concatenate((sparse_dict, SnapMat[:, least_linearly_dep].reshape(-1, 1)), axis=1)

                st = np.linalg.solve(C, k_tilde_prev_chosen)
                # Check for ill-conditioning
                if k_tt_chosen <= np.linalg.norm(st) ** 2:
                    print('The Cholesky factor is ill-conditioned.\n '
                          'Perhaps increase the sparsity parameter (tolerance) or change the kernel hyperparameters.')
                # Update the Cholesky factor
                C = np.vstack([C, st.T])
                new_column_vals = np.concatenate(
                    [np.zeros((m, 1)), np.maximum(np.abs(np.sqrt(k_tt_chosen - np.linalg.norm(st) ** 2)), 0)])

                C = np.hstack([C, new_column_vals.reshape(-1, 1)])
                m += 1

            sparse_dicts_all[i] = sparse_dict
        pbar.update()
    pbar.close()

    return sparse_dicts_all
