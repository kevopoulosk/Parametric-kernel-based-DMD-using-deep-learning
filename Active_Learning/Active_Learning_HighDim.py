from Heat_Equation.HighDim_pLANDO import *
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.stats import qmc
import random


def rel_err(y_test, prediction, tensor=False, mean=True):
    """
    Function that calculates the relative error between two matrices
    :param y_test: The predicted matrix
    :param prediction: The reference matrix
    :param tensor: Boolean: whether or not the data are expressed in tensors
    :param mean: Boolean: whether or not to calculate the mean relative error over the rows of the matrix
    :return:
    """
    err_list = []
    for row in range(y_test.shape[0]):
        if tensor:
            err = np.linalg.norm(y_test[row] - prediction[row].detach().numpy()) / np.linalg.norm(y_test[row])
        else:
            err = np.linalg.norm(y_test[row] - prediction[row]) / np.linalg.norm(y_test[row])

        err_list.append(err)

    if mean:
        return np.mean(err_list)
    else:
        return err_list


class FNN(nn.Module):

    def __init__(self, num_input, num_output, depth, width, activation):
        """
        Class that implements the fully connected neural network
        It is used to learn the mapping from the parameter space --> to x or f(x)
        :param num_input: The number of input nodes (e.g. 4 for Lotka-Volterra model)
        :param num_output: The number of output nodes
        :param depth: number of hidden layers
        :param width: number of nodes in each layer
        :param activation: define the activation function to be used in the network
        """
        super().__init__()

        self.Depth = depth
        self.Width = width
        self.NumInput = num_input
        self.NumOutput = num_output

        if activation == "relu":
            self.Activation = nn.ReLU()

        layers = []

        layers.append(nn.Linear(in_features=self.NumInput, out_features=self.Width))
        layers.append(self.Activation)

        for i in range(self.Depth):
            layers.append(nn.Linear(in_features=self.Width, out_features=self.Width))
            layers.append(self.Activation)

        layers.append(nn.Linear(in_features=self.Width, out_features=self.NumOutput))

        self.fnn_stack = nn.Sequential(*layers)
        self.b = torch.nn.parameter.Parameter(torch.zeros(self.NumOutput))

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize the weights of the network. Several different initialization methods can be chosen
        :return:
        """
        seed = random.randint(0, 2 ** 32 - 1)
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        for module in self.fnn_stack:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, generator=generator)

    def forward(self, x):
        fnn_output = self.fnn_stack(x)
        fnn_output += self.b

        return fnn_output


class Ensemble_NN:

    def __init__(self, num_networks, num_params_varied, dofs, depths, widths, activations, epochs,
                 batch_frac):
        """
        Class that implements the NN ensemble, used to calculate the uncertainty bands of the NN prediction
        :param num_networks: number of baseline models
        :param num_params_varied: dimensionality of parameter space
        :param dofs: state dimension of the physical system
        :param depths: depths of the baseline models
        :param widths: widths of the baseline models
        :param activations: activation functions of the baseline models
        :param epochs: epochs used for the training of the baseline models
        :param batch_frac: batch number for the training of the baseline models
        """
        self.num_networks = num_networks
        self.num_params_varied = num_params_varied
        self.dofs = dofs
        self.epochs = epochs
        self.batch_frac = batch_frac

        ### Define the different depths, widths, activation functions and initializations of the ensemble members
        self.ensemble_depths = depths
        self.ensemble_widths = widths
        self.ensemble_activations = activations

        self.times_run = 1

    @staticmethod
    def relative_error(y_test, prediction, tensor=False, mean=True):
        err_list = []
        for row in range(y_test.shape[0]):
            if tensor:
                err = np.linalg.norm(y_test[row] - prediction[row].detach().numpy()) / np.linalg.norm(y_test[row])
            else:
                err = np.linalg.norm(y_test[row] - prediction[row]) / np.linalg.norm(y_test[row])

            err_list.append(err)

        if mean:
            return np.mean(err_list)
        else:
            return err_list

    def train_net(self, nn_depth, nn_width, nn_activation, nn_epochs,
                  X_train, y_train, X_valid, y_valid, batch_size_frac, verbose=False):
        """
        Function that implements the training of each baseline model
        :param nn_depth:
        :param nn_width:
        :param nn_activation:
        :param nn_epochs:
        :param X_train:
        :param y_train:
        :param X_valid:
        :param y_valid:
        :param batch_size_frac:
        :param verbose: Boolean: whether or not to plot and visualize specific results of the NN training
        :return:
        """

        ### Set up the network used for training
        NN_member = FNN(num_input=self.num_params_varied, num_output=y_train.shape[1],
                        depth=nn_depth, width=nn_width, activation=nn_activation)

        optimizer = torch.optim.Adam(NN_member.parameters(), lr=7e-4)
        loss_criterion = torch.nn.MSELoss()

        TrainSamples = X_train.shape[0]
        ValidSamples = X_valid.shape[0]

        dataset_train = Data(X=X_train, y=y_train)
        dataset_valid = Data(X=X_valid, y=y_valid)

        train_loader = DataLoader(dataset=dataset_train, batch_size=int(TrainSamples * batch_size_frac))
        valid_loader = DataLoader(dataset=dataset_valid, batch_size=int(ValidSamples * batch_size_frac))

        ### Training of the FNN
        loss_epochs = []
        valid_errors = []
        best_val_loss = float('inf')
        best_model_weights = None

        if verbose:
            pbar = tqdm(total=nn_epochs, desc="Epochs training...")
        for epoch in range(nn_epochs):
            # Training Phase
            NN_member.train(True)
            relative_error_train = []
            relative_error_valid = []
            for x, y in train_loader:
                optimizer.zero_grad()
                y_pred = NN_member(x)
                loss = loss_criterion(y_pred, y)
                loss.backward()
                optimizer.step()

                ### Mean relative error of the batch
                relative_error_train.append(
                    np.linalg.norm(y.detach().numpy() - y_pred.detach().numpy()) / np.linalg.norm(y.detach().numpy()))

            # Mean relative error of the epoch
            loss_epoch = np.mean(relative_error_train)
            loss_epochs.append(loss_epoch)

            # Validation Phase
            NN_member.eval()
            with torch.no_grad():
                for x_val, y_val in valid_loader:
                    y_val_pred = NN_member(x_val)
                    relative_error_valid.append(np.linalg.norm(y_val.detach().numpy() - y_val_pred.detach().numpy())
                                                / np.linalg.norm(y_val.detach().numpy()))

                mean_relative_err_val = np.mean(relative_error_valid)
            valid_errors.append(mean_relative_err_val)

            ### Keep track of the model that results to the minimum validation error
            if mean_relative_err_val < best_val_loss:
                best_val_loss = mean_relative_err_val
                best_model_weights = NN_member.state_dict()

            if verbose:
                print(f"Epoch   Training   Validation\n"
                      f"{epoch}   {loss_epoch}   {mean_relative_err_val}\n"
                      f"====================================================")
            if verbose:
                pbar.update()
        if verbose:
            pbar.close()
        print("Done training!")
        print(f"min train error: {min(loss_epochs)}, min valid error: {min(valid_errors)}")

        if verbose:
            ### Plot the losses
            plt.semilogy(loss_epochs, label='Training error')
            plt.semilogy(valid_errors, label='Validation error')
            plt.xlabel("# Epochs")
            plt.ylabel("Relative MSE")
            plt.legend()
            plt.show()

        if best_model_weights:
            NN_member.load_state_dict(best_model_weights)

        return NN_member

    def EnsembleTraining(self, X_train, y_train, X_valid, y_valid):
        """
        Method that performs the ensemble NN training, in order to get the uncertainty bands on the prediction
        :param X_train:
        :param y_train:
        :param X_valid:
        :param y_valid:
        :return:
        """

        ensemble_members = []
        pbar = tqdm(total=self.num_networks)
        for i in range(self.num_networks):
            net = self.train_net(nn_depth=self.ensemble_depths[i], nn_width=self.ensemble_widths[i],
                                 nn_activation=self.ensemble_activations[i],
                                 nn_epochs=self.epochs, X_train=X_train, y_train=y_train,
                                 X_valid=X_valid, y_valid=y_valid, batch_size_frac=self.batch_frac, verbose=False)

            ensemble_members.append(net)
            pbar.update()
        pbar.close()

        ### For the mean training error
        nets_evaluated_train = [neural_net(torch.tensor(X_train, dtype=torch.float32))
                                for neural_net in ensemble_members]

        train_errors = [self.relative_error(y_train, nets_evaluated_train[i].detach().numpy(),
                                            tensor=False)
                        for i in range(len(nets_evaluated_train))]
        mean_train_errors = np.mean(train_errors)

        ### Now the parameters "theta" of all the ensemble members have been learned
        ### Also for the mean valid errors
        nets_evaluated_valid = [neural_net(torch.tensor(X_valid, dtype=torch.float32))
                                for neural_net in ensemble_members]

        valid_errors = [
            self.relative_error(y_valid, nets_evaluated_valid[i].detach().numpy(), tensor=False)
            for i in range(len(nets_evaluated_valid))]
        mean_valid_errors = np.mean(valid_errors)

        means = [nets_evaluated_valid[i].detach().numpy() for i in range(self.num_networks)]

        ensemble_mean = np.mean(np.array(means), axis=0)
        ensemble_variance = np.var(np.array(means), axis=0)
        ensemble_variance = np.max(ensemble_variance, axis=1)

        max_variance = np.argmax(ensemble_variance)
        point_adaptive = X_valid[max_variance]

        return point_adaptive, ensemble_mean, ensemble_variance, mean_train_errors, mean_valid_errors, ensemble_members


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


def ActiveLearning_Algorithm(num_init, pLANDO, n_all, dim_init, low_init, upp_init, num_valid,
                             onlinephase_args, ensemble_args):
    """
    Function that implements the active learning algorithm used in this work
    :param num_init: The number of initial samples, collected with LHS
    :param pLANDO: The pLANDO instance that will be used throughout the algorithm
    :param n_all: The number of desired final samples
    :param dim_init: Dimensionality of parameter space
    :param low_init: Boundaries of the parameter space
    :param upp_init: Boundaries of the parameter space
    :param num_valid: Number of samples in the validation set
    :param onlinephase_args: Arguments used in the online phase of pLANDO
    :param ensemble_args: Arguments used in the NN ensemble training
    :return:
    """
    ### First, we generate the initial points of the training dataset
    ### Also, generate the validation samples that will be the same throughout the active learning simulation
    initial_samples = LatinHypercube(dim_init, low_init, upp_init, num_init)
    validation_samples = LatinHypercube(dim_init, low_init, upp_init, num_valid)

    training_samples = initial_samples

    ensemble = Ensemble_NN(*ensemble_args)
    adaptive_samples = []
    ensemble_means = []
    ensemble_vars = []
    train_errors = []
    valid_errors = []
    i = 0

    pbar = tqdm(total=n_all - training_samples.shape[0], desc="Progress of Active Learning algorithm...", position=0)
    while training_samples.shape[0] < n_all:
        pLANDO.OfflinePhase(samples_train_al=training_samples, samples_valid_al=validation_samples)
        X_train, y_train, X_valid, y_valid, mean_err_train, mean_err_test = pLANDO.OnlinePhase(*onlinephase_args)

        ### Train the Ensemble NN
        new_sample, mean, var, train_err, valid_err, ensemble_nets = ensemble.EnsembleTraining(X_train, y_train,
                                                                                               X_valid, y_valid)
        adaptive_samples.append(new_sample)
        ensemble_means.append(mean)
        ensemble_vars.append(var)

        train_errors.append(train_err)
        valid_errors.append(valid_err)

        ### Augment the training dataset
        training_samples = np.vstack((X_train, new_sample))

        i += 1
        pbar.update()
    pbar.close()

    adaptive = np.vstack((adaptive_samples))

    plt.scatter(initial_samples[:, 0], initial_samples[:, 1], label='Initial Samples')
    plt.scatter(adaptive[:, 0], adaptive[:, 1], label='Active Learning Samples')
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1.11), ncols=2)
    plt.xlabel(r"$\mu_1 = \lambda$")
    plt.ylabel(r"$\mu_2 = \epsilon$")
    plt.show()

    plt.figure()
    plt.plot(train_errors, '-o', color='blue', label='Training errors')
    plt.plot(valid_errors, '-o', color='red', label='Validation errors')
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1.11), ncols=2)
    plt.xlabel("Number of adaptive samples")
    plt.grid(True)
    plt.ylabel(r"Mean $L_2$ norm of relative error")
    plt.show()

    return training_samples


