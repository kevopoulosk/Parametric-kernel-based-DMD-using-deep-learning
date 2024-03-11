from scipy.stats import qmc
from Sparse_Dictionary_Learning import *
from torch.utils.data import DataLoader
from sklearn import preprocessing
import torch.utils.data as data
import torch
from torch import nn


class FNN(nn.Module):

    def __init__(self, num_input, num_output, depth, width):
        """
        Class that implements the fully connected neural network
        It is used to learn the mapping from the parameter space --> to x or f(x)
        :param num_input: The number of input nodes (e.g. 4 for Lotka-Volterra model)
        :param num_output: The number of output nodes
        :param depth: number of hidden layers
        :param width: number of nodes in each layer
        """
        super().__init__()
        # Number of hidden layers of the network
        self.Depth = depth
        # Number of nodes in each hidden layer
        self.Width = width
        # Number of input nodes for the network
        self.NumInput = num_input
        # Number of output nodes for the network
        self.NumOutput = num_output

        layers = []
        # Start with the input layer
        layers.append(nn.Linear(in_features=self.NumInput, out_features=self.Width))
        layers.append(nn.ReLU())

        for i in range(self.Depth):
            # Add hidden layers
            layers.append(nn.Linear(in_features=self.Width, out_features=self.Width))
            layers.append(nn.ReLU())

        # output layer
        layers.append(nn.Linear(in_features=self.Width, out_features=self.NumOutput))
        # not relu activation in the output

        self.fnn_stack = nn.Sequential(*layers)
        # Add bias to enhance performance in unseen parameters
        self.b = torch.nn.parameter.Parameter(torch.zeros(self.NumOutput))

    def forward(self, x):
        fnn_output = self.fnn_stack(x)

        fnn_output += self.b

        return fnn_output


class Data(data.Dataset):

    def __init__(self, X, y):
        """
        Class that preprocesses the data that go into the FNN architecture.
        This class is needed for the Dataloader.
        :param X: Inputs of the neural network (e.g. the parameters "mu")
        :param y: Outputs of the neural network (e.g. x or f(x) for a specific timestep t*)
        """

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        # Number of data points we have.
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        data_point_x = self.X[idx]
        data_point_y = self.y[idx]
        return data_point_x, data_point_y


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

    l_bounds = low_bounds  # ,0.001 , 0.02, 0.0015
    u_bounds = upp_bounds  # , 0.0025 ,0.3, 0.0025
    sample_params = qmc.scale(sample, l_bounds, u_bounds)
    return sample_params


class ParametricLANDO:

    def __init__(self, kernel, horizon_train, sparsity_tol, num_samples_train, num_sensors):
        """
        Class that implements the parametric form of the LANDO framework.
        :param kernel: The chosen kernel. For instance this be linear, quadratic or gaussian. For this Lotka-Volterra
        problem, the quadratic kernel is used.
        :param horizon_train: The time horizon that we train the model. e.g. if horizon_train = 400, we train the model
        for t in [0, 400].
        :param sparsity_tol: The sparsity threshold needed for the sparse dictionary algorithm
        :param num_samples_train: Number of "mu" samples that are used for training
        :param num_sensors: Number of sensors that the f(x) is discretized.
        """
        self.kernel = kernel
        self.T_end_train = horizon_train
        self.sparsity_tol = sparsity_tol
        self.num_samples = num_samples_train
        self.num_sensors = num_sensors

    def OfflinePhase(self):
        # First, the parameters for training are sampled from a latin hypercube
        # This will change with the active learning method.
        param_samples_lh = LatinHypercube(dim_sample=1, low_bounds=[0.01], upp_bounds=[0.1],
                                          num_samples=self.num_samples)
        self.params_not_varied = np.array([0.002, 0.2, 0.0025])  # 0.002, 0.2
        self.params_not_varied = np.tile(self.params_not_varied, (self.num_samples, 1))

        self.param_samples = np.concatenate((param_samples_lh, self.params_not_varied), axis=1)

        self.scaler = preprocessing.MaxAbsScaler()

        self.SparseDicts_all = []
        self.scaled_X_all = []
        X_perm_all = []
        Y_vals = []
        Y_perm_all = []
        X_all = []

        pbar = tqdm(total=self.num_samples, desc=f"Offline Phase -> Generation of training data...")
        # Now we generate the sparse dictionaries for all the parametric samples.
        # So, for each training sample we have W_tilde, x_tilde(sparse dictionary), and k(x_tilde, x)
        for val, param_sample in enumerate(self.param_samples):
            X, _ = Lotka_Volterra_Snapshot(params=param_sample, T=self.T_end_train, num_sensors=self.num_sensors)
            Y = Lotka_Volterra_Deriv(X, *param_sample)

            scaledX = Scale(X)

            Xperm, perm = Permute(X)
            Yperm = Y[:, perm]

            SparseDict, _, _, _ = SparseDictionary(Xperm, scaledX, self.kernel,
                                                   tolerance=self.sparsity_tol, pbar_bool=False)
            self.SparseDicts_all.append(SparseDict)

            self.scaled_X_all.append(scaledX)
            X_perm_all.append(Xperm)
            Y_perm_all.append(Yperm)
            X_all.append(X)
            Y_vals.append(Y)
            pbar.update()
        pbar.close()

        # Compute W tilde and form the model for all the training samples.
        self.W_tildes = [y_perm @ np.linalg.pinv(self.kernel(Sparse_Dict, scaled_X * x_perm))
                         for y_perm, Sparse_Dict, scaled_X, x_perm in
                         zip(Y_perm_all, self.SparseDicts_all, self.scaled_X_all, X_perm_all)]

        kernels = [self.kernel(Sparse_Dict, scaled_X * X_mat)
                   for Sparse_Dict, scaled_X, X_mat in zip(self.SparseDicts_all, self.scaled_X_all, X_all)]

        models = [W_tilde_mat @ kernel_mat for W_tilde_mat, kernel_mat in zip(self.W_tildes, kernels)]

        # sanity check -> compute reconstruction error to make sure it is small
        reconstruction_relative_errors = [abs(Y_vals[i][0] - models[i][0]) / Y_vals[i][0] for i in range(len(Y_vals))]
        mean_reconstruction_relative_errors = [np.mean(list_err) for list_err in reconstruction_relative_errors]

        print(f"Training Data: The mean relative reconstruction errors are: {mean_reconstruction_relative_errors}")

        return self.W_tildes, self.SparseDicts_all, self.param_samples

    def OnlinePhase(self, num_samples_test, T_end_test, fraction_train, fnn_depth, fnn_width, epochs):
        """
        Method that implements the online phase of the algorithm.
        :param num_samples_test: The number of test samples. These are parameters "mu" that are unseen during the training
        :param T_end_test: The time horizon for the prediction, "t*". Note that typically t* > horizon_train
        This means that we try to predict the state of the system x or f(x), for timesteps that are not included in the training.
        For instance, maybe the model is trained for t in [0,400] but we want to predict f(x) farther than that -> extrapolation
        :param fraction_train: The fraction of the parameters that are used for training. The rest are used for validation
        :param fnn_depth: Depth of the neural network
        :param fnn_width: Width of the neural network
        :param epochs: Epochs that the neural network is trained for
        :return:
        """
        # Generate the test data -> new parameters to test for the specific T_end_test
        param_samples_lh_test = LatinHypercube(dim_sample=1, low_bounds=[0.01], upp_bounds=[0.1],
                                               num_samples=num_samples_test)

        params_not_varied_test = np.array([0.002, 0.2, 0.0025])  # 0.002, 0.2
        params_not_varied_test = np.tile(params_not_varied_test, (num_samples_test, 1))

        param_samples_test = np.concatenate((param_samples_lh_test, params_not_varied_test), axis=1)

        # For each parameter in the training set run the LANDO algorithm to compute the dynamics until T_end_test
        # T_end_test > horizon_train ---> extrapolation
        lando_dynamics = []
        true_dynamics = []
        pbar = tqdm(total=self.num_samples, desc=f"Online Phase -> Prediction of f(x, t*; mu)...")
        for i, sample in enumerate(self.param_samples):
            X, _ = Lotka_Volterra_Snapshot(params=sample, T=T_end_test, num_sensors=self.num_sensors)
            Y = Lotka_Volterra_Deriv(X, *sample)

            f_mu = self.W_tildes[i] @ quadratic_kernel(self.SparseDicts_all[i], self.scaled_X_all[i] * X)

            lando_dynamics.append(f_mu)
            true_dynamics.append(Y)
            pbar.update()
        pbar.close()

        # sanity check -> compute reconstruction error to make sure it is small
        reconstruction_relative_errors = [abs(true_dynamics[i][0] - lando_dynamics[i][0]) / true_dynamics[i][0]
                                          for i in range(len(true_dynamics))]

        mean_reconstruction_relative_errors = [np.mean(list_err) for list_err in reconstruction_relative_errors]
        print(f"The mean relative reconstruction errors are: {mean_reconstruction_relative_errors}")

        # Now, for each parameter value in the test set, compute the true dynamics for comparison
        true_dynamics_test = []
        for val, sample_test in enumerate(param_samples_test):
            X, _ = Lotka_Volterra_Snapshot(params=sample_test, T=T_end_test, num_sensors=self.num_sensors)
            Y = Lotka_Volterra_Deriv(X, *sample_test)
            true_dynamics_test.append(Y)

        # Set up the neural network to learn the mapping
        Mapping_FNN = FNN(num_input=self.param_samples.shape[1],
                          num_output=2 * self.num_sensors, depth=fnn_depth, width=fnn_width)

        optimizer = torch.optim.Adam(Mapping_FNN.parameters(), lr=1e-5)
        loss_criterion = torch.nn.MSELoss()

        # Split the param_samples matrix into training and validation data
        TrainSamples = int(self.param_samples.shape[0] * fraction_train)
        # Number of samples for validation
        ValidSamples = self.param_samples.shape[0] - TrainSamples
        # Number of samples for test
        TestSamples = num_samples_test

        # These are fed to the Dataloader, and then to the NN for the training
        # Note that now we scale the parameters to the range (0, 1).
        # Hopefully, this will enhance the performance of the NN interpolator
        X_train = self.scaler.fit_transform(self.param_samples[:TrainSamples, :])
        y_train = lando_dynamics[:TrainSamples]
        y_train = np.vstack([y_train[val].flatten() for val in range(len(y_train))])

        X_valid = self.scaler.fit_transform(self.param_samples[TrainSamples:, :])
        y_valid = lando_dynamics[TrainSamples:]
        y_valid = np.vstack([y_valid[val_].flatten() for val_ in range(len(y_valid))])

        X_test = self.scaler.fit_transform(param_samples_test)
        y_test = true_dynamics_test
        y_test = np.vstack([y_test[val__].flatten() for val__ in range(len(y_test))])

        # Construct the datasets
        dataset_train = Data(X=X_train, y=y_train)
        dataset_valid = Data(X=X_valid, y=y_valid)
        dataset_test = Data(X=X_test, y=y_test)

        train_loader = DataLoader(dataset=dataset_train, batch_size=int(TrainSamples * 0.45))
        valid_loader = DataLoader(dataset=dataset_valid, batch_size=int(ValidSamples * 0.45))
        test_loader = DataLoader(dataset=dataset_test, batch_size=int(TestSamples * 0.45))

        # Now we train the model
        epochs = epochs

        pbar = tqdm(total=epochs, desc="Epochs training...")
        loss_epochs = []
        valid_errors = []
        best_val_loss = float('inf')
        best_model_weights = None

        for epoch in range(epochs):
            # Training Phase
            Mapping_FNN.train(True)
            relative_error_train = []
            relative_error_valid = []
            for x, y in train_loader:
                y_pred = Mapping_FNN(x)
                loss = loss_criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # compute the mean relative error of the batch
                relative_error_train.append(torch.mean(abs(y - y_pred) / y).detach().numpy())

            # compute the relative error of the epoch
            loss_epoch = np.mean(relative_error_train)
            loss_epochs.append(loss_epoch)

            # Validation Phase
            Mapping_FNN.eval()
            with torch.no_grad():
                for x_val, y_val in valid_loader:
                    y_val_pred = Mapping_FNN(x_val)
                    relative_error_valid.append(torch.mean(abs(y_val - y_val_pred) / y_val).detach().numpy())

                mean_relative_err_val = np.mean(relative_error_valid)
            valid_errors.append(mean_relative_err_val)

            # Keep track of the model that results to the minimum validation error
            if mean_relative_err_val < best_val_loss:
                best_val_loss = mean_relative_err_val
                best_model_weights = Mapping_FNN.state_dict()
            # The below code is used if we want to stop the training process earlier, if validation error < 1%
            # Now this feature is disabled, because we want to test the performance of the FNN during the whole training

            # if mean_relative_err_val < 0.009:
            #     print(f"Stopping early at epoch {epoch}, since mean_relative_err_val < 0.009")
            #     break
            #
            # # Reduce the learning rate when we have validation error < 0.01
            # if mean_relative_err_val < 0.0092:
            #     scheduler.step(mean_relative_err_val)

            print(f"Epoch   Training   Validation\n"
                  f"{epoch}   {loss_epoch}   {mean_relative_err_val}\n"
                  f"====================================================")

            pbar.update()
        pbar.close()
        print("Done training!")

        # Plot the losses
        plt.semilogy(loss_epochs, label='Training error')
        plt.semilogy(valid_errors, label='Validation error')
        plt.xlabel("# Epochs")
        plt.ylabel("Relative MSE")
        plt.legend()
        plt.show()

        if best_model_weights:
            Mapping_FNN.load_state_dict(best_model_weights)

        # For all these unseen test parameters, evaluate the neural network after training and approximate f(x,t*;mu*)
        test_error = []
        predicted_dynamics = []
        Mapping_FNN.eval()
        with torch.no_grad():
            for x_test, y_test in test_loader:
                y_pred_test = Mapping_FNN(x_test)
                for row1 in range(len(y_test)):
                    predicted_dynamics.append(y_pred_test[row1])

                error_batch_test = torch.mean(abs(y_test - y_pred_test) / y_test)
                test_error.append(error_batch_test.detach().numpy())
        print(f"The mean test error is {np.mean(test_error)} based on {TestSamples} test samples")

        return predicted_dynamics, true_dynamics_test, test_error


T_train = 400
NumSamples_Train = 5000
sensors = 600
sparsity_lando = 1e-6

NumSamples_Test = 10000
T_test = int(1.5 * T_train)
train_frac = 0.6
depth = 5
width = 1500
epochs = 25000

# Perform the whole procedure of the parametric LANDO. First we perform the offline phase of the algorithm
# and subsequently, the online phase is performed

parametric_lando = ParametricLANDO(kernel=quadratic_kernel, horizon_train=T_train, num_samples_train=NumSamples_Train,
                                   num_sensors=sensors, sparsity_tol=sparsity_lando)

w_tildes, sparse_dicts, mu_samples_train = parametric_lando.OfflinePhase()

fx_pred, fx_true, error_test = parametric_lando.OnlinePhase(num_samples_test=NumSamples_Test, T_end_test=T_test,
                                                            fraction_train=train_frac, fnn_depth=depth, fnn_width=width,
                                                            epochs=epochs)

print(f"fx_pred has shape {fx_pred[0].shape}")

# Plot the predicted f(x) vs ground truth dxdt
t = np.linspace(0, T_test, sensors)
plt.style.use('_mpl-gallery')
for i in range(len(fx_pred)):
    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(9, 6))

    # Plot the first subplot
    axs[0].plot(t, fx_true[i][0, :], color='blue', label='Prey True')
    axs[0].plot(t, fx_pred[i][:sensors], linestyle='dotted', color='red', label="Prey Pred.")
    axs[0].axvline(x=400, linestyle='--')
    axs[0].legend()
    axs[0].set_title(
        f'Prediction relative error = {np.mean(abs(fx_true[i][0, :] - fx_pred[i][:sensors].detach().numpy()) / fx_true[i][0, :])} ')
    axs[0].set_xlabel(r"$t$")
    axs[0].set_ylabel("f(x) - Population")

    # Plot the second subplot
    axs[1].plot(t, fx_true[i][1, :], color='blue', label='Predator True')
    axs[1].plot(t, fx_pred[i][sensors:], linestyle='dotted', color='black', label="Predator Pred.")
    axs[1].axvline(x=400, linestyle='--')
    axs[1].legend()
    axs[1].set_title(
        f'Prediction relative error = {np.mean(abs(fx_true[i][1, :] - fx_pred[i][sensors:].detach().numpy()) / fx_true[i][1, :])} ')
    axs[1].set_xlabel(r"$t$")
    axs[1].set_ylabel("f(x) - Population")
    plt.tight_layout()
    plt.show()

# Notes: 1) Currently this is not point prediction, because the model is low-dimensional we can predict the whole f(x)
#  2) For the same reason, the step of dimension reduction to the degrees of freedom with POD is not implemented yet
#  Remember that this approach holds only for low-dim dynamical systems.
#  For high-dim systems the POD method should also be implemented for dimensionality reduction (or autoencoder)
# ALso, for high-dim systems we can have only the point prediction

