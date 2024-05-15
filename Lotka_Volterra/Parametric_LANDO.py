import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc
from Lotka_Volterra.Lotka_Volterra_model import *
from Lotka_Volterra.Sparse_Dictionary_Learning import *
from torch.utils.data import DataLoader
from sklearn import preprocessing
import torch.utils.data as data
import torch
from torch import nn
from scipy.interpolate import RBFInterpolator


class Snake(nn.Module):
    def __init__(self, alpha=0.5):
        """
        Implementation of the snake activation function for the NN
        :param alpha: The assumed frequency of the data passed to NN.
        """
        super(Snake, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return x + (1 / self.alpha) * torch.sin(self.alpha * x) ** 2


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

        self.Depth = depth
        self.Width = width
        self.NumInput = num_input
        self.NumOutput = num_output

        layers = []

        layers.append(nn.Linear(in_features=self.NumInput, out_features=self.Width))
        layers.append(Snake())

        for i in range(self.Depth):
            layers.append(nn.Linear(in_features=self.Width, out_features=self.Width))
            layers.append(Snake())

        layers.append(nn.Linear(in_features=self.Width, out_features=self.NumOutput))

        self.fnn_stack = nn.Sequential(*layers)
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
        :param X: Inputs of the neural network (the parameters "mu")
        :param y: Outputs of the neural network (x for a specific timestep t*)
        """

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
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

    l_bounds = low_bounds
    u_bounds = upp_bounds
    sample_params = qmc.scale(sample, l_bounds, u_bounds)
    return sample_params


class ParametricLANDO:

    def __init__(self, kernel, horizon_train, sparsity_tol, num_samples_train, num_sensors, batch_frac, params_varied,
                 low_bound, upp_bound, fixed_params_val, rbf=False):
        """
        Class that implements the parametric form of the LANDO framework.
        :param kernel: The chosen kernel. For instance this be linear, quadratic or gaussian. For this Lotka-Volterra
        problem, the quadratic kernel is used, since it yields the best performance.
        :param horizon_train: The time horizon that we train the model. e.g. if horizon_train = 400, we train the model
        for t in [0, 400].
        :param sparsity_tol: The sparsity threshold needed for the sparse dictionary algorithm
        :param num_samples_train: Number of "mu" samples that are used for training
        :param num_sensors: Number of sensors that the f(x) is discretized.
        :param batch_frac: fraction of the traininng data to be used in 1 batch
        :param params_varied: number of parameters that are varied
        :param low_bound: the lower bound of parameters that are sampled
        :param upp_bound: the upper bound of parameters that are sampled
        :param fixed_params_val: the values of the parameters that are assumed to be fixed
        """
        self.kernel = kernel
        self.T_end_train = horizon_train
        self.sparsity_tol = sparsity_tol
        self.num_samples = num_samples_train
        self.num_sensors = num_sensors
        self.batch_size_frac = batch_frac
        self.num_params_varied = params_varied
        self.low_bounds = low_bound
        self.upp_bounds = upp_bound
        self.params_fixed = fixed_params_val
        self.rbf = rbf

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

    def RBF_Interp(self, fraction_train, num_samples_test, param_samples_test, lando_dynamics, true_dynamics_test):
        ### Split the parametric samples into training and validation data
        TrainSamples = int(self.param_samples.shape[0] * fraction_train)
        ### Number of samples for test
        TestSamples = num_samples_test

        if self.num_params_varied > 1:
            scaler = preprocessing.MaxAbsScaler()

        ### If 1+ parameters are varied, then they are scaled to the range (0, 1).
        ### This scaling enhances the performance of the NN interpolator
        if self.num_params_varied > 1:
            X_train = scaler.fit_transform(self.param_samples[:TrainSamples, :self.num_params_varied])
            X_test = scaler.fit_transform(param_samples_test[:, :self.num_params_varied])
        else:
            X_train = self.param_samples[:TrainSamples, :self.num_params_varied]
            X_test = param_samples_test[:, :self.num_params_varied]

        y_train = np.vstack([lando_dynamics[:TrainSamples][i][:, -1] for i in range(TrainSamples)])
        y_test = np.vstack([true_dynamics_test[i] for i in range(TestSamples)])

        ### RBF interpolation
        rbf_function = RBFInterpolator(X_train, y_train, kernel='cubic', degree=1)

        prediction = rbf_function(X_test)

        mean_relative_error = self.relative_error(y_test=y_test, prediction=prediction)

        print(f"The mean test error is {mean_relative_error} based on {TestSamples} test samples")

        return prediction, mean_relative_error, rbf_function, X_train, y_train, X_test, y_test

    def train_fnn(self, fnn_depth, fnn_width, fraction_train, num_samples_test,
                  lando_dynamics, true_dynamics_test, epochs, param_samples_test, verbose=True):

        ### Set up the neural network to learn the mapping
        Mapping_FNN = FNN(num_input=self.num_params_varied, num_output=self.dofs, depth=fnn_depth, width=fnn_width)

        optimizer = torch.optim.Adam(Mapping_FNN.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.04, patience=3)
        loss_criterion = torch.nn.MSELoss()

        ### Split the parametric samples into training and validation data
        TrainSamples = int(self.param_samples.shape[0] * fraction_train)
        ### Number of samples for validation
        ValidSamples = self.param_samples.shape[0] - TrainSamples
        ### Number of samples for test
        TestSamples = num_samples_test

        if self.num_params_varied > 1:
            scaler = preprocessing.MaxAbsScaler()

        ### If 1+ parameters are varied, then they are scaled to the range (0, 1).
        ### This scaling enhances the performance of the NN interpolator
        if self.num_params_varied > 1:
            X_train = scaler.fit_transform(self.param_samples[:TrainSamples, :self.num_params_varied])
            X_valid = scaler.fit_transform(self.param_samples[TrainSamples:, :self.num_params_varied])
            X_test = scaler.fit_transform(param_samples_test[:, :self.num_params_varied])
        else:
            X_train = self.param_samples[:TrainSamples, :self.num_params_varied]
            X_valid = self.param_samples[TrainSamples:, :self.num_params_varied]
            X_test = param_samples_test[:, :self.num_params_varied]

        y_train = np.vstack([lando_dynamics[:TrainSamples][i][:, -1] for i in range(TrainSamples)])
        y_valid = np.vstack([lando_dynamics[TrainSamples:][i][:, -1] for i in range(ValidSamples)])
        y_test = np.vstack([true_dynamics_test[i] for i in range(TestSamples)])

        dataset_train = Data(X=X_train, y=y_train)
        dataset_valid = Data(X=X_valid, y=y_valid)

        train_loader = DataLoader(dataset=dataset_train, batch_size=int(TrainSamples * self.batch_size_frac))
        valid_loader = DataLoader(dataset=dataset_valid, batch_size=int(ValidSamples * self.batch_size_frac))

        loss_epochs = []
        valid_errors = []
        best_val_loss = float('inf')
        best_model_weights = None

        if verbose:
            pbar = tqdm(total=epochs, desc="Epochs training...")
        for epoch in range(epochs):
            # Training Phase
            Mapping_FNN.train(True)
            relative_error_train = []
            relative_error_valid = []
            for x, y in train_loader:
                optimizer.zero_grad()
                y_pred = Mapping_FNN(x)
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
            Mapping_FNN.eval()
            with torch.no_grad():
                for x_val, y_val in valid_loader:
                    y_val_pred = Mapping_FNN(x_val)
                    relative_error_valid.append(np.linalg.norm(y_val.detach().numpy() - y_val_pred.detach().numpy())
                                                / np.linalg.norm(y_val.detach().numpy()))

                mean_relative_err_val = np.mean(relative_error_valid)
            valid_errors.append(mean_relative_err_val)

            ### Keep track of the model that results to the minimum validation error
            if mean_relative_err_val < best_val_loss:
                best_val_loss = mean_relative_err_val
                best_model_weights = Mapping_FNN.state_dict()

            ### Stop the training process if validation error < 0.5%
            if mean_relative_err_val < 0.006:
                print(f"Stopping early at epoch {epoch}, since mean_relative_err_val < 0.009")
                break

            ### Reduce the learning rate when we have validation error < 0.6%
            if mean_relative_err_val < 0.007:
                scheduler.step(mean_relative_err_val)
            if verbose:
                print(f"Epoch   Training   Validation\n"
                      f"{epoch}   {loss_epoch}   {mean_relative_err_val}\n"
                      f"====================================================")
            if verbose:
                pbar.update()
        if verbose:
            pbar.close()
        print("Done training!")

        if verbose:
            ### Plot the losses
            plt.semilogy(loss_epochs, label='Training error')
            plt.semilogy(valid_errors, label='Validation error')
            plt.xlabel("# Epochs")
            plt.ylabel("Relative MSE")
            plt.legend()
            plt.show()

        if best_model_weights:
            Mapping_FNN.load_state_dict(best_model_weights)

        ### For all unseen test parameters, evaluate the neural network after training and approximate f(x,t*;mu*)

        Mapping_FNN.eval()
        prediction = Mapping_FNN(torch.from_numpy(X_test).to(torch.float32))

        mean_relative_error = self.relative_error(y_test=y_test, prediction=prediction, tensor=True)

        print(f"The mean test error is {mean_relative_error} based on {TestSamples} test samples")

        return Mapping_FNN, X_train, y_train, X_test, y_test, mean_relative_error

    def Visual_1D(self, x_train, y_train, x_test, y_test, interp_model, IC_predict, T_end_test):
        X_train_sort = np.sort(x_train, axis=0)
        sort_index = np.squeeze(np.argsort(x_train, axis=0))
        y_train_sort = y_train[sort_index, :]

        if self.rbf:
            y_final_pred_train = interp_model(X_train_sort)
            error_train = self.relative_error(y_train_sort, y_final_pred_train)
        else:
            y_final_pred_train = interp_model(torch.from_numpy(X_train_sort).to(torch.float32))
            error_train = self.relative_error(y_train_sort, y_final_pred_train, tensor=True)

        plt.figure(figsize=(11, 5))
        plt.plot(X_train_sort, y_train_sort[:, 0], label=r'$x_1$')
        plt.plot(X_train_sort, y_train_sort[:, 1], label=r'$x_2$')
        plt.xlabel(r'$\mu_1 = \alpha$')
        plt.ylabel(r"$\mathbf{x}$")

        if self.rbf:
            plt.plot(X_train_sort, y_final_pred_train[:, 0], '.', label=r'$x^{pred}_{1}$')
            plt.plot(X_train_sort, y_final_pred_train[:, 1], '.', label=r'$x^{pred}_{2}$')
            filename = "Training_RBF.png"
        else:
            plt.plot(X_train_sort, y_final_pred_train[:, 0].detach().numpy(), '.', label=r'$x^{pred}_{1}$')
            plt.plot(X_train_sort, y_final_pred_train[:, 1].detach().numpy(), '.', label=r'$x^{pred}_{2}$')
            filename = "Training.png"

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)

        directory = f"/Users/konstantinoskevopoulos/Desktop/Lotka_Volterra_Results/Param1D/IC={IC_predict}, t_end={T_end_test}/"

        plt.savefig(directory + filename)
        plt.show()

        X_test_sort = np.sort(x_test, axis=0)
        sort_index = np.squeeze(np.argsort(x_test, axis=0))
        y_test_sort = y_test[sort_index, :]

        if self.rbf:
            y_final_pred = interp_model(X_test_sort)
            error_test = self.relative_error(y_test_sort, y_final_pred)
        else:
            y_final_pred = interp_model(torch.from_numpy(X_test_sort).to(torch.float32))
            error_test = self.relative_error(y_test_sort, y_final_pred, tensor=True)

        plt.figure(figsize=(11, 5))
        plt.plot(X_test_sort, y_test_sort[:, 0], label=r'$x_1$')
        plt.plot(X_test_sort, y_test_sort[:, 1], label=r'$x_2$')

        plt.grid(True)

        plt.xlabel(r'$\mu_1 = \alpha$')
        plt.ylabel(r"$\mathbf{x}$")

        if self.rbf:
            plt.plot(X_test_sort, y_final_pred[:, 0], '.', label=r'$x^{pred}_{1}$')
            plt.plot(X_test_sort, y_final_pred[:, 1], '.', label=r'$x^{pred}_{2}$')
            filename = "Test_RBF.png"

        else:
            plt.plot(X_test_sort, y_final_pred[:, 0].detach().numpy(), '.', label=r'$x^{pred}_{1}$')
            plt.plot(X_test_sort, y_final_pred[:, 1].detach().numpy(), '.', label=r'$x^{pred}_{2}$')
            filename = "Test.png"

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(directory + filename)
        plt.show()

        plt.figure(figsize=(11, 5))
        plt.plot(X_test_sort, y_test_sort[:, 0], label=r'$x_1$ Ground Truth')
        plt.plot(X_test_sort, y_test_sort[:, 1], label=r'$x_2$ Ground Truth')
        plt.plot(X_train_sort, y_train_sort[:, 0], '-.', color='black', label=r'$x_1$ LANDO')
        plt.plot(X_train_sort, y_train_sort[:, 1], '-.', color='black', label=r'$x_2$ LANDO')

        plt.ylabel(r"$\mathbf{x}$")
        plt.xlabel(r'$\mu_1 = \alpha$')
        plt.legend(loc='best')
        filename = "LANDO_Vs_Truth.png"
        plt.grid(True)
        plt.savefig(directory + filename)
        plt.show()

        return error_train, error_test

    def Visual_2D(self, x_train, y_train, x_test, y_test, test_samples, train_samples, interp_model, lando_error,
                  IC_predict, T_end_test):

        if self.rbf:
            y_final_pred_train = interp_model(x_train)
            train_error = self.relative_error(y_train, y_final_pred_train, mean=False)

            y_final_pred_test = interp_model(torch.from_numpy(x_test).to(torch.float32))
            test_error = self.relative_error(y_test, y_final_pred_test, mean=False)
        else:
            y_final_pred_train = interp_model(torch.from_numpy(x_train).to(torch.float32))
            train_error = self.relative_error(y_train, y_final_pred_train, mean=False, tensor=True)

            y_final_pred_test = interp_model(torch.from_numpy(x_test).to(torch.float32))
            test_error = self.relative_error(y_test, y_final_pred_test, mean=False, tensor=True)

        directory = f"/Users/konstantinoskevopoulos/Desktop/Lotka_Volterra_Results/Param2D/IC={IC_predict}, t_end={T_end_test}/"
        plt.figure()
        plt.scatter(train_samples[:, 0], train_samples[:, 1], c=train_error, cmap="plasma")
        plt.xlabel(r"$\mu_1$")
        plt.ylabel(r"$\mu_2$")

        if self.rbf:
            filename = "Training_RBF.png"
        else:
            filename = "Training.png"

        plt.colorbar(label=r'$L_2$ relative error')
        plt.savefig(directory + filename)
        plt.show()

        plt.figure()
        plt.scatter(test_samples[:, 0], test_samples[:, 1], c=test_error, cmap="plasma")
        plt.xlabel(r"$\mu_1$")
        plt.ylabel(r"$\mu_2$")

        if self.rbf:
            filename = "Test_RBF.png"
        else:
            filename = "Test.png"
        plt.colorbar(label=r'$L_2$ relative error')
        plt.savefig(directory + filename)
        plt.show()

        plt.figure()
        plt.scatter(train_samples[:, 0], train_samples[:, 1], c=lando_error, cmap="plasma")
        plt.xlabel(r"$\mu_1$")
        plt.ylabel(r"$\mu_2$")
        filename = "LANDO_Vs_Truth.png"
        plt.colorbar(label=r'LANDO $L_2$ relative error')
        plt.savefig(directory + filename)
        plt.show()

        return np.mean(train_error), np.mean(test_error)

    def OfflinePhase(self):
        ### First, the parameters for training are sampled from a latin hypercube
        ### This will change with the active learning method.
        param_samples_lh = LatinHypercube(dim_sample=self.num_params_varied, low_bounds=self.low_bounds,
                                          upp_bounds=self.upp_bounds, num_samples=self.num_samples)

        self.params_not_varied = np.tile(self.params_fixed, (self.num_samples, 1))

        self.param_samples = np.concatenate((param_samples_lh, self.params_not_varied), axis=1)

        self.SparseDicts_all = []
        self.scaled_X_all = []
        X_perm_all = []
        Y_vals = []
        Y_perm_all = []
        X_all = []

        ### The sparse dictionaries for all the parametric samples are generated
        ### So, for each training sample we have W_tilde, x_tilde(sparse dictionary), and k(x_tilde, x)
        pbar = tqdm(total=self.num_samples, desc=f"Offline Phase -> Generation of training data...")
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

        ### Degrees of freedom of the system of interest
        self.dofs = X.shape[0]

        # Compute W tilde and form the model for all the training samples.
        self.W_tildes = [y_perm @ np.linalg.pinv(self.kernel(Sparse_Dict, scaled_X * x_perm))
                         for y_perm, Sparse_Dict, scaled_X, x_perm in
                         zip(Y_perm_all, self.SparseDicts_all, self.scaled_X_all, X_perm_all)]

        kernels = [self.kernel(Sparse_Dict, scaled_X * X_mat)
                   for Sparse_Dict, scaled_X, X_mat in zip(self.SparseDicts_all, self.scaled_X_all, X_all)]

        models = [W_tilde_mat @ kernel_mat for W_tilde_mat, kernel_mat in zip(self.W_tildes, kernels)]

        ### Compute the reconstruction error to make sure it is sufficiently small
        reconstruction_relative_errors = [np.linalg.norm(Y_vals[i] - models[i]) / np.linalg.norm(Y_vals[i])
                                          for i in range(len(Y_vals))]

        print(f"Training Data: The mean relative reconstruction errors are: {reconstruction_relative_errors}")

        return self.W_tildes, self.SparseDicts_all, self.param_samples

    def OnlinePhase(self, num_samples_test, T_end_test, fraction_train, fnn_depth, fnn_width, epochs, IC_predict, verb):
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
        ### Generate the test data --> new parameters to test for the specific T_end_test (t*)
        param_samples_lh_test = LatinHypercube(dim_sample=self.num_params_varied, low_bounds=self.low_bounds,
                                               upp_bounds=self.upp_bounds, num_samples=num_samples_test)

        params_not_varied_test = self.params_fixed
        params_not_varied_test = np.tile(params_not_varied_test, (num_samples_test, 1))

        param_samples_test = np.concatenate((param_samples_lh_test, params_not_varied_test), axis=1)

        ### For each parameter in the training set integrate the LANDO prediction f(x) to compute the dynamics until T_end_test
        ### Typically we can have: T_end_test > horizon_train
        lando_dynamics = []
        true_dynamics = []
        pbar = tqdm(total=self.num_samples, desc=f"Online Phase -> Prediction of f(x, t*; mu)...")
        for i, sample in enumerate(self.param_samples):
            ### This is to compare with the predicted value

            X, _ = Lotka_Volterra_Snapshot(params=sample, T=T_end_test, num_sensors=self.num_sensors,
                                           x0=IC_predict[0], y0=IC_predict[1])

            true_dynamics.append(X)

            def Model_General(t, z):
                """
                Approximation of f(x) = dxdt produced by LANDO framework
                f(x) needs to be integrated, to obtain the state of the system "x"
                """
                x0, x1 = z
                x = np.array([[x0], [x1]])

                return (self.W_tildes[i] @ self.kernel(self.SparseDicts_all[i], self.scaled_X_all[i] * x)).flatten()

            x_pred = Predict(model=Model_General, Tend=T_end_test, IC=IC_predict,
                             sensors=self.num_sensors)
            lando_dynamics.append(x_pred)

            pbar.update()
        pbar.close()

        ### Compute reconstruction errors for "x" to make sure they are sufficiently small
        reconstruction_relative_errors = [
            np.linalg.norm(true_dynamics[i] - lando_dynamics[i]) / np.linalg.norm(true_dynamics[i])
            for i in range(len(true_dynamics))]

        print(f"The mean relative reconstruction errors are: {reconstruction_relative_errors}")

        ### For each parameter value in the test set, compute the true dynamics for comparison
        true_dynamics_test = []
        for val, sample_test in enumerate(param_samples_test):
            X, _ = Lotka_Volterra_Snapshot(params=sample_test, T=T_end_test, num_sensors=self.num_sensors,
                                           x0=IC_predict[0], y0=IC_predict[1])

            ### The prediction is made only for t*
            true_dynamics_test.append(X[:, -1])

        if self.rbf:
            prediction, mean_test_error, interp_model, x_train, y_train, x_test, y_test = self.RBF_Interp(
                fraction_train=fraction_train,
                lando_dynamics=lando_dynamics,
                num_samples_test=num_samples_test,
                param_samples_test=param_samples_test,
                true_dynamics_test=true_dynamics_test)
        else:
            interp_model, x_train, y_train, x_test, y_test, mean_test_error = self.train_fnn(fnn_depth=fnn_depth,
                                                                                             fnn_width=fnn_width,
                                                                                             fraction_train=fraction_train,
                                                                                             lando_dynamics=lando_dynamics,
                                                                                             num_samples_test=num_samples_test,
                                                                                             true_dynamics_test=true_dynamics_test,
                                                                                             epochs=epochs,
                                                                                             param_samples_test=param_samples_test,
                                                                                             verbose=verb)
        if self.num_params_varied == 1:

            mean_train_error, mean_test_error = self.Visual_1D(x_train=x_train, y_train=y_train, x_test=x_test,
                                                               y_test=y_test,
                                                               interp_model=interp_model, IC_predict=IC_predict,
                                                               T_end_test=T_end_test)
        else:

            test_samples = param_samples_lh_test
            train_samples = self.param_samples[:int(fraction_train * self.num_samples), :self.num_params_varied]
            lando_error = reconstruction_relative_errors[:int(fraction_train * self.num_samples)]

            mean_train_error, mean_test_error = self.Visual_2D(x_train=x_train, x_test=x_test, y_train=y_train,
                                                               y_test=y_test,
                                                               interp_model=interp_model, test_samples=test_samples,
                                                               train_samples=train_samples, lando_error=lando_error,
                                                               IC_predict=IC_predict, T_end_test=T_end_test)

        return mean_train_error, mean_test_error
