import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc
from Parametric_LANDO.Sparse_Dictionary_Learning import *
from torch.utils.data import DataLoader
from sklearn import preprocessing
import torch.utils.data as data
import torch
from torch import nn
from tqdm import tqdm
import time

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
        It is used to learn the mapping from the parameter space to the state space
        :param num_input: The number of input nodes
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
                 low_bound, upp_bound, fixed_params_val, generate_snapshot, generate_deriv):
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
        :param generate_snapshot: function that generates the snapshot data X
        :param generate_deriv: function that generates the derivative data (Y) for a system
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
        self.X_Snapshot = generate_snapshot
        self.Y_Deriv = generate_deriv

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
            return np.mean(err_list), np.std(err_list)
        else:
            return err_list

    def train_fnn(self, fnn_depth, fnn_width, fraction_train, lando_dynamics, epochs, verbose=True):
        start_train = time.time()

        ### Split the parametric samples into training and validation data
        TrainSamples = int(self.param_samples.shape[0] * fraction_train)
        ### Number of samples for validation
        ValidSamples = self.param_samples.shape[0] - TrainSamples

        ### If 1+ parameters are varied, then they are scaled to the range (0, 1).
        ### This scaling enhances the performance of the NN interpolator
        if self.num_params_varied > 1:
            scaler = preprocessing.MaxAbsScaler()
            X_train = scaler.fit_transform(self.param_samples[:TrainSamples, :self.num_params_varied])
            X_valid = scaler.fit_transform(self.param_samples[TrainSamples:, :self.num_params_varied])
        else:
            X_train = self.param_samples[:TrainSamples, :self.num_params_varied]
            X_valid = self.param_samples[TrainSamples:, :self.num_params_varied]

        y_train = np.vstack([lando_dynamics[:TrainSamples][i][:, -1] for i in range(TrainSamples)])
        y_valid = np.vstack([lando_dynamics[TrainSamples:][i][:, -1] for i in range(ValidSamples)])

        ### Set up the neural network to learn the mapping
        Mapping_FNN = FNN(num_input=self.num_params_varied, num_output=self.dofs, depth=fnn_depth, width=fnn_width)

        optimizer = torch.optim.Adam(Mapping_FNN.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.04, patience=3)
        loss_criterion = torch.nn.MSELoss()

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
                    np.linalg.norm(y.detach().numpy() - y_pred.detach().numpy()) / np.linalg.norm(
                        y.detach().numpy()))

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
            if mean_relative_err_val < 0.0075:
                print(f"Stopping early at epoch {epoch}, since mean_relative_err_val < 0.009")
                break

            ### Reduce the learning rate when we have validation error < 0.6%
            if mean_relative_err_val < 0.008:
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
        end_train = time.time()

        Mapping_FNN.eval()
        prediction_train = Mapping_FNN(torch.from_numpy(X_train).to(torch.float32))
        mean_relative_error_train = self.relative_error(y_test=y_train, prediction=prediction_train,
                                                        tensor=True, mean=False)

        prediction_valid = Mapping_FNN(torch.from_numpy(X_valid).to(torch.float32))
        mean_relative_error_valid = self.relative_error(y_test=y_valid, prediction=prediction_valid,
                                                        tensor=True, mean=False)

        print(f"Execution time for DNN train {end_train - start_train}")

        return Mapping_FNN, X_train, y_train, X_valid, y_valid, mean_relative_error_train, mean_relative_error_valid

    def Visual_1D(self, x_train, y_train, x_test, y_test, interp_model, directory):
        X_train_sort = np.sort(x_train, axis=0)
        sort_index = np.squeeze(np.argsort(x_train, axis=0))
        y_train_sort = y_train[sort_index, :]

        y_final_pred_train = interp_model(torch.from_numpy(X_train_sort).to(torch.float32))
        error_train_mean, _ = self.relative_error(y_train_sort, y_final_pred_train, tensor=True)

        plt.figure(figsize=(9, 2.5))

        params = {
            'axes.labelsize': 15.4,
            'font.size': 15.4,
            'legend.fontsize': 15.4,
            'xtick.labelsize': 15.4,
            'ytick.labelsize': 15.4,
            'text.usetex': False,
            'axes.linewidth': 2,
            'xtick.major.width': 2,
            'ytick.major.width': 2,
            'xtick.major.size': 2,
            'ytick.major.size': 2,
        }
        plt.rcParams.update(params)

        plt.plot(X_train_sort, y_train_sort[:, 0], label= 'ground truth', linewidth=2.15)
        plt.plot(X_train_sort, y_train_sort[:, 1], linewidth=2.15)

        plt.plot(X_train_sort, y_final_pred_train[:, 0].detach().numpy(), '--', label=r'$x_1$ prediction',
                 linewidth=2)
        plt.plot(X_train_sort, y_final_pred_train[:, 1].detach().numpy(), '--', label=r'$x_2$ prediction',
                 linewidth=2)

        filename = "Training.png"

        plt.legend(ncol=4, bbox_to_anchor=(1.01, 1.25), frameon=False)
        plt.grid(True)

        plt.savefig(directory + filename)
        plt.clf()

        X_test_sort = np.sort(x_test, axis=0)
        sort_index = np.squeeze(np.argsort(x_test, axis=0))
        y_test_sort = y_test[sort_index, :]

        y_final_pred = interp_model(torch.from_numpy(X_test_sort).to(torch.float32))
        error_test_mean, error_test_std = self.relative_error(y_test_sort, y_final_pred, tensor=True)

        plt.figure(figsize=(9, 2.5))
        plt.rcParams.update(params)
        plt.plot(X_test_sort, y_test_sort[:, 0], color='black', label='ground truth', linewidth=2.15)
        plt.plot(X_test_sort, y_test_sort[:, 1], color='black', linewidth=2.15)

        plt.grid(True)

        plt.plot(X_test_sort, y_final_pred[:, 0].detach().numpy(), '--', color='tab:red', label=r'$x_1$ prediction',
                 linewidth=2)
        plt.plot(X_test_sort, y_final_pred[:, 1].detach().numpy(), '--', color='tab:green', label=r'$x_2$ prediction',
                 linewidth=2)
        filename = "Test.png"

        plt.legend(ncol=4, bbox_to_anchor=(1.01, 1.25), frameon=False)
        plt.savefig(directory + filename)
        plt.clf()

        plt.figure(figsize=(9, 2.5))
        plt.plot(X_test_sort, y_test_sort[:, 0], label='ground truth')
        plt.plot(X_test_sort, y_test_sort[:, 1], label='ground truth')
        plt.plot(X_train_sort, y_train_sort[:, 0], '-.', color='black', label=r'$x_1$ LANDO')
        plt.plot(X_train_sort, y_train_sort[:, 1], '-.', color='black', label=r'$x_2$ LANDO')
        plt.legend(loc='best')
        filename = "LANDO_Vs_Truth.png"
        plt.grid(True)
        plt.savefig(directory + filename)
        plt.clf()

        return error_train_mean, error_test_mean, error_test_std

    def Visual_2D(self, x_train, y_train, x_test, y_test, test_samples, train_samples, interp_model, lando_error,
                  IC_predict, T_end_test):

        y_final_pred_train = interp_model(torch.from_numpy(x_train).to(torch.float32))
        train_error = self.relative_error(y_train, y_final_pred_train, mean=False, tensor=True)

        y_final_pred_test = interp_model(torch.from_numpy(x_test).to(torch.float32))
        test_error = self.relative_error(y_test, y_final_pred_test, mean=False, tensor=True)

        directory = f"/Users/konstantinoskevopoulos/Desktop/Lotka_Volterra_Results/Param2D/IC={IC_predict}, t_end={T_end_test}/"

        plt.figure()

        params = {
            'axes.labelsize': 15.4,
            'font.size': 15.4,
            'legend.fontsize': 15.4,
            'xtick.labelsize': 15.4,
            'ytick.labelsize': 15.4,
            'text.usetex': False,
            'axes.linewidth': 2,
            'xtick.major.width': 2,
            'ytick.major.width': 2,
            'xtick.major.size': 2,
            'ytick.major.size': 2,
        }
        plt.rcParams.update(params)

        plt.scatter(train_samples[:, 0], train_samples[:, 1], c=train_error, cmap="plasma")

        filename = "Training.png"

        plt.colorbar(label=r'$L_2$ relative error')
        plt.savefig(directory + filename)
        plt.clf()

        plt.figure()
        plt.scatter(test_samples[:, 0], test_samples[:, 1], c=test_error, cmap="plasma")

        filename = "Test.png"

        plt.colorbar()
        plt.savefig(directory + filename)
        plt.clf()

        plt.figure()
        plt.scatter(train_samples[:, 0], train_samples[:, 1], c=lando_error, cmap="plasma")
        plt.xlabel(r"$\mu_1$")
        plt.ylabel(r"$\mu_2$")
        filename = "LANDO_Vs_Truth.png"
        plt.colorbar()
        plt.savefig(directory + filename)
        plt.clf()

        train_error_mean, _ = self.relative_error(y_train, y_final_pred_train, mean=True, tensor=True)
        test_error_mean, test_error_std = self.relative_error(y_test, y_final_pred_test, mean=True, tensor=True)

        return train_error_mean, test_error_mean, test_error_std

    def OfflinePhase(self):
        start_offline = time.time()

        params_not_varied = np.tile(self.params_fixed, (self.num_samples, 1))

        param_samples_lh = LatinHypercube(dim_sample=self.num_params_varied, low_bounds=self.low_bounds,
                                          upp_bounds=self.upp_bounds, num_samples=self.num_samples)

        self.param_samples = np.concatenate((param_samples_lh, params_not_varied), axis=1)

        self.SparseDicts_all = []
        self.scaled_X_all = []
        X_perm_all = []
        Y_vals = []
        Y_perm_all = []
        X_all = []
        exec_times = []

        ### The sparse dictionaries for all the parametric samples (train + valid OR train+test+valid) are generated
        ### So, for each training sample we have W_tilde, x_tilde(sparse dictionary), and k(x_tilde, x)
        pbar = tqdm(total=self.param_samples.shape[0], desc=f"Offline Phase -> Generation of training data...")
        for val, param_sample in enumerate(self.param_samples):
            start_time = time.time()
            X, _ = self.X_Snapshot(params=param_sample, T=self.T_end_train, num_sensors=self.num_sensors)

            Y = self.Y_Deriv(X, *param_sample)
            end_time = time.time()

            exec_time = end_time - start_time
            exec_times.append(exec_time)


            scaledX = Scale(X)

            Xperm, perm = Permute(X)
            Yperm = Y[:, perm]

            SparseDict = SparseDictionary(Xperm, scaledX, self.kernel,
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

        end_offline = time.time()

        print(f"Mean execution time per sample: {np.mean(exec_times)}")
        print(f"Execution time of offline phase: {end_offline- start_offline}")

        return self.W_tildes, self.SparseDicts_all, self.param_samples

    def OnlinePhase(self, T_end_test, IC_predict, fraction_train=None,
                    epochs=None, verb=False, fnn_depth=None, fnn_width=None):
        """
        Method that implements the online phase of the algorithm.
        :param verb:
        :param IC_predict:
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

        start_online = time.time()
        self.T_test = T_end_test
        self.ic_predict = IC_predict
        self.train_frac = fraction_train
        ### For all samples (train + validation OR train + validation + test)
        ### integrate the LANDO prediction f(x) to compute the dynamics until T_end_test
        ### Typically we can have: T_end_test > horizon_train
        lando_dynamics = []
        true_dynamics = []
        pbar = tqdm(total=self.param_samples.shape[0], desc=f"Online Phase -> Prediction of f(x, t*; mu)...")
        for i, sample in enumerate(self.param_samples):
            ### This is to compare with the predicted value

            # X, _ = self.X_Snapshot(params=sample, T=self.T_test, num_sensors=self.num_sensors,
            #                        x0=self.ic_predict[0], y0=self.ic_predict[1])
            #
            # true_dynamics.append(X)

            def Model_General(t, z):
                """
                Approximation of f(x) = dxdt produced by LANDO framework
                f(x) needs to be integrated, to obtain the state of the system "x"
                """
                x0, x1 = z
                x = np.array([[x0], [x1]])

                return (self.W_tildes[i] @ self.kernel(self.SparseDicts_all[i], self.scaled_X_all[i] * x)).flatten()

            x_pred = Predict(model=Model_General, Tend=self.T_test, IC=self.ic_predict,
                             sensors=self.num_sensors)
            lando_dynamics.append(x_pred)

            pbar.update()
        pbar.close()
        end_online = time.time()

        interp_model, X_train, y_train, X_valid, y_valid, rel_err_train, rel_err_valid = self.train_fnn(
            fnn_depth=fnn_depth,
            fnn_width=fnn_width,
            fraction_train=self.train_frac,
            lando_dynamics=lando_dynamics,
            epochs=epochs,
            verbose=verb)

        print(f"Execution time of online phase: {end_online - start_online}")

        return interp_model, X_train, y_train, X_valid, y_valid, rel_err_train, rel_err_valid

    def TestPhase(self, num_samples_test, interp_model, x_train, y_train,
                  reconstruction_relative_errors, directory_1d, visuals=True):



        ### First, we generate the test data --> new parameters to test for the specific T_end_test (t*)
        param_samples_lh_test = LatinHypercube(dim_sample=self.num_params_varied, low_bounds=self.low_bounds,
                                               upp_bounds=self.upp_bounds, num_samples=num_samples_test)

        params_not_varied_test = self.params_fixed
        params_not_varied_test = np.tile(params_not_varied_test, (num_samples_test, 1))

        param_samples_test = np.concatenate((param_samples_lh_test, params_not_varied_test), axis=1)

        ### For each parameter value in the test set, compute the true dynamics for comparison
        true_dynamics_test = []
        for val, sample_test in enumerate(param_samples_test):
            X, _ = self.X_Snapshot(params=sample_test, T=self.T_test, num_sensors=self.num_sensors,
                                   x0=self.ic_predict[0], y0=self.ic_predict[1])

            ### The prediction is made only for t*
            true_dynamics_test.append(X[:, -1])

        ### Number of samples for test
        TestSamples = num_samples_test
        X_test = param_samples_test[:, :self.num_params_varied]
        if self.num_params_varied > 1:
            scaler = preprocessing.MaxAbsScaler()
            X_test = scaler.fit_transform(X_test)

        y_test = np.vstack([true_dynamics_test[i] for i in range(TestSamples)])

        start_test = time.time()
        ### For all unseen test parameters, evaluate the neural network after training and approximate f(x,t*;mu*)
        interp_model.eval()
        prediction = interp_model(torch.from_numpy(X_test).to(torch.float32))

        end_test = time.time()

        mean_relative_errors, std_relative_errors = self.relative_error(y_test=y_test,
                                                                        prediction=prediction,
                                                                        tensor=True)

        print(f"NN mean test error: {mean_relative_errors}, {TestSamples} test samples")
        print(f"Execution time for testing: {(end_test-start_test)/X_test.shape[0]}")

        if visuals:

            if self.num_params_varied == 1:

                mean_train_error, mean_test_error, std_test_error = self.Visual_1D(x_train=x_train, y_train=y_train,
                                                                                   x_test=X_test,
                                                                                   y_test=y_test,
                                                                                   interp_model=interp_model,
                                                                                   directory=directory_1d)
            else:

                test_samples = param_samples_lh_test
                train_samples = self.param_samples[:int(self.train_frac * self.num_samples), :self.num_params_varied]
                lando_error = reconstruction_relative_errors[:int(self.train_frac * self.num_samples)]

                mean_train_error, mean_test_error, std_test_error = self.Visual_2D(x_train=x_train, x_test=X_test,
                                                                                   y_train=y_train,
                                                                                   y_test=y_test,
                                                                                   interp_model=interp_model,
                                                                                   test_samples=test_samples,
                                                                                   train_samples=train_samples,
                                                                                   lando_error=lando_error,
                                                                                   IC_predict=self.ic_predict,
                                                                                   T_end_test=self.T_test)

        else:

            if self.num_params_varied == 1:

                ### Error of the training set
                X_train_sort = np.sort(x_train, axis=0)
                sort_index = np.squeeze(np.argsort(x_train, axis=0))
                y_train_sort = y_train[sort_index, :]

                y_final_pred_train = interp_model(torch.from_numpy(X_train_sort).to(torch.float32))
                mean_train_error, _ = self.relative_error(y_train_sort, y_final_pred_train, tensor=True)

                ### Error of the test set
                X_test_sort = np.sort(X_test, axis=0)
                sort_index = np.squeeze(np.argsort(X_test, axis=0))
                y_test_sort = y_test[sort_index, :]

                y_final_pred = interp_model(torch.from_numpy(X_test_sort).to(torch.float32))
                mean_test_error, std_test_error = self.relative_error(y_test_sort, y_final_pred, tensor=True)

            else:
                y_final_pred_train = interp_model(torch.from_numpy(x_train).to(torch.float32))
                mean_train_error = self.relative_error(y_train, y_final_pred_train, tensor=True)

                y_final_pred_test = interp_model(torch.from_numpy(X_test).to(torch.float32))
                mean_test_error, std_test_error = self.relative_error(y_test, y_final_pred_test, tensor=True)

        return mean_train_error, mean_test_error, std_test_error
