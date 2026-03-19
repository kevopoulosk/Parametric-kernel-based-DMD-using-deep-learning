from Parametric_LANDO.Sparse_Dictionary_Learning import *
import matplotlib.pyplot as plt
import torch
from sklearn import preprocessing
from torch import nn
import torch.utils.data as data
from torch.utils.data import DataLoader


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



class ParametricLANDO:

    def __init__(self, param_samples, T_end_train, num_sensors, kernel,
                 sparsity_tol, X_snapshot, Y_deriv,
                 num_params_varied, dofs, batch_size_frac):

        self.param_samples = param_samples  ### (N_samples, 4) for LV case

        self.T_end_train = T_end_train  ### last 't' of training horizon
        self.num_sensors = num_sensors  ### discretization of solution

        ### Sparse dictionary learning settings
        self.kernel = kernel
        self.sparsity_tol = sparsity_tol

        ### Functions to generate the training data
        self.X_Snapshot = X_snapshot
        self.Y_Deriv = Y_deriv

        self.num_params_varied = num_params_varied
        self.dofs = dofs  ### degrees of freedom of the state of the system

        self.batch_size_frac = batch_size_frac

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

        ### Split the parametric samples into training and validation data
        TrainSamples = int(self.param_samples.shape[0] * fraction_train)
        ### Number of samples for validation
        ValidSamples = self.param_samples.shape[0] - TrainSamples

        scaler = preprocessing.MinMaxScaler()
        X_train = scaler.fit_transform(self.param_samples[:TrainSamples, :self.num_params_varied])
        X_valid = scaler.transform(self.param_samples[TrainSamples:, :self.num_params_varied])

        y_train = np.vstack([lando_dynamics[:TrainSamples][i][:, -1] for i in range(TrainSamples)])
        y_valid = np.vstack([lando_dynamics[TrainSamples:][i][:, -1] for i in range(ValidSamples)])

        ### Set up the neural network to learn the mapping
        Mapping_FNN = FNN(num_input=self.num_params_varied, num_output=self.dofs, depth=fnn_depth, width=fnn_width)

        optimizer = torch.optim.Adam(Mapping_FNN.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.04, patience=3)
        loss_criterion = torch.nn.MSELoss()

        dataset_train = Data(X=X_train, y=y_train)
        dataset_valid = Data(X=X_valid, y=y_valid)

        train_loader = DataLoader(dataset=dataset_train, batch_size=int(TrainSamples * self.batch_size_frac), shuffle=True)
        valid_loader = DataLoader(dataset=dataset_valid, batch_size=ValidSamples)

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

        Mapping_FNN.eval()

        prediction_valid = Mapping_FNN(torch.from_numpy(X_valid).to(torch.float32))
        mean_relative_error_valid = self.relative_error(y_test=y_valid, prediction=prediction_valid,
                                                        tensor=True, mean=False)

        return Mapping_FNN, mean_relative_error_valid

    def OfflinePhase(self):

        self.SparseDicts_all = []
        self.scaled_X_all = []
        X_perm_all = []
        Y_vals = []
        Y_perm_all = []
        X_all = []

        ### The sparse dictionaries for all the parametric samples (train + valid OR train+test+valid) are generated
        ### So, for each training sample we have W_tilde, x_tilde(sparse dictionary), and k(x_tilde, x)
        pbar = tqdm(total=self.param_samples.shape[0], desc=f"Offline Phase -> Generation of training data...")
        for val, param_sample in enumerate(self.param_samples):
            X, _ = self.X_Snapshot(params=param_sample, T=self.T_end_train, num_sensors=self.num_sensors)

            Y = self.Y_Deriv(X, *param_sample)

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

        # Compute W tilde and form the model for all the training samples.
        self.W_tildes = [y_perm @ np.linalg.pinv(self.kernel(Sparse_Dict, scaled_X * x_perm))
                         for y_perm, Sparse_Dict, scaled_X, x_perm in
                         zip(Y_perm_all, self.SparseDicts_all, self.scaled_X_all, X_perm_all)]

        return self.W_tildes, self.SparseDicts_all, X_all, Y_vals

    def OnlinePhase(self, T_end_test, IC_predict, train_frac, fnn_depth, fnn_width, epochs, verb):

        T_test = T_end_test
        ic_predict = IC_predict

        ### For all samples (train + validation OR train + validation + test)
        ### integrate the LANDO prediction f(x) to compute the dynamics until T_end_test
        ### Typically we can have: T_end_test > horizon_train
        lando_dynamics = []
        pbar = tqdm(total=self.param_samples.shape[0], desc=f"Online Phase -> Prediction of f(x, t*; mu)...")
        for i, sample in enumerate(self.param_samples):
            def Model_General(t, z):
                """
                Approximation of f(x) = dxdt produced by LANDO framework
                f(x) needs to be integrated, to obtain the state of the system "x"
                """
                x0, x1 = z
                x = np.array([[x0], [x1]])

                return (self.W_tildes[i] @ self.kernel(self.SparseDicts_all[i], self.scaled_X_all[i] * x)).flatten()

            x_pred = Predict(model=Model_General, Tend=T_test, IC=ic_predict,
                             sensors=self.num_sensors)
            lando_dynamics.append(x_pred)

            pbar.update()
        pbar.close()

        interp_model, mean_relative_error_test = self.train_fnn(
            fnn_depth=fnn_depth,
            fnn_width=fnn_width,
            fraction_train=train_frac,
            lando_dynamics=lando_dynamics,
            epochs=epochs,
            verbose=verb)

        return interp_model, mean_relative_error_test
