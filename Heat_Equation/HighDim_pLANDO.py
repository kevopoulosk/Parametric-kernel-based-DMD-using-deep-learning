import numpy as np

from Lotka_Volterra.Sparse_Dictionary_Learning import *
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch
import matplotlib.pyplot as plt
from torch import nn
import os


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


class ParametricLANDO:

    def __init__(self, kernel, horizon_train, dt, sparsity_tol, directory_processed_data, directory_samples,
                 training_frac, validation_frac):
        self.kernel = kernel
        self.T_end_train = horizon_train
        self.sparsity_tol = sparsity_tol
        self.directory_data = directory_processed_data
        self.directory_samples = directory_samples
        self.dt = dt  # Timesteps that progress the simulation

        self.samples = np.loadtxt(self.directory_samples + "/samples_heat_eq.txt")
        self.num_samples = self.samples.shape[0]

        ### Number of training, validation and test samples
        self.num_train_samples = int(training_frac * self.samples.shape[0])
        self.num_valid_samples = int(validation_frac * self.samples.shape[0])
        self.num_test_samples = self.num_samples - self.num_train_samples - self.num_valid_samples

        ### The actual index of training, validation and test_samples in the self.samples matrix
        self.train_samples = np.arange(self.num_samples)[:self.num_train_samples]
        self.valid_samples = np.arange(self.num_samples)[
                             self.num_train_samples:(self.num_train_samples + self.num_valid_samples)]
        self.test_samples = np.arange(self.num_samples)[(self.num_train_samples + self.num_valid_samples):]

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

    @staticmethod
    def SVD(num_components, snapshot_mat):
        print("Initializing SVD for stacked snapshot matrix X1")

        # compute and subtract the mean (mean centered data/everything to the center of the mean)
        # split in training, validation and test cases

        U, s, Vh = np.linalg.svd(snapshot_mat)
        U_red = U[:, :num_components]
        s_selected = s[:num_components]

        system_energy = np.sum(s_selected) / np.sum(s)

        print(f"SVD completed with {system_energy} % of the system energy explained")
        print(f"The number of singular values used (truncation threshold) is {num_components}")

        # plot of the explained variance of the system
        plt.figure()
        plt.plot(np.cumsum(s), "-o", label="Singular Values")
        plt.ylabel("Explained variance ratio")
        plt.xlabel("Singular values")
        plt.legend()
        plt.title("Explained variance of the system")
        plt.show()
        # plot the singular values
        plt.figure()
        plt.semilogy(s, "-o", label='Singular Values')
        plt.ylabel("Value of each singular value")
        plt.xlabel("Singular values")
        plt.legend()
        plt.title("Singular values")
        plt.show()
        print(f"SVD for stacked snapshot matrix X1 finished")
        # compute the "alpha" projection coefficient of the dataset
        rank_r_SVD_error = np.sum(s[num_components:] ** 2) / np.linalg.norm(snapshot_mat, ord='fro') ** 2
        POD_projection_error = 1 - np.sum(s[:num_components] ** 2) / np.sum(s ** 2)

        # Print the LS relative error of rank-r SVD approximation. This error is related to the fraction of kinetic energy
        # that is missing in the approximation of the snapshot matrix
        print(f"The relative rank-r SVD approximation error is {rank_r_SVD_error * 100}%")
        print(
            f"The relative error of POD projection is {POD_projection_error * 100}%\nThis is "
            f"the cumulative energy not captured by the projection")

        return U_red

    def train_fnn(self, reduced_output, fnn_depth, fnn_width, reduced_state_mat, epochs, batch_size_frac, verbose):

        ###  First set up the neural network to learn the mapping
        Mapping_FNN = FNN(num_input=self.samples.shape[1],
                          num_output=reduced_output, depth=fnn_depth, width=fnn_width)

        optimizer = torch.optim.Adam(Mapping_FNN.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.04, patience=3)
        loss_criterion = torch.nn.MSELoss()

        ### Split the parametric samples into training and validation data
        TrainSamples = self.num_train_samples
        ValidSamples = self.num_valid_samples
        TestSamples = self.num_test_samples

        ###The actual parametric samples
        X_train = self.samples[self.train_samples, :]
        X_valid = self.samples[self.valid_samples, :]
        X_test = self.samples[self.test_samples, :]

        y_train = reduced_state_mat[:, self.train_samples].T
        y_valid = reduced_state_mat[:, self.valid_samples].T
        y_test = reduced_state_mat[:, self.test_samples].T

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

            ### Stop the training process if validation error < 0.7%
            if mean_relative_err_val < 0.006:
                print(f"Stopping early at epoch {epoch}, since mean_relative_err_val < 0.006")
                break

            ### Reduce the learning rate when we have validation error < 0.95%
            if mean_relative_err_val < 0.0065:
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

        print(f"FNN Training, Mean test error, {TestSamples} samples: {mean_relative_error}")

        return Mapping_FNN, X_train, y_train, X_test, y_test, mean_relative_error

    def Predict2FreeFem(self, predictions, T_test):
        """
        Function that transforms the data from python to FreeFem format.
        This is important, to later visualise the results as .vtk files in Paraview
        :param predictions: The predicted states of the system for different parameter values, ready to be visualised
        :return:
        """
        vertices = predictions[0].shape[0]
        directory = self.directory_samples.replace("/Parameter_Samples", "") + f'/Predicted_Data/t_test = {T_test}'

        for i, vector in enumerate(predictions):

            filename = f"predict_test_{i}.txt"
            filepath = os.path.join(directory, filename)
            with open(filepath, 'w') as f:
                f.write(str(vertices) + '\t\n')
                for i in range(0, len(vector), 5):
                    row = vector[i:i + 5]
                    row_str = '\t'.join(map(str, row))
                    f.write('\t' + row_str + '\t\n')

    def Test_Eval(self, model_interp, test_instance):

        ### First, we evaluate the mapping on the test samples
        test_samples = self.samples[self.test_samples]
        errors_parametric_prediction = []
        x_fom_predictions = []
        for i in self.test_samples:
            x_true = np.load(self.directory_data + f"/sample{i}.npy")[:, test_instance]

            sample = self.samples[i, :]
            x_pred_reduced = model_interp(torch.tensor(sample, dtype=torch.float32)).detach().numpy()
            x_pred_fom = self.U @ x_pred_reduced
            x_fom_predictions.append(x_pred_fom)

            relative_error_pred = np.linalg.norm(x_true - x_pred_fom) / np.linalg.norm(x_true)
            errors_parametric_prediction.append(relative_error_pred)

        print(
            f"Mean Test Error Prediction VS Ground Truth, {len(test_samples)} samples: {np.mean(errors_parametric_prediction)}")

        return x_fom_predictions

    def Visual_2D(self, x_train, x_test, model_interp, T_end_test, lando_predict_rel_errors, test_instance):

        y_train = []
        for i in self.train_samples:
            true_state = np.load(self.directory_data + f"/sample{i}.npy")[:, test_instance]
            y_train.append(true_state)
        y_train = np.vstack(y_train)

        y_test = []
        for i in self.test_samples:
            true_state = np.load(self.directory_data + f"/sample{i}.npy")[:, test_instance]
            y_test.append(true_state)
        y_test = np.vstack(y_test)

        y_final_pred_train = self.U @ model_interp(torch.from_numpy(x_train).to(torch.float32)).detach().numpy().T
        train_error = self.relative_error(y_train, y_final_pred_train.T, mean=False, tensor=False)

        y_final_pred_test = self.U @ model_interp(torch.from_numpy(x_test).to(torch.float32)).detach().numpy().T
        test_error = self.relative_error(y_test, y_final_pred_test.T, mean=False, tensor=False)

        directory = f"/Users/konstantinoskevopoulos/Desktop/Heat_eq_Results/t_test={T_end_test}/"

        ### Visualise the performance for training
        plt.figure()
        plt.scatter(x_train[:, 0], x_train[:, 1], c=train_error, cmap="plasma")
        plt.xlabel(r"$\mu_1 = D$")
        plt.ylabel(r"$\mu_2 = \alpha$")
        filename = "Training.png"
        plt.colorbar(label=r'$L_2$ relative error')
        plt.savefig(directory + filename)
        plt.show()

        ### Testing performance
        plt.figure()
        plt.scatter(x_test[:, 0], x_test[:, 1], c=test_error, cmap="plasma")
        plt.xlabel(r"$\mu_1 = D$")
        plt.ylabel(r"$\mu_2 = \alpha$")
        filename = "Test.png"
        plt.colorbar(label=r'$L_2$ relative error')
        plt.savefig(directory + filename)
        plt.show()

        ### LANDO Vs Ground Truth
        lando_error_train_samples = lando_predict_rel_errors[:len(self.train_samples)]
        plt.figure()
        plt.scatter(x_train[:, 0], x_train[:, 1], c=lando_error_train_samples, cmap="plasma")
        plt.xlabel(r"$\mu_1 = D$")
        plt.ylabel(r"$\mu_2 = \alpha$")
        filename = "LANDO_Vs_Truth_TRAIN.png"
        plt.colorbar(label=r'LANDO $L_2$ relative error')
        plt.savefig(directory + filename)
        plt.show()

        lando_error_test_samples = lando_predict_rel_errors[(len(self.train_samples) + len(self.valid_samples)):]
        plt.figure()
        plt.scatter(x_test[:, 0], x_test[:, 1], c=lando_error_test_samples, cmap="plasma")
        plt.xlabel(r"$\mu_1 = D$")
        plt.ylabel(r"$\mu_2 = \alpha$")
        filename = "LANDO_Vs_Truth_TEST.png"
        plt.colorbar(label=r'LANDO $L_2$ relative error')
        plt.savefig(directory + filename)
        plt.show()

        return np.mean(train_error), np.mean(test_error), np.std(train_error), np.std(test_error)

    def LANDO_Vs_GrTr(self, X_true_all, X_lando_all):

        train_horizon = int(self.T_end_train / self.dt)
        train_x_true = [X_true_all[i][:, :train_horizon] for i in range(len(X_true_all))]
        train_x_lando = [X_lando_all[i][:, :train_horizon] for i in range(len(X_lando_all))]

        train_error = [np.linalg.norm(train_x_true[i] - train_x_lando[i]) /
                       np.linalg.norm(train_x_true[i]) for i in range(self.num_samples)]

        ### Extrapolation in time
        extrap_x_true = [X_true_all[i][:, train_horizon:] for i in range(len(X_true_all))]
        extrap_x_lando = [X_lando_all[i][:, train_horizon:] for i in range(len(X_lando_all))]

        extrap_error = [np.linalg.norm(extrap_x_true[i] - extrap_x_lando[i]) /
                        np.linalg.norm(extrap_x_true[i]) for i in range(self.num_samples)]

        return train_error, extrap_error

    def OfflinePhase(self):
        self.SparseDicts_all = []
        self.scaled_X_all = []
        X_perm_all = []
        Y_vals = []
        Y_perm_all = []
        X_all = []
        train_horizon = int(self.T_end_train / self.dt)
        ### Load the data (for dynamics up to t = t_train)
        ### For all training + validation + testing parameters, perform the LANDO framework
        pbar = tqdm(total=self.num_samples, desc=f"Offline Phase -> Generation of training data...")
        for i in range(self.num_samples):
            ### In this case, we have a discrete-time system
            X = np.load(self.directory_data + f"/sample{i}.npy")[:, :train_horizon][:, :-1]
            Y = np.load(self.directory_data + f"/sample{i}.npy")[:, :train_horizon][:, 1:]

            ### Perform LANDO up to t = t_train for all the training samples
            X_all.append(X)

            scaledX = Scale(X)

            Xperm, perm = Permute(X)
            Yperm = Y[:, perm]

            SparseDict, _, _, _ = SparseDictionary(Xperm, scaledX, self.kernel,
                                                   tolerance=self.sparsity_tol, pbar_bool=False)
            self.SparseDicts_all.append(SparseDict)

            self.scaled_X_all.append(scaledX)
            X_perm_all.append(Xperm)
            Y_perm_all.append(Yperm)

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

        ### For sanity check, to ensure that everything is going well
        ### compute reconstruction error (Y - f(x)) to make sure it is small
        reconstruction_relative_errors = [np.linalg.norm(Y_vals[i] - models[i]) / np.linalg.norm(Y_vals[i])
                                          for i in range(len(Y_vals))]

        print(
            f"Mean LANDO training error, {self.num_samples} samples: {np.round(np.mean(reconstruction_relative_errors), decimals=7) * 100}%")

        return np.mean(reconstruction_relative_errors), self.SparseDicts_all

    def OnlinePhase(self, T_end_test, trunk_rank, fnn_depth, fnn_width, batch_size, epochs, verbose=True):
        ### For all parameters in train + test + validation set run the LANDO algorithm to compute the dynamics until T_end_test (t*)
        ### T_end_test = t* > horizon_train = T_end_train_
        test_instance = int(T_end_test / self.dt)

        X_lando = []
        X_lando_all = []
        X_true = []
        X_true_all = []
        pbar = tqdm(total=self.num_samples, desc=f"Online Phase -> Prediction of f(x, t*; mu)...")
        for i in range(self.num_samples):
            X_true_mi = np.load(self.directory_data + f"/sample{i}.npy")[:, :test_instance]

            ### Collect the snapshots for the test instances
            X_true.append(X_true_mi[:, -1])
            X_true_all.append(X_true_mi)

            def model(x):
                return self.W_tildes[i] @ self.kernel(self.SparseDicts_all[i], self.scaled_X_all[i] * x)

            ### Compute the reduced training snapshots by integrating the LANDO framework
            ### Compute the snapshots up to t=t*
            x_pred_lando = Predict(model=model, dt=self.dt, comp=X_true_mi, Tend=T_end_test,
                                   IC=X_true_mi[:, 0], type="Discrete")
            X_lando_all.append(x_pred_lando)

            ### Only fot t = t* again
            X_lando.append(x_pred_lando[:, -1])

            pbar.update()
        pbar.close()

        ### sanity check --> compute reconstruction error (x_true - x_pred) to make sure it is small.
        ### This is just for t = t*
        reconstruction_relative_errors = [
            np.linalg.norm(X_true[i] - X_lando[i]) / np.linalg.norm(X_true[i])
            for i in range(len(X_true))]

        ### Now this is for all t
        reconstruction_errors_all = [np.linalg.norm(X_true_all[i] - X_lando_all[i]) /
                                     np.linalg.norm(X_true_all[i]) for i in self.train_samples]

        print(f"Mean Relative Prediction Errors FOR X_t*: {reconstruction_relative_errors}")

        print(f"Mean Relative Prediction Errors FOR ALL TIMESTEPS OF X: {reconstruction_errors_all}")

        ### Perform the SVD to the X_lando matrix
        ### The reduced matrix contains all the parametric samples (training + test +validation)
        X1_lando = np.vstack(X_lando).T

        self.U = self.SVD(trunk_rank, X1_lando)
        X_reduced_mi_lando = self.U.T @ X1_lando

        ### Now we employ the NN to learn the mapping from parameter space to the reduced state of the system.

        I_Mapping, x_train, y_train, x_test, y_test, _ = self.train_fnn(reduced_output=X_reduced_mi_lando.shape[0],
                                                                        fnn_depth=fnn_depth,
                                                                        fnn_width=fnn_width,
                                                                        reduced_state_mat=X_reduced_mi_lando,
                                                                        epochs=epochs, batch_size_frac=batch_size,
                                                                        verbose=verbose)

        ### Now, we evaluate the mapping on the test samples
        x_fom_predictions = self.Test_Eval(model_interp=I_Mapping, test_instance=test_instance)

        mean_train_error, mean_test_error, std_train_error, std_test_error = self.Visual_2D(x_train=x_train,
                                                                                            x_test=x_test,
                                                                                            model_interp=I_Mapping,
                                                                                            T_end_test=T_end_test,
                                                                                            lando_predict_rel_errors=reconstruction_relative_errors,
                                                                                            test_instance=test_instance)

        lando_train, lando_extrap = self.LANDO_Vs_GrTr(X_true_all=X_true_all, X_lando_all=X_lando_all)

        ### Create directory to store prediction data
        try:
            os.mkdir(
                self.directory_samples.replace("/Parameter_Samples", "") + f'/Predicted_Data/t_test = {T_end_test}')
            ### for vtk visuals
            os.mkdir(
                self.directory_samples.replace("/Parameter_Samples",
                                               "") + f'/Predicted_Data_vtk/t_test = {T_end_test}')
        except:
            print('Sth went wrong, please check')

        ### Transform the prediction files into FreeFem format
        self.Predict2FreeFem(predictions=x_fom_predictions, T_test=T_end_test)

        return [mean_train_error, mean_test_error, std_train_error, std_test_error, lando_train, lando_extrap]


