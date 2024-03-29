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


### The autoencoder is not used yet. I've just written the code for it, but I do not use it
class Autoencoder(torch.nn.Module):
    def __init__(self, depth, input_dim, latent_dim):
        super().__init__()
        self.depth = depth
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        layers_encoder = []
        prev_dim = self.input_dim

        for i in range(self.depth):
            next_dim = prev_dim // 4 if i != self.depth - 1 else latent_dim
            layers_encoder.append(nn.Linear(prev_dim, next_dim))
            layers_encoder.append(nn.ReLU())
            prev_dim = next_dim

        self.encoder = nn.Sequential(*layers_encoder)

        layers_decoder = []
        prev_dim = latent_dim
        for i in range(self.depth):
            next_dim = prev_dim * 4 if i != self.depth - 1 else input_dim
            layers_decoder.append(nn.Linear(prev_dim, next_dim))
            layers_decoder.append(nn.ReLU())
            prev_dim = next_dim

        self.decoder = nn.Sequential(*layers_decoder)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class ParametricLANDO:

    def __init__(self, kernel, horizon_train, dt, sparsity_tol, directory_processed_data, directory_samples,
                 scale_data_fnn, train_frac):
        self.kernel = kernel
        self.T_end_train = horizon_train
        self.sparsity_tol = sparsity_tol
        self.directory_data = directory_processed_data
        self.directory_samples = directory_samples
        self.scale_fnn = scale_data_fnn
        self.dt = dt  # Timesteps that progress the simulation

        self.samples = np.loadtxt(self.directory_samples + "/samples_heat_eq.txt")
        num_samples = self.samples.shape[0]

        ### Number of training and test samples
        self.param_train_samples = int(train_frac * self.samples.shape[0])
        self.param_test_samples = num_samples - self.param_train_samples

        ### The actual index of training and test_samples in the self.samples matrix
        self.train_samples = np.arange(num_samples)[:self.param_train_samples]
        self.test_samples = np.arange(num_samples)[self.param_train_samples:]

    @staticmethod
    def SVD(num_components, snapshot_mat):
        print("Initializing SVD for stacked snapshot matrix X1")
        snapshot_matrix = snapshot_mat

        # compute and subtract the mean (mean centered data/everything to the center of the mean)
        mean_X = np.expand_dims(np.mean(snapshot_matrix, axis=1), axis=1)
        X_hat = snapshot_matrix - mean_X
        # split in training, validation and test cases

        U, s, Vh = np.linalg.svd(X_hat)
        U_red = U[:, :num_components]
        s_selected = s[:num_components]

        system_energy = np.sum(s_selected) / np.sum(s)

        print(f"SVD completed with {system_energy} % of the system energy explained")
        print(f"The most informative singular values are {s_selected}")
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
        rank_r_SVD_error = np.sum(s[num_components:] ** 2) / np.linalg.norm(snapshot_matrix, ord='fro') ** 2
        POD_projection_error = 1 - np.sum(s[:num_components] ** 2) / np.sum(s ** 2)

        # Print the LS relative error of rank-r SVD approximation. This error is related to the fraction of kinetic energy
        # that is missing in the approximation of the snapshot matrix
        print(f"The relative rank-r SVD approximation error is {rank_r_SVD_error * 100}%")
        print(
            f"The relative error of POD projection is {POD_projection_error * 100}%\nThis is "
            f"the cumulative energy not captured by the projection")

        return U_red, X_hat

    @staticmethod
    def train_autoencoder(ae_depth, ae_input_dim, ae_latent_dim, X1, ae_epochs):
        ### Initialize the model

        ae_model = Autoencoder(depth=ae_depth, input_dim=ae_input_dim, latent_dim=ae_latent_dim)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(ae_model.parameters(), lr=1e-3)

        x1_mat = X1
        x_train = torch.utils.data.DataLoader(dataset=x1_mat,
                                              batch_size=int(x1_mat.shape[0] * 0.45))

        losses = []
        outputs = []
        for epoch in range(ae_epochs):
            for x, _ in x_train:
                ### Outputs of the model
                encoded, reconstructed = ae_model(x)

                loss = loss_fn(reconstructed, x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss)
                outputs.append((encoded, reconstructed))

        return losses, outputs

    def fnn_train(self, reduced_output, fnn_depth, fnn_width, reduced_state_mat, fnn_epochs):

        ###  First set up the neural network to learn the mapping
        Mapping_FNN = FNN(num_input=self.samples.shape[1],
                          num_output=reduced_output, depth=fnn_depth, width=fnn_width)

        optimizer = torch.optim.Adam(Mapping_FNN.parameters(), lr=1e-4)
        loss_criterion = torch.nn.MSELoss()

        # These are fed to the Dataloader, and then to the NN for the training

        X_train = self.samples[self.train_samples]
        y_train = reduced_state_mat

        # Construct the datasets
        dataset_train = Data(X=X_train, y=y_train)

        train_loader = DataLoader(dataset=dataset_train, batch_size=int(len(self.train_samples) * 0.45))

        # Now we train the model
        pbar = tqdm(total=fnn_epochs, desc="Epochs training...")
        loss_epochs = []
        best_train_loss = float('inf')
        best_model_weights = None

        for epoch in range(fnn_epochs):
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
                relative_error_train.append(np.linalg.norm(y - y_pred.detach().numpy()) / np.linalg.norm(y))

            # compute the relative error of the epoch
            loss_epoch = np.mean(relative_error_train)
            loss_epochs.append(loss_epoch)

            # Keep track of the model that results to the minimum validation error
            if loss_epoch < best_train_loss:
                best_train_loss = loss_epoch
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

            print(f"Epoch   Training\n"
                  f"{epoch}   {loss_epoch}\n"
                  f"====================================================")

            pbar.update()
        pbar.close()
        print("Done training!")

        # Plot the losses
        plt.semilogy(loss_epochs, label='Training error')
        plt.xlabel("# Epochs")
        plt.ylabel("Relative MSE")
        plt.legend()
        plt.show()

        if best_model_weights:
            Mapping_FNN.load_state_dict(best_model_weights)

        return Mapping_FNN, loss_epochs

    def Predict2FreeFem(self, predictions):
        """
        Function that transforms the data from python to FreeFem format.
        This is important, to later visualise the results as .vtk files in Paraview
        :param predictions: The predicted states of the system for different parameter values, ready to be visualised
        :return:
        """
        vertices = predictions[0].shape[0]
        directory = self.directory_samples.replace("/Parameter_Samples", "") + '/Predicted_Data'

        for i, vector in enumerate(predictions):

            filename = f"predict_test_{i}.txt"
            filepath = os.path.join(directory, filename)
            with open(filepath, 'w') as f:
                f.write(str(vertices) + '\t\n')
                for i in range(0, len(vector), 5):
                    row = vector[i:i + 5]
                    row_str = '\t'.join(map(str, row))
                    f.write('\t' + row_str + '\t\n')

    def OfflinePhase(self):
        self.SparseDicts_all = []
        self.scaled_X_all = []
        X_perm_all = []
        Y_vals = []
        Y_perm_all = []
        X_all = []
        train_horizon = int(self.T_end_train / self.dt)
        ### Load the data (for dynamics up to t = t_train)
        ### For each parameter sample in the training data, perform the LANDO framework
        pbar = tqdm(total=len(self.train_samples), desc=f"Offline Phase -> Generation of training data...")
        for i in self.train_samples:
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
            f"Training Reduced Data: The mean relative reconstruction errors FOR Y are: {reconstruction_relative_errors}")

        return np.mean(reconstruction_relative_errors)

    def OnlinePhase(self, T_end_test, fnn_depth, fnn_width, epochs):
        ### For each parameter in the training set run the LANDO algorithm to compute the dynamics until T_end_test (t*)
        ### T_end_test = t* > horizon_train = T_end_train_
        t_instance = int(T_end_test / self.dt)

        pbar = tqdm(total=len(self.train_samples), desc=f"Online Phase -> Prediction of f(x, t*; mu)...")
        X_lando = []
        X_lando_all = []
        X_mi_t = [ ]
        X_true_all = []
        for i in self.train_samples:
            X_true = np.load(self.directory_data + f"/sample{i}.npy")[:, :t_instance]

            ### This corresponds to the last column of the snapshot matrix, i.e. the snapshot for t = t*
            X_mi_t.append(X_true[:, -1])
            X_true_all.append(X_true)

            def model(x):
                return self.W_tildes[i] @ self.kernel(self.SparseDicts_all[i], self.scaled_X_all[i] * x)

            ### Compute the reduced training snapshots by integrating the LANDO framework
            ### Compute the snapshots up to t=t*
            x_pred_lando = Predict(model=model, dt=self.dt, comp=X_true, Tend=T_end_test,
                                   IC=X_true[:, 0], type="Discrete")
            X_lando_all.append(x_pred_lando)

            ### Only fot t = t* again
            X_lando.append(x_pred_lando[:, -1])

            pbar.update()
        pbar.close()

        ### sanity check --> compute reconstruction error (x_true - x_pred) to make sure it is small.
        ### This is just for t = t*
        reconstruction_relative_errors = [
            np.linalg.norm(X_mi_t[i] - X_lando[i]) / np.linalg.norm(X_mi_t[i])
            for i in range(len(X_mi_t))]

        ### Now this is for all t
        reconstruction_errors_all = [np.linalg.norm(X_true_all[i] - X_lando_all[i]) /
                                     np.linalg.norm(X_true_all[i]) for i in self.train_samples]

        print(f"The mean relative reconstruction errors FOR X_t* are: {reconstruction_relative_errors}")

        print(f"The mean relative reconstruction errors FOR ALL TIMESTEPS OF X are: {reconstruction_errors_all}")

        ### Perform the SVD to the X_lando matrix

        X_mi_lando = np.vstack(X_lando)
        self.U, _ = self.SVD(num_components=20, snapshot_mat=X_mi_lando)
        X_reduced_mi_lando = self.U.T @ X_mi_lando

        ### Now we employ the NN to learn the mapping from parameter space to the reduced state of the system.

        I_Mapping, _ = self.fnn_train(reduced_output=X_reduced_mi_lando.shape[0], fnn_depth=fnn_depth,
                                      fnn_width=fnn_width, reduced_state_mat=X_reduced_mi_lando, fnn_epochs=epochs)

        ### Now, we evaluate the mapping on the test samples
        test_samples = self.samples[self.test_samples]
        errors_parametric_prediction = []
        x_fom_predictions = []
        for i in self.test_samples:
            x_true = np.load(self.directory_data + f"/sample{i}.npy")[:, t_instance]

            sample = self.samples[i, :]
            x_pred_reduced = I_Mapping(torch.tensor(sample, dtype=torch.float32)).detach().numpy()
            x_pred_fom = self.U @ x_pred_reduced
            x_fom_predictions.append(x_pred_fom)

            relative_error_pred = np.linalg.norm(x_true - x_pred_fom) / np.linalg.norm(x_true)
            errors_parametric_prediction.append(relative_error_pred)

        ### Create directory to store prediction data

        try:
            os.mkdir(self.directory_samples.replace("/Parameter_Samples", "") + '/Predicted_Data/')
            os.mkdir(self.directory_samples.replace("/Parameter_Samples", "") + '/Predicted_Data_vtk/') ### for vtk visuals
        except:
            print('Sth went wrong, please check')

        ## Transform the prediction files into FreeFem format
        self.Predict2FreeFem(predictions=x_fom_predictions)

        return errors_parametric_prediction, test_samples


parametric_lando = ParametricLANDO(kernel=gauss_kernel, horizon_train=2.2, dt=0.01, sparsity_tol=1e-4,
                                   directory_processed_data="/Users/konstantinoskevopoulos/Documents/Heat_Eq_Thesis"
                                                            "/SnapshotData_Processed",
                                   directory_samples="/Users/konstantinoskevopoulos/Documents/Heat_Eq_Thesis"
                                                     "/Parameter_Samples",
                                   scale_data_fnn=False, train_frac=0.05)

meanerr = parametric_lando.OfflinePhase()
print(f"The mean error of reconstruction over all the samples is {meanerr}")

parametric_rel_err, test_param_samples = parametric_lando.OnlinePhase(T_end_test=4.2, fnn_depth=2, fnn_width=150,
                                                                      epochs=3000)

print(f"The relative errors for the unseen parameters are: {parametric_rel_err}")
print(f"The mean relative error for {test_param_samples.shape[0]} unseen parameters is {np.mean(parametric_rel_err)}")






