import numpy as np

from MIONet_construction_test import *


# Generate training and test data
TrainSamples = 100
TestSamples = 50
sensors = 300
x_dim = 1
depth = 4
width = 200
# Generate the training data
kernels_train, models_train, _ = LANDO(kernel_choice=quadratic_kernel, T_end=400, tol=1e-6, num_samples=TrainSamples,
                                       train_test="training")

np.savez("Kernels_Train.npz", kernels=kernels_train, models=models_train)
data_set_train = Data(m_sensors=sensors, T_end=400, kernels=kernels_train, models=models_train)

# Generate the test data
kernels_test, models_test, _ = LANDO(kernel_choice=quadratic_kernel, T_end=400, tol=1e-6, num_samples=TestSamples,
                                     train_test="test")
np.savez("Kernels_Test.npz", kernels=kernels_test, models=models_test)
data_set_test = Data(m_sensors=sensors, T_end=400, kernels=kernels_test, models=models_test)


# Call MIONet for the actual model training
model = MIONet(num_branch=kernels_train[0].shape[0], kernel_vector_input=sensors, trunk_input=x_dim,
               depth=depth, width=width, feature_expansion=True, num_features=3)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_criterion = torch.nn.MSELoss()
train_loader = DataLoader(dataset=data_set_train, batch_size=sensors)
test_loader = DataLoader(dataset=data_set_test, batch_size=sensors)

# Train the model
epochs = 500

# Train the model

pbar = tqdm(total=epochs, desc="Epochs training...")
loss_epochs = []
t = np.linspace(0, 400, sensors)
for epoch in range(epochs):
    losses = []
    for x, y in train_loader:
        y_pred = model(x)
        # plt.plot(t, y, label="true")
        # plt.plot(t, x[0][:sensors], label='1st kernel')
        # plt.plot(t, x[0][300:600], label='2nd kernel')
        # plt.plot(t, x[0][600:900], label='3rd kernel')
        # #plt.plot(t, y_pred.detach().numpy(), label='prediction')
        # plt.legend()
        # plt.show()
        loss = loss_criterion(y_pred, y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_epoch = np.mean(losses)
    loss_epochs.append(loss_epoch)
    print(f"epoch = {epoch}, mean error = {loss_epoch}")
    pbar.update()
pbar.close()
print("Done training!")

# Plot the losses
plt.plot(loss_epochs)
plt.xlabel("no. of iterations")
plt.ylabel("total loss")
plt.show()

with torch.no_grad():
    for x_test, y_test in test_loader:
        y_pred_test = model(x_test)
        plt.plot(y_test, label='Ground truth', color='black')
        plt.plot(y_pred_test, label='Prediction', color='red')
        plt.legend()
        plt.show()
        print(f"prediction is {y_pred_test}")