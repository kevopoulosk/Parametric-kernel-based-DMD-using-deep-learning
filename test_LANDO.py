from Lotka_Volterra.Sparse_Dictionary_Learning import *

### This file is used just to test the prediction of LANDO( i.e. the transformation from f(x) ---> x)

dt = 0.01
### We train the LANDO up to t = T_train = 2.3
T_train = 2.3

### We make the prediction up to t* = T_end = 4.2
T_end = 4.2
DirectoryProcessed = "/Users/konstantinoskevopoulos/Documents/Heat_Eq_Thesis/SnapshotData_Processed"
train_t_instance = int(T_train / dt)
test_t_instance = int(T_end/dt)

kernel = gauss_kernel

X = np.load(DirectoryProcessed + "/sample38.npy")[:, :train_t_instance][:, :-1]
Y = np.load(DirectoryProcessed + "/sample38.npy")[:, :train_t_instance][:, 1:]

X_comp = np.load(DirectoryProcessed + "/sample38.npy")[:, :test_t_instance]

scaledX = Scale(X)

### randomly permute X and Y, for more efficient dictionary learning
Xperm, perm = Permute(X)
Yperm = Y[:, perm]

Dict, _, _, _ = SparseDictionary(Xperm, scaledX, kernel=kernel, tolerance=1e-4)
print(f"The shape of the sparse dictionary is {Dict.shape}")

W_tilde = Yperm @ np.linalg.pinv(kernel(Dict, scaledX * Xperm))
Model = W_tilde @ kernel(Dict, scaledX * X)

recErr_quad = np.linalg.norm(Y - Model) / np.linalg.norm(Y)
print(f"The reconstruction error using gaussian kernel is {recErr_quad}")


### model to be integrated
def model(x):
    return W_tilde @ kernel(Dict, scaledX * x)


x_pred = Predict(model, dt=dt, IC=X[:, 0], Tend=T_end, comp=X_comp, type="Discrete")

relative_error = np.linalg.norm(X_comp - x_pred) / np.linalg.norm(X_comp)
print(f"The relative error of the reconstruction (x_true - x_pred) is:  {relative_error}")
