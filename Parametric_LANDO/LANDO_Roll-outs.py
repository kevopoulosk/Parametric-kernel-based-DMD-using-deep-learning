import optax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from Parametric_LANDO.Sparse_Dictionary_Learning import *
from Allen_Cahn.Allen_Cahn_equation import Allen_Cahn_eq


def relative_error(y_test, prediction):
    """
    Function to calculate mean relative error between two quantites
    :param y_test:
    :param prediction:
    :return:
    """
    err_list = []
    for row in range(y_test.shape[0]):
        err = np.linalg.norm(y_test[row] - prediction[row]) / np.linalg.norm(y_test[row])
        err_list.append(err)

    return np.mean(err_list)


def LANDO(X, Y, kernel, sparsity_tol):
    """
    Function that performs the standard LANDO
    :param X:
    :param Y:
    :param kernel:
    :param sparsity_tol:
    :return:
    """
    scaledX = Scale(X)

    Xperm, perm = Permute(X)
    Yperm = Y[:, perm]

    X_tilde = SparseDictionary(Xperm, scaledX, kernel, tolerance=sparsity_tol, pbar_bool=False)

    W_tilde = Yperm @ np.linalg.pinv(kernel(X_tilde, scaledX * Xperm))

    return X_tilde, W_tilde


def z_func(F, x0, n):
    """
    Function that implements the flow map.
    This is the nested function F(F(F...F(.)))..)
    :param F:
    :param x0:
    :param n:
    :return:
    """
    result_old = x0
    for _ in range(n):
        result = F(result_old)
        result_old = result
    return result


def Rollout_func(x, M, R, reference, rows, cols, X_tilde):
    """
    This is the objective function to be optimized.
    This function implements the rollout
    :param x:
    :param M:
    :param R:
    :param reference:
    :param rows:
    :param cols:
    :param X_tilde:
    :return:
    """
    W_tilde = x.reshape((rows, cols))

    def F(state):
        return W_tilde @ kernel(X_tilde, Scale(X) * state)

    loss = 0
    for i in range(M-R):
        for j in range(1, R):
            ### Here I calculate the actual difference
            rollout_error = reference[:, i+j] - z_func(F, reference[:, i].reshape(-1, 1), j).reshape(-1)
            loss += jnp.linalg.norm(rollout_error)**2

    return loss


def objective(params):
    return Rollout_func(params, X.shape[1], 4, X, W.shape[0], W.shape[1], sparse_dict)


kernel = linear_kernel

X = Allen_Cahn_eq(D=0.00075, a=1)[:, :-1]
Y = Allen_Cahn_eq(D=0.00075, a=1)[:, 1:]


sparse_dict, W = LANDO(X, Y, kernel, sparsity_tol=1e-5)

### Use adam for the optimization
solver = optax.adam(learning_rate=1e-5)
### As initial guess we give the W matrix from LANDO (flattened).
### The optimization is performed with respect to the elements of this matrix
params = jnp.array(W.reshape(-1))
opt_state = solver.init(params)

w_tildes = []
obj = []
w_opt = W
obj_loss = np.inf
### Perform the optimization
pbar = tqdm(total=80, desc="Optimization of the rollout function")
for _ in range(80):
    grad = jax.grad(objective)(params)
    updates, opt_state = solver.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    w_tildes.append(params)
    loss = objective(params)
    print('Objective function: {:.2E}'.format(loss))
    obj.append(objective(params))

    ### Keep the W matrix that results in the smallest error
    if loss < obj_loss:
        obj_loss = loss
        w_opt = params.reshape((W.shape[0], W.shape[1]))
    pbar.update()
pbar.close()


### Plot the objective function
plt.semilogy(obj, '.-')
plt.ylabel(r"$J$")
plt.xlabel("Epochs")
plt.show()

### Form the models
surrogate_lando = W @ kernel(sparse_dict, Scale(X)*X)
surrogate_rollout = w_opt @ kernel(sparse_dict, Scale(X)*X)

### Make prediction for the Y derivative matrix using the two models
error_init = relative_error(Y, surrogate_lando)
error_rollout = relative_error(Y, surrogate_rollout)

print(f'Error LANDO Y:{error_init}\n'
      f'Error Rollout Y: {error_rollout}')


def Surrogate_LANDO(x):
    return W @ kernel(sparse_dict, Scale(X) * x)


def Surrogate_Rollouts(x):
    return w_opt @ kernel(sparse_dict, Scale(X) * x)

### Make prediction for the X snapshot matrix using the two models
X_pred_lando = Predict(Surrogate_LANDO, Tend=1, IC=X[:, 0], dt=1/1000, type='Discrete')
X_pred_rollouts = Predict(Surrogate_Rollouts, Tend=1, IC=X[:, 0], dt=1/1000, type='Discrete')


x_error_init = relative_error(X, X_pred_lando)
x_error_rollout = relative_error(X, X_pred_rollouts)

print(f'Error LANDO X:{x_error_init}\n'
      f'Error Rollout X: {x_error_rollout}')













