import optax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from Parametric_LANDO.Sparse_Dictionary_Learning import *
from Allen_Cahn.Allen_Cahn_equation import Allen_Cahn_eq


class RolloutLANDO:

    def __init__(self, rollout_length, X, Y, kernel, sparsity_tol):
        self.X = X
        self.Y = Y
        self.r_length = rollout_length
        self.kernel = kernel
        self.sparsity_tol = sparsity_tol

    @staticmethod
    def relative_error(y_test, prediction):
        err_list = []
        for row in range(y_test.shape[0]):
            err = np.linalg.norm(y_test[row] - prediction[row]) / np.linalg.norm(y_test[row])
            err_list.append(err)

        return np.mean(err_list)

    def LANDO(self):
        scaledX = Scale(self.X)

        Xperm, perm = Permute(self.X)
        Yperm = self.Y[:, perm]

        self.X_tilde = SparseDictionary(Xperm, scaledX, self.kernel, tolerance=self.sparsity_tol, pbar_bool=False)

        self.W_tilde = Yperm @ np.linalg.pinv(self.kernel(self.X_tilde, scaledX * Xperm))

    def F(self, w_tilde_obj, state):
        return w_tilde_obj @ self.kernel(self.X_tilde, Scale(self.X) * state)

    def Phi_func(self, x0, n, w):
        result = x0
        for _ in range(n):
            result = self.F(w, result)
        return result

    def Rollout_func(self, x, record=False):

        rows = self.W_tilde.shape[0]
        cols = self.W_tilde.shape[1]
        if record:
            loss_record = []

        loss = 0
        for i in range(self.X.shape[1] - self.r_length):
            for j in range(1, self.r_length):
                ### Here I calculate the actual difference
                rollout_error = self.X[:, i + j] - self.Phi_func(self.X[:, i].reshape(-1, 1),
                                                                 j, w=x.reshape((rows, cols))).reshape(-1)
                loss += jnp.square(jnp.linalg.norm(rollout_error))
                if record:
                    loss_record.append(loss)

        if record:
            return loss, loss_record
        else:
            return loss

    def optimize(self, epochs, lr):
        ### Use adam for the optimization
        solver = optax.adam(learning_rate=lr)
        ### As initial guess we give the W matrix from LANDO (flattened).
        ### The optimization is performed with respect to the elements of this matrix
        params = jnp.array(self.W_tilde.reshape(-1, 1))
        opt_state = solver.init(params)

        w_tildes = []
        obj = []
        self.w_opt = self.W_tilde
        obj_loss = np.inf
        ### Perform the optimization
        pbar = tqdm(total=epochs, desc="Optimization of the rollout function")
        for _ in range(epochs):
            grad = jax.grad(self.Rollout_func)(params)
            updates, opt_state = solver.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            w_tildes.append(params)
            loss = self.Rollout_func(params)

            print('Objective function: {:.2E}'.format(loss))
            obj.append(self.Rollout_func(params))

            ### Keep the W matrix that results in the smallest error
            if loss < obj_loss:
                obj_loss = loss
                self.w_opt = params.reshape((self.W_tilde.shape[0], self.W_tilde.shape[1]))
            pbar.update()
        pbar.close()

        ### Plot the objective function
        plt.semilogy(obj, '.-')
        plt.ylabel(r"$J$")
        plt.xlabel("Epochs")
        plt.show()

        plt.plot(self.W_tilde.reshape(-1) - self.w_opt.reshape(-1), label='init', color='red')

        plt.show()

        return self.w_opt

    def Surrogate_LANDO(self, x):
        return self.W_tilde @ self.kernel(self.X_tilde, Scale(self.X) * x)

    def Surrogate_Rollouts(self, x):
        return self.w_opt @ self.kernel(self.X_tilde, Scale(self.X) * x)

    def standard_vs_rollout(self, t_end, X):
        ### Make prediction for the X snapshot matrix using the two models
        X_pred_lando = Predict(self.Surrogate_LANDO, Tend=t_end, IC=self.X[:, 0], dt=1 / 1000,
                               type='Discrete')
        X_pred_rollouts = Predict(self.Surrogate_Rollouts, Tend=t_end, IC=self.X[:, 0], dt=1 / 1000,
                                  type='Discrete')

        x_error_init = self.relative_error(X, X_pred_lando)
        x_error_rollout = self.relative_error(X, X_pred_rollouts)

        print(f'Error LANDO X:{x_error_init}\n'
              f'Error Rollout X: {x_error_rollout}')

        losses_init = self.Rollout_func(self.W_tilde, record=True)[1]
        losses_after_rollout = self.Rollout_func(w_optimal, record=True)[1]

        plt.semilogy(losses_init, '.-', label=r'$W_{init}$')
        plt.semilogy(losses_after_rollout, '.-', label=r'$W_{optimal}$')
        plt.xlabel(r'Summation element of  $J$')
        plt.ylabel('Errors')
        plt.legend()
        plt.show()
        print(len(losses_init))
        print(len(losses_after_rollout))


r_length = 2
sparsity_lando = 1e-5
kernel = linear_kernel
train_horizon = 0.6
t_train = int(train_horizon / 0.001)

X = Allen_Cahn_eq(D=0.00075, a=1)[:, :t_train][:, :-1]
Y = Allen_Cahn_eq(D=0.00075, a=1)[:, :t_train][:, 1:]

lando_rollouts = RolloutLANDO(rollout_length=r_length, X=X, Y=Y, kernel=kernel, sparsity_tol=sparsity_lando)
lando_rollouts.LANDO()
w_optimal = lando_rollouts.optimize(epochs=60, lr=1e-4)
lando_rollouts.standard_vs_rollout(t_end=train_horizon, X=Allen_Cahn_eq(D=0.00075, a=1)[:, :t_train])

