import numpy as np
import matplotlib.pyplot as plt
import pickle

t_test = [0.05, 0.17, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.97]

with open(f'errors_dict.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

test_error_mean = []
test_error_std = []
for t in t_test:
    test_error_mean.append(loaded_dict[f"t = {t}"][0][1])
    test_error_std.append(loaded_dict[f"t = {t}"][0][3])

plt.figure()
plt.errorbar(t_test, test_error_mean, yerr=test_error_std, marker='o', capsize=5)
plt.grid(True)
plt.ylabel(r"Mean $L_2$ Relative Error")
plt.xlabel(r"t")
plt.yscale("log")
plt.axvline(x=0.6, linestyle='--', color='black')
plt.show()

plt.figure()
plt.semilogy(loaded_dict['t = 0.97'][0][-2], 'o')
plt.xlabel("Parameter Index")
plt.ylabel(r"Mean $L_2$ Relative Error")
plt.grid(True)
plt.show()

plt.figure()
plt.semilogy(loaded_dict['t = 0.97'][0][-1], 'o')
plt.xlabel("Parameter Index")
plt.ylabel(r"Mean $L_2$ Relative Error")
plt.grid(True)
plt.show()


