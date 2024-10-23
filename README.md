![fig_github](https://github.com/user-attachments/assets/52756efe-4177-4404-84fc-d0226e7ea0ea)


# A parametric framework for kernel-based dynamic mode decomposition using deep learning


This repository contains the methods and techniques implemented in the context of the thesis
research project of MSc in Computational Science at the University of Amsterdam.





In this work, a non-intrusive parametric surrogate model is presented, able to provide
temporal and parametric predictions for systems characterized by high state dimension and nonlinear
behavior. In detail, we propose a parametrization algorithm to extend the kernel-based
LANDO framework (Baddoo et al., [2022](https://royalsocietypublishing.org/doi/full/10.1098/rspa.2021.0830)) to a parametric format (parametric LANDO) using deep learning techniques. 
Additionally, we propose the application of dimensionality reduction techniques to reduce the training cost of parametric LANDO for high-dimensional problems. 
We showcase the effectiveness of the proposed parametric surrogate model in predicting the parametric dynamics of
three numerical problems with nonlinear behavior; namely the Lotka-Volterra model, a
2D heat equation, and the Allen-Cahn equation.


## Experiments

Several experiments are performed, regarding the application of the parametric surrogate model in the three aforementioned numerical problems. 
Additionally, we perform experiments to investigate whether the performance of parametric LANDO is influenced by the employment of the aforementioned dimensionality reduction techniques.

### Lotka-Volterra model
To perform the experiments regarding the application of parametric LANDO in the Lotka-Volterra model, the following python files should be run:

* `Experiments_1D.py` (The parameter space is one-dimensional)
* `Experiments_2D.py` (The parameter space is two-dimensional)
* `Experiment_Cardinality_Training.py` (Investigate the dependency of the parametric surrogate model's performance on the cardinality of the training set)



### Heat Equation 

For this numerical problem, the snapshot data for the training of the parametric framework are generated with the finite element method (FEM). 
The implementation of the finite element simulation is based on the open source software [FreeFEM](https://freefem.org/). 

To generate the numerical data, please run the `Data_Generation.py`, `Heat_problem_thesis.edp`, and `Data_Preprocessing.py` files in that order. 
Subsequently, parametric LANDO can be employed to approximate the parametric dynamics of this system with the `Experiments_DiffTimes.py` file.

When running this file, several results regarding the performance of parametric LANDO are automatically saved in a .pkl file. 
These results can be visualized by running the `Visualise_errors.py`file. 


As a last step, the `Visualization.edp` file can be run to obtain the .vtk visualisations of the reference solutions, predicted solutions, and prediction errors of the parametric surrogate model for several parametric instances. 



### Allen-Cahn equation

To generate the snapshot data, run the `Data_Generation.py` file. 

To obtain results from the application of parametric LANDO in this system, run the `Experiments_DiffTimes.py`, in the folder of the Allen-Cahn equation. 

To obtain results regarding the application of POD during the training of parametric LANDO for the Allen-Cahn equation or the heat equation, run the `POD_Experiment.py` file.

Again, results regarding the performance of parametric LANDO in approximating the dynamics of the Allen-Cahn equation can be visualised by running the `Visualise_errors.py`file, in the Allen-Cahn folder. 




### Important Notes:
* In order to run the aforementioned files, please change the names of the directories in the code, to save the results and figures in your local device.
* Note that FEM simulation for data generation of the heat equation is computationally expensive, so this might take a longer time to run. 

