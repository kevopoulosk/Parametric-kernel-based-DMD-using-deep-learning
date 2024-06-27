# Kernel-based parametric Dynamic Mode Decomposition with Active Learning 

This repository contains the methods and techniques implemented in the context of the thesis
research project of MSc in Computational Science at the University of Amsterdam & University of Twente.


In this work, a non-intrusive parametric surrogate model is presented, able to provide
temporal predictions for systems characterized by high state dimension and nonlinear
behavior. In detail, a parametrization algorithm is developed to extend the kernel-based
LANDO framework (Baddoo et al., [2022](https://royalsocietypublishing.org/doi/full/10.1098/rspa.2021.0830)) to a parametric format (pLANDO) using regres-
sion techniques. Additionally, an adaptive sampling technique based on deep ensemble
learning is developed, to smartly explore the parameter space and extract the most in-
formative set of samples for the training of pLANDO. We showcase the effectiveness
of the proposed parametric surrogate model in predicting the parametric dynamics of
three numerical problems with nonlinear behavior; namely the Lotka-Volterra model, a
2D heat equation, and the Allen-Cahn equation

