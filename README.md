# Risk Sensitive Perception

## Dependencies
To run this code you will need to add the following unregistered julia packages
* https://github.com/ancorso/POMDPGym.jl
* https://github.com/ancorso/Crux.jl

## Pendulum Example
Our initial toy problem will be the control of an inverted pendulum from images. Below is a description of each of the files
* `inverted_pendulum/nn_controller_training.jl`: Code used to train a neural network controller for the pendulum [NOTE: We currently just use the rule-based controller defined in `inverted_pendulum/controllers/rule_based.jl`]
* `inverted_pendulum/nn_surrogate_training.jl`: Code used to generate the risk tables and then train a surrogate model to encode the risk function.
* `inverted_pendulum/risk_estimation.jl`: Code used to generate the risk tables and show that they are correct with respect to sampling [NOTE: This file isn't necessary for the NN surrogate training.]
* `inverted_pendulum/perception_training.jl`: Code used to train the nominal and risk-sensitive perception systems for the image-based pendulum environment.

