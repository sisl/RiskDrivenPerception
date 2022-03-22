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

#### Controller
The controller was trained using Proximal Policy Optimization on the 2D pendulum state and has the resulting behavior and policy map. The controller is stored in `inverted_pendulum/controllers/`

![pendulum control from state](inverted_pendulum/figures/pendulum_control.gif)
![Controller policy map](inverted_pendulum/figures/controller_policy.png)

#### Perception System
To train a perception system, we generate pendulum images with the corresponding state, creating a dataset that looks like the following:

![Perception dataset](inverted_pendulum/figures/training_data.png)

We train a simple MLP that has one hidden layer with 64 units and get the following distribution of prediction errors. The model is stored in `inverted_pendulum/perception/`

![Perception erros](inverted_pendulum/figures/perception_model_accuracy.png)

#### Combined System
We can now concatenate the perception system with the controller to construct the combined agent. The behavior and policy map of the combined agent is shown below.

![Pendulum control from images](inverted_pendulum/figures/img_pendulum.gif)
![Controller policy map](inverted_pendulum/figures/image_control_policy.png)
