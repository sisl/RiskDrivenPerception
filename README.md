# Risk-Driven Design of Perception Systems
![](https://github.com/smkatz12/CS230_DAA/blob/main/collision_avoidance/figures/inference_test_above_v2.gif)

## Dependencies
To run this code you will need to add the following unregistered julia packages
* https://github.com/ancorso/POMDPGym.jl
* https://github.com/ancorso/Crux.jl

## Common Code
* `src/risk_solvers.jl`: Implementation of the risk solvers for the abstracted perception MDP

## Pendulum Example
Our initial toy problem will be the control of an inverted pendulum from images. Below is a description of each of the files and folders in `inverted_pendulum/`
* `controllers`: Contains a rule-based policy and a neural network policy for mapping pendulum state to torque
* `problem_setup.jl`: Contains the definition of the abstracted perception MDP problem
* `nn_controller_training.jl`: Code used to train a neural network controller for the pendulum [NOTE: We currently just use the rule-based controller defined in `inverted_pendulum/controllers/rule_based.jl`]
* `nn_surrogate_training.jl`: Code used to generate the risk tables and then train a surrogate model to encode the risk function.
* `risk_estimation.jl`: Code used to generate the risk tables and show that they are correct with respect to sampling [NOTE: This file isn't necessary for the NN surrogate training.]
* `perception_training.jl`: Code used to train the nominal and risk-sensitive perception systems for the image-based pendulum environment.
* `weight_perception_training.jl`: Code used to train the weighted risk-sensitive perception system

## Collision Avoidance Example
The other test case is a realistic vision-based collision avoidance system based on Yolov5. Below is a description of each of the files and folders in `collision_avoidance/`
* `data_generation/`: Code to produce training image datasets and corresponding labels using the Xplane 11 simulation environment
* `encounter_model/`: Code to define the straight-line encounters, each leading to NMAC
* `models/`: The trained yolo models corresponding to the experiments done in the paper
* `yolov5/`: Contains the unedited code from yolov5 (https://github.com/ultralytics/yolov5)
* `yolov5_risk/`: Contains the yolov5 edited to handle risk labels and risk in the loss function.
* `analyze_results.jl`: Code for evaluating the different perception models
* `enc_analysis_fig.jl`: Generates figures for analyzing an encounter
* `nominal_errors.jl`: Code for computing the nominal error models
* `risk_estimation_daa.jl`: Code for estimating the risk of the abstracted perception MDP
* `simulate.jl`: Code for running the perception system in Xplane 11. 


 
