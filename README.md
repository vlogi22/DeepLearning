# DeepLearning
DeepLearning 23/24 homeworks

to execute:
python3 hw1-q1.py <model_name> -<flags> 

example:
python3 hw1-q1.py logistic_regression -epochs 50 -learning_rate 0.01 -image_path "images" -image_name "logistic_regression_0_01" > outputs/logistic_regression.out

Models names:
> perceptron
> logistic_regression

Flags:
> epochs: Number of epochs to train for. You should not need to change this value for your plots.
> hidden_size: Number of units in hidden layers (needed only for MLP, not perceptron or logistic regression)
> learning_rate: Learning rate for parameter updates (needed for logistic regression and MLP, but not perceptron)
> image_path: The path which you want to save the generated plot
> image_name: The name which you want to name the generated image