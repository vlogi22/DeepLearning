# DeepLearning
DeepLearning 23/24 homeworks

to execute:
python3 hw1-q1.py [model_name] -[flags]

<b>example:</b>
Q1.1(b)
python3 hw1-q1.py logistic_regression -epochs 50 -learning_rate 0.01 -image_path "images" -image_name "logistic_regression_0_01" > outputs/logistic_regression.out

<b>Q2.1</b>
python3 hw1-q2.py logistic_regression -epochs 20 -batch_size 16 -learning
_rate 0.1 -image_path "images" -image_name "Q2_logistic_regression_0_1"

<b>Q2.2</b>
python3 hw1-q2.py mlp -epochs 20 -batch_size 16 -learning
_rate 0.1 -hidden_size 200 -layers 2 -dropout 0.0 -activation relu -optimizer sgd -image_path "images" -image_name "Q2_mlp_0_1"

<b>Models names:</b>
\> perceptron
\> logistic_regression

<b>Flags:</b>
\> epochs: Number of epochs to train for. You should not need to change this value for your plots.
\> hidden_size: Number of units in hidden layers (needed only for MLP, not perceptron or logistic regression)
\> learning_rate: Learning rate for parameter updates (needed for logistic regression and MLP, but not perceptron)
\> image_path: The path which you want to save the generated plot
\> image_name: The name which you want to name the generated image

<b>Only for hw1-q2:</b>
-layers 
-dropout
-activation 
-optimizer