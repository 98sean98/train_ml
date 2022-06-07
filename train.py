"""
Train a linear layer that represents the coefficients of the taylor series expansion terms of the sine function
"""


import os
import math
from time import sleep

import torch
import mlflow

# only if running an experiment locally (this must come before setting mlflow tracking uri and experiment)
# os.environ["MLFLOW_TRACKING_USERNAME"] = "98sean98/project-2/test"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "generated token"
# setup mlflow for this experiment
# mlflow.set_tracking_uri('https://community.mlflow.deploif.ai')
# mlflow.set_experiment("98sean98/Deploifai/test")

torch.manual_seed(122)

def generate_features(x):
    # Prepare the input tensor (x, x^2, x^3).
    p = torch.tensor([1, 2, 3])
    return x.unsqueeze(-1).pow(p)

def main(log_metric, log_param):
    # Create Tensors to hold input and outputs.
    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)

    xx = generate_features(x)

    # Use the nn package to define our model and loss function.
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 1),
        torch.nn.Flatten(0, 1)
    )
    loss_fn = torch.nn.MSELoss(reduction='sum')

    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us. Here we will use RMSprop; the optim package contains many other
    # optimization algorithms. The first argument to the RMSprop constructor tells the
    # optimizer which Tensors it should update.
    learning_rate = 1e-3
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    epochs = 2000

    log_param('learning_rate', learning_rate)
    log_param('epochs', epochs)

    print('start training')

    for t in range(epochs):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(xx)

        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        if t % 100 == 99:
            log_metric('training_loss', loss.item())
            # add sleep function to artifically prolong training
            # sleep(5)

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()


    linear_layer = model[0]
    print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')

    # test the model using a 100 random samples from the training dataset
    test_x = x[torch.randint(0, x.shape[0], (100, ))]
    test_y = torch.sin(test_x)
    test_xx = generate_features(test_x)
    test_y_pred = model(test_xx)
    test_loss = loss_fn(test_y_pred, test_y)
    print(f'Test loss: {test_loss.item()}')
    log_metric('test_loss', test_loss.item())

    # save model weights
    print('saving model weights')
    script_dir = os.path.dirname(__file__)
    artifacts_dir = os.path.join(script_dir, 'artifacts')
    model_path = os.path.join(artifacts_dir, 'model.pt')

    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    # with mlflow.start_run() as run:
        # print('mlflow run id', run.info.run_id)

        # main(lambda k, v: mlflow.log_metric(k, v), lambda k, v: mlflow.log_param(k, v))
    main(lambda k, v: print(k,v), lambda k, v: print(k,v))
