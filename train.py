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
# mlflow.set_tracking_uri('https://helpless-cow-17.loca.lt')
# mlflow.set_experiment("98sean98/Deploifai/test")
# mlflow.set_experiment("98sean98/project-2/test-3")

torch.manual_seed(122)

start = 101
end = pow(2, 10) - 1
size = end - start + 1

def get_bit_num():
    n = 1
    while pow(2, n) < start + size:
        n += 1
    return n

bit_num = get_bit_num()
classes = torch.tensor([
    [0, 0, 0, 1], # fizzbuzz
    [0, 0, 1, 0], # fizz
    [0, 1, 0, 0], # buzz
    [1, 0, 0, 0], # print integer as is
], dtype=torch.float32)

def encode(x):
    q = pow(2, bit_num - 1)
    bits = []
    while q >= 1:
        i = math.floor(x / q)
        x = x % q
        bits.append(i)
        q = q / 2
    return bits

def decode(bits):
    x = 0
    for b in bits:
        x *= 2
        x += b
    return x

def generate_dataset():
    x = torch.tensor([encode(i) for i in range(start, start + size)], dtype=torch.float32)
    y = torch.empty((size, classes.shape[1]))
    for i in range(start, start + size):
        if i % 15 == 0: y_i = classes[0]
        elif i % 3 == 0: y_i = classes[1]
        elif i % 5 == 0: y_i = classes[2]
        else: y_i = classes[3]
        y[i - start] = y_i
    return x, y

class FizzbuzzDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, size):
        self.x, self.y = x, y
        self.size = size

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.size

def get_dataloader(x, y, dataset_size, batch_size):
    return torch.utils.data.DataLoader(FizzbuzzDataset(x, y, dataset_size), batch_size=batch_size)

class NNModel(torch.nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_dim, hid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_dim, hid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_dim, output_dim),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        w = self.network(x)
        return w

def train_one_epoch(model, criterion, optimizer, train_dataloader, train_size):
    model.train()
    train_loss = 0

    for x, y in train_dataloader:
        batch_size = x.shape[0]
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item() * batch_size

    train_loss = train_loss / train_size
    return train_loss

def val_one_epoch(model, criterion, val_dataloader, val_size):
    correct = 0
    val_loss = 0
    model.eval()

    with torch.no_grad():
        for x, y in val_dataloader:
            batch_size = x.shape[0]
            y_hat = model(x)
            loss = criterion(y_hat, y)
            predicted = torch.argmax(y_hat, dim=1)
            val_loss += loss.item() * batch_size
            labeled = torch.argmax(y, dim=1)
            correct += (predicted == labeled).sum().item()

    val_loss = val_loss / val_size
    val_acc = 100 * correct / val_size
    return val_loss, val_acc

def main(log_metric):
    script_dir = os.path.dirname(__file__)
    output_path = os.path.join(script_dir, 'artifacts/output.txt')

    x, y = generate_dataset()
    train_proportion = 0.9
    train_size = math.floor(train_proportion * x.shape[0])
    val_size = x.shape[0] - train_size
    dataset_split = [train_size, val_size]
    x_train, x_val = torch.utils.data.random_split(x, dataset_split)
    y_train, y_val = torch.utils.data.random_split(y, dataset_split)

    # train and val set label summary
    # print(y_train[:].sum(dim=0), y_val[:].sum(dim=0))

    # hyperparameters
    hid_dim = 100
    lr = 0.1
    epochs = 10
    batch_size = 100

    train_dataloader = get_dataloader(x_train, y_train, train_size, batch_size)
    val_dataloader = get_dataloader(x_val, y_val, val_size, batch_size)
    model = NNModel(bit_num, hid_dim, classes.shape[1])
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)

    train_losses = []
    val_losses = []
    val_acces = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, criterion, optimizer, train_dataloader, train_size)
        val_loss, val_acc = val_one_epoch(model, criterion, val_dataloader, val_size)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_acces.append(val_acc)

        log_metric('train_loss', train_loss)
        log_metric('val_loss', val_loss)
        log_metric('val_acc', val_acc)

        print(
            'Epoch: [{:4d}] | * training loss : {:.3f} | o validation loss : {:.3f} | + validation acc. {:.2f} %'
            .format(epoch, train_loss, val_loss, val_acc)
        )
        sleep(5)

    script_dir = os.path.dirname(__file__)
    artifacts_dir = os.path.join(script_dir, 'artifacts')
    model_path = os.path.join(artifacts_dir, 'model.pt')

    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    # with mlflow.start_run() as run:
    #     print('mlflow run id', run.info.run_id)
    #
    #     main(lambda k, v: mlflow.log_metric(k, v))

    main(lambda k, v: print(k, v))
