import torch
import math 


"""BNE: basic neural network explanation, how to execute and learning with
a simple mathematical approximation as the model system"""
def bne():
    # Here is a basic start on how to train a neural network
    # The beginning is setting the device, on the computer's accelerator if the
    # accelerator is available, if not then it will go for the cpu

    dtype = torch.float
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    # Here we are defining the function that we want to fit, basically
    # this is the problem we are solving. In many cases this will be the 
    # thing we want the model to learn!
    # For example, y = sin(x) * cos(x) as defined below
    x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
    y = torch.sin(x)*torch.cos(x)
    # Prepare the polynomial tensor
    p = torch.tensor([1, 2, 3])
    xx = x.unsqueeze(-1).pow(p)

    # use the nn.package to define the model and loss function
    # The sequential lets you pass layers sequentially, in our
    # case we pass a linear and then flatten it.
    model = torch.nn.Sequential(
        torch.nn.Linear(3,1),
        torch.nn.Flatten(0,1)
    )
    loss_fn = torch.nn.MSELoss(reduction = 'sum')
    # Lr is the learning rate
    lr = 1e-6
    optimizer = torch.optim.RMSprop(model.parameters(), lr = lr)
    for t in range(2000):
        # forward pass
        y_pred = model(xx)
        # calculate loss
        loss = loss_fn(y_pred, y)
        if t%100 == 99:
            print(t,loss.item())
        #  set optimizer to zero gradient
        optimizer.zero_grad
        # do the backwards pass
        loss.backward()
        # calling step function on optimizer makes it update the params
        optimizer.step()
    linear_layer= model[0]

    return print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')



from torch import nnfrom torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

training_data = datasets.FashionMNIST(
    root = 'data'
    train = True
    download = True
    transform = ToTensor(),
)
test_data = datasets.FashionMNIST(
    root = 'data'
    train = False
    download = True
    transform = ToTensor(),
)

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size = batch_size)

for x, y in test_dataloader:
    print(f"Shape of X (N, C, H, W): {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_ReLU_stack = nn.sequential(
            nn.Linear(28*28, 512)
            nn.ReLU()
            nn.Linear(512,512),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        x = self. self.flattten(x)
        logits = self.linear_ReLU_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 50 == 0:
            loss, current = loss.item(), (batch+1)*len(x)

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fun(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum.item()
    test_loss /= num_batchets
    correct /= size
    print(f"Test Error: \n Accuracy: {{100*correct}:>0.1f%}, Avg loss: {test_loss:>8f} \n")

epochs = 150

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")     

