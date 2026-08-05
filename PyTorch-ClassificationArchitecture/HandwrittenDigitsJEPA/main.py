import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
import evaluate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("ylecun/mnist")
dataset.set_format("torch", device=device)

train_loader = DataLoader(dataset["train"], batch_size=64, shuffle=True)
test_loader = DataLoader(dataset["test"], batch_size=64, shuffle=False)


class Encoder(nn.Module):
    def __init__(self, input_dim=28 * 28, hidden_dim = 512, output_dim = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        """
        x = [batch_size, input_dim]
        """
        x = x.view(-1, self.input_dim)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x

class Predictor(nn.Module):
    def __init__(self, input_dim=128, hidden_dim = 512):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, x):
        """
        x = [batch_size, input_dim]
        """
        x = x.view(-1, self.input_dim)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x


class Trainer:
    def __init__(self, encoder, predictior, criteria, optimizer, train_loader, test_loader):
        self.encoder = encoder
        self.predictior = predictior
        self.criteria = criteria
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
    
    def train(self, num_epoch):
        for epoch in range(num_epoch):
            pbar = tqdm.tqdm(self.train_loader, desc=f"Epoch: {epoch} / {num_epoch}")
            total_loss = 0

            for idx, batch in enumerate(pbar):
                inputs = batch["image"].float()
                labels = batch["label"]

                self.optimizer.zero_grad()

                predictions = self.encoder(inputs)

                loss = self.criteria(predictions, labels)
                loss.backward()

                self.optimizer.step()

                if idx > 0:
                    pbar.set_postfix(loss=f"{total_loss / idx}")

    def eval(self, metrics):
        ...
