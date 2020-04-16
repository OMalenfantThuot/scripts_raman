import torch
from torch.optim import Adam

class Trainer:
    def __init__(self, model, train_loader, val_loader, loss_fn, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = Adam(self.model.parameters)
        self.loss_fn = loss_fn
        self.device = device
        self.max_epochs = 50

    def train():
        self.model.to(self.device)
        for state in self.optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        
        for _ in range(self.max_epochs):
            train_iter = self.train_loader
            for train_batch in train_iter:
                self.optimizer.zero_grad()
                train_batch = {k: v.to(device) for k, v in train_batch.items()}
                result = self.model(train_batch)
                loss = self.loss_fn


def get_trainer(model, train_loader, val_loader, device, args):
    loss_fn = get_loss_fn(args)
    trainer = Trainer(model, train_loader, val_loader, loss_fn, device)
    return trainer
