import os
import sys
import torch
from torch.optim import Adam

sys.path.append(os.environ["RAMAN"])


class Trainer:
    def __init__(self, model, train_loader, val_loader, loss_fn, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = Adam(self.model.parameters())
        self.loss_fn = loss_fn
        self.device = device
        self.max_epochs = 10000

    def train(self):
        self.model.to(self.device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
        
        n_train = sum([len(t["x"]) for t in self.train_loader])
        n_val = sum([len(v["x"]) for v in self.val_loader])

        patience, it_count = 30, 0
        best_loss = 10**6
        
        with open("train.log", "w") as out:
            for i in range(self.max_epochs):
                it_train_loss, it_val_loss = 0.0, 0.0
                for train_batch in self.train_loader:
                    self.optimizer.zero_grad()
                    train_batch = {
                        k: v.reshape(-1, 1).to(self.device) for k, v in train_batch.items()
                    }
                    result = self.model(train_batch["x"])
                    batch_train_loss = self.loss_fn(train_batch, result)
                    it_train_loss += batch_train_loss * len(train_batch)
                    batch_train_loss.backward()
                    self.optimizer.step()
                it_train_loss = it_train_loss / n_train

                for val_batch in self.val_loader:
                    val_batch = {
                        k: v.reshape(-1, 1).to(self.device) for k, v in val_batch.items()
                    }
                    result = self.model(val_batch["x"])
                    batch_val_loss = self.loss_fn(val_batch, result)
                    it_val_loss += batch_val_loss * len(val_batch)
                it_val_loss = it_val_loss / n_val

                if it_val_loss < best_loss:
                    best_loss = it_val_loss
                    it_count = 0
                    torch.save(self.model, "best_model")
                else:
                    it_count += 1

                out.write("Iteration : {} \tTrain loss : {} \tVal loss : {}\n".format(i, it_train_loss, it_val_loss))
                if it_count >= patience:
                    out.write("Patience exceeded.\n")
                    break
            out.write("Training completed.")



def get_trainer(model, train_loader, val_loader, device, args):
    loss_fn = get_loss_fn(args)
    trainer = Trainer(model, train_loader, val_loader, loss_fn, device)
    return trainer


def get_loss_fn(args):
    if args.loss == "default":
        from utils.functions.custom_loss import default_loss

        return default_loss()
    elif args.loss == "up":
        from utils.functions.custom_loss import up_loss

        return up_loss()
    elif args.loss == "down":
        from utils.functions.custom_loss import down_loss

        return down_loss()
