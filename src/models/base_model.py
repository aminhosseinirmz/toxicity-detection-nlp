import collections
import pprint

import pandas as pd
import torch
from torch import nn
import wandb
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelF1Score,
    MultilabelAUROC,
)

pp = pprint.PrettyPrinter(indent=4)

threshold = 0.5
average_mode = "micro"


class BaseModel(nn.Module):
    def __init__(
        self,
        target_classes_len: int = 6,
        optimizer_type: str = "adam",
        learning_rate: float = 0.001,
        maximum_tokens: int = 50,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.optimizer = None
        self.loss_fn = None

        self.target_classes_len = target_classes_len
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.maximum_tokens = maximum_tokens

        self.op = {
            "adam": torch.optim.Adam,
            "adadelta": torch.optim.Adadelta,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
        }
        if optimizer_type not in self.op:
            raise ValueError("Invalid value")

        self.log_values = {}

    def _prepare_model(self):
        self.optimizer = self.op[self.optimizer_type](
            params=self.parameters(),
            lr=self.learning_rate,
        )
        self.loss_fn = nn.BCELoss()

    def train_model(
        self,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        num_of_epochs: int,
        device: torch.device,
        name: str,
        architecture: str,
    ):
        wandb.init(
            project="Toxic Comment Detection",
            name=name,
            config={
                "learning_rate": self.learning_rate,
                "architecture": architecture,
                "optimizer": self.optimizer_type,
                "epochs": num_of_epochs,
            },
            anonymous="allow",
        )

        for epoch_num in range(num_of_epochs):
            self.train()
            total_loss = 0

            accuracy = MultilabelAccuracy(
                num_labels=self.target_classes_len,
                average=average_mode,
                threshold=threshold,
            ).to(device=device)
            precision = MultilabelPrecision(
                num_labels=self.target_classes_len,
                average=average_mode,
                threshold=threshold,
            ).to(device=device)
            recall = MultilabelRecall(
                num_labels=self.target_classes_len,
                average=average_mode,
                threshold=threshold,
            ).to(device=device)
            f1_score = MultilabelF1Score(
                num_labels=self.target_classes_len,
                average=average_mode,
                threshold=threshold,
            ).to(device=device)
            auroc = MultilabelAUROC(
                num_labels=self.target_classes_len,
                average=average_mode,
            ).to(device=device)

            for text_batch, label_batch in train_dataloader:
                text_batch = text_batch.to(device)
                label_batch = label_batch.to(device)

                pred = self(text_batch)
                loss = self.loss_fn(pred, label_batch)

                accuracy.update(pred, label_batch)
                precision.update(pred, label_batch)
                recall.update(pred, label_batch)
                f1_score.update(pred, label_batch)
                auroc.update(pred, label_batch.to(torch.long))

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                self.optimizer.zero_grad()

            print(f"Epoch {epoch_num + 1}:")
            val_logs = self.validate_model(
                dataloader=validation_dataloader,
                device=device,
            )
            all_logs = {
                "Training loss": total_loss / len(train_dataloader),
                "Training accuracy": accuracy.compute().item(),
                "Training precision": precision.compute().item(),
                "Training recall": recall.compute().item(),
                "Training F1_score": f1_score.compute().item(),
                "Training AUROC": auroc.compute().item(),
                **val_logs,
            }
            wandb.log(all_logs)
            pp.pprint(all_logs)

            all_logs.update({"Epoch": int(epoch_num + 1)})

            for key in all_logs.keys():
                all_logs[key] = float(format(all_logs[key], ".3f"))

            self.log_values[epoch_num + 1] = all_logs

            print("=" * 50)

        self.write_logs_to_file(name)
        self.save_model_to_disk(name)

    def save_model_to_disk(self, name: str):
        print("Saving model...")
        torch.save(self.state_dict(), f"model_weights/{name}.pth")
        print(f"Saved `{name}` to disk.\n")

    def load_model_from_disk(self, path: str):
        print("Loading model from disk...")
        if not path:
            raise ValueError("Invalid value")

        self.load_state_dict(torch.load(path))
        print(f"Loaded `{path}` to memory.")
        self.eval()

    def write_logs_to_file(self, name: str):
        all_series = collections.deque()
        for (key, log_dict) in self.log_values.items():
            all_series.append(pd.Series(log_dict))

        df = pd.DataFrame(all_series)
        df.to_csv(f"metric_logs/{name}.csv", index=False)

    def validate_model(
        self,
        dataloader: DataLoader,
        device: torch.device,
    ):
        self.eval()
        total_loss = 0

        accuracy = MultilabelAccuracy(
            num_labels=self.target_classes_len,
            average=average_mode,
            threshold=threshold,
        ).to(device=device)
        precision = MultilabelPrecision(
            num_labels=self.target_classes_len,
            average=average_mode,
            threshold=threshold,
        ).to(device=device)
        recall = MultilabelRecall(
            num_labels=self.target_classes_len,
            average=average_mode,
            threshold=threshold,
        ).to(device=device)
        f1_score = MultilabelF1Score(
            num_labels=self.target_classes_len,
            average=average_mode,
            threshold=threshold,
        ).to(device=device)
        auroc = MultilabelAUROC(
            num_labels=self.target_classes_len,
            average=average_mode,
        ).to(device=device)

        with torch.no_grad():
            for text_batch, label_batch in dataloader:
                text_batch = text_batch.to(device)
                label_batch = label_batch.to(device)

                pred = self(text_batch)
                loss = self.loss_fn(pred, label_batch)

                accuracy.update(pred, label_batch)
                precision.update(pred, label_batch)
                recall.update(pred, label_batch)
                f1_score.update(pred, label_batch)
                auroc.update(pred, label_batch.to(torch.long))

                total_loss += loss.item()

        return {
            "Validation loss": total_loss / len(dataloader),
            "Validation accuracy": accuracy.compute().item(),
            "Validation precision": precision.compute().item(),
            "Validation recall": recall.compute().item(),
            "Validation F1_score": f1_score.compute().item(),
            "Validation AUROC": auroc.compute().item(),
        }
