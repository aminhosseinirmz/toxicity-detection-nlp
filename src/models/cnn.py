import torch
from torch import nn

from .base_model import BaseModel


class CNN(BaseModel):
    def __init__(
        self,
        embed_dim: int,
        conv_config: dict = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if conv_config is None:
            conv_config = {"num_channels": 50, "kernel_sizes": [1, 2, 3, 4, 5, 6]}

        self.conv_config = conv_config

        self.convolutions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        embed_dim,
                        self.conv_config["num_channels"],
                        kernel_size=kernel,
                    ),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d((1,)),
                )
                for kernel in self.conv_config["kernel_sizes"]
            ]
        )

        self.fc1 = nn.Linear(
            len(self.conv_config["kernel_sizes"]) * self.conv_config["num_channels"],
            self.target_classes_len,
        )
        self.sigmoid = nn.Sigmoid()

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001) # works
        # self.optimizer = torch.optim.RMSprop(self.parameters(), lr=0.001)  # works good
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001) # doesn't work

        self._prepare_model()

    def forward(self, text):
        # CNN
        reshaped_cnn_in = torch.permute(text, (0, 2, 1))
        cnn_out = [conv(reshaped_cnn_in).squeeze(2) for conv in self.convolutions]
        concat_out = torch.cat(cnn_out, dim=1)

        # FC
        fc_out = self.fc1(concat_out)
        fc_out = self.sigmoid(fc_out)

        return fc_out
