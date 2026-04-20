import torch
from torch import nn

from .base_model import BaseModel


class CNNBiGRU(BaseModel):
    def __init__(
        self,
        embed_dim: int,
        conv_config: dict = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if conv_config is None:
            conv_config = {"num_channels": 50, "kernel_sizes": [1, 2, 3]}

        self.convolutions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        embed_dim,
                        conv_config["num_channels"],
                        kernel_size=kernel,
                    ),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d((1,)),
                )
                for kernel in conv_config["kernel_sizes"]
            ]
        )

        rnn_hidden_size = int(
            conv_config["num_channels"] * len(conv_config["kernel_sizes"])
        )

        self.bigru = nn.GRU(
            input_size=embed_dim,
            hidden_size=rnn_hidden_size,
            num_layers=5,
            batch_first=True,
            bidirectional=True,
            dropout=0.6,
        )

        self.fc1 = nn.Linear(rnn_hidden_size, self.target_classes_len)
        self.sigmoid = nn.Sigmoid()

        # self.optimizer = torch.optim.Adadelta(self.parameters(), lr=1.0)

        self._prepare_model()

    def forward(self, text):
        # CNN
        reshaped_cnn_in = torch.permute(text, (0, 2, 1))
        cnn_out = [conv(reshaped_cnn_in).squeeze(2) for conv in self.convolutions]
        concat_out = torch.cat(cnn_out, dim=1)

        # BiGRU
        bigru_out, hidden = self.bigru(text)
        bigru_out = hidden[-2, :] + hidden[-1, :]

        # concat CNN and BiGRU
        bigru_out_mean = torch.mean(bigru_out)
        concat_out = concat_out - bigru_out_mean
        fc_in = concat_out * bigru_out

        # FC
        fc_out = self.fc1(fc_in)
        fc_out = self.sigmoid(fc_out)

        return fc_out
