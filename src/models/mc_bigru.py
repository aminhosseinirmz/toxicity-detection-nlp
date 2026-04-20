import torch
from torch import nn

from .base_model import BaseModel


class MCBiGRU(BaseModel):
    def __init__(
        self,
        embed_dim: int,
        rnn_hidden_size: int = 10,
        conv_config: dict = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if conv_config is None:
            conv_config = {"num_channels": 128, "kernel_sizes": [1, 2, 3, 5, 6]}

        self.conv_config = conv_config

        self.convolutions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        embed_dim,
                        conv_config["num_channels"],
                        kernel_size=kernel,
                    ),
                    nn.ReLU(),
                    nn.Dropout(0.6),
                    nn.AdaptiveMaxPool1d((4,)),
                )
                for kernel in conv_config["kernel_sizes"]
            ]
        )

        self.rnns = nn.ModuleList(
            [
                nn.GRU(
                    input_size=conv_config["num_channels"],
                    hidden_size=rnn_hidden_size,
                    num_layers=10,
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.6,
                )
                for _ in conv_config["kernel_sizes"]
            ]
        )

        self.fc1 = nn.Linear(
            rnn_hidden_size * 2 * len(conv_config["kernel_sizes"]),
            self.target_classes_len,
        )
        self.batch_nn = nn.BatchNorm1d(self.target_classes_len)
        self.sigmoid = nn.Sigmoid()

        self._prepare_model()

    def forward(self, text):
        # CNN
        reshaped_cnn_in = torch.permute(text, (0, 2, 1))
        cnn_out = [conv(reshaped_cnn_in).squeeze(2) for conv in self.convolutions]
        cnn_out = [
            out_.view(text.size(0), 4, self.conv_config["num_channels"])
            for out_ in cnn_out
        ]

        # BiGRU
        bigru_out = [self.rnns[idx](out_)[0][:, -1] for idx, out_ in enumerate(cnn_out)]

        # concat
        concat_out = torch.concat(bigru_out, dim=1)

        # FC
        fc_out = self.fc1(concat_out)
        batch_n = self.batch_nn(fc_out)
        fc_out = self.sigmoid(batch_n)

        return fc_out
