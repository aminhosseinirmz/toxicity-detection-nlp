from pprint import pprint

import torch
from torch import nn

from models import BaseModel


class MLP(BaseModel):
    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        def get_layers():
            layers = [
                nn.Embedding(
                    num_embeddings=vocab_size,
                    embedding_dim=embed_dim,
                    padding_idx=0,
                ),
                nn.Flatten(),
            ]

            temp = self.maximum_tokens * embed_dim
            while temp / 5 > self.target_classes_len * 5:
                layers.append(nn.Linear(int(temp), int(temp / 5)))
                layers.append(nn.ReLU())
                temp /= 5

            layers.append(nn.Linear(int(temp), self.target_classes_len))
            layers.append(nn.Sigmoid())
            return layers

        self.fc_layers = nn.Sequential(*get_layers())

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)  # works
        # self.optimizer = torch.optim.RMSprop(self.parameters(), lr=0.001)  # works
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001) # doesn't work

        self._prepare_model()

    def forward(self, text):
        return self.fc_layers(text)


if __name__ == "__main__":
    mlp = MLP(
        embed_dim=100,
        vocab_size=291000,
        optimizer_type="adam",
        learning_rate=0.001,
        maximum_tokens=50,
    ).to("cuda")
    pprint(mlp)

    mlp(torch.randint(0, 291000, (256, 50)).to("cuda"))
