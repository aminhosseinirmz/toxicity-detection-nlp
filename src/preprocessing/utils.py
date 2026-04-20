import collections
from functools import partial
from typing import List, Tuple, Optional

import pandas as pd
import spacy
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, RandomSampler, DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe, vocab, Vocab
from tqdm import tqdm
from transformers import BertTokenizer

vector_cache_dir = ".vector_cache"
spacy.load("en_core_web_sm")
spacy_tokenizer = get_tokenizer("spacy")


class ToxicCommentsDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_convertor: str = "glove_300",
        maximum_tokens: int = 50,
        target_classes: List[str] = None,
    ):
        if target_classes is None:
            target_classes = [
                "toxic",
                "severe_toxic",
                "obscene",
                "threat",
                "insult",
                "identity_hate",
            ]
        self.vocab_: Optional[Vocab] = None

        if not maximum_tokens:
            raise ValueError(f"Invalid `maximum_token` value: {maximum_tokens}")

        if target_convertor not in ("glove_300", "glove_100", "vocab", "bert"):
            raise ValueError(f"Invalid value!")

        if target_convertor == "glove_300":
            glove_300 = GloVe(name="840B", dim=300, cache=vector_cache_dir)
            fn = partial(self.text_to_embedding, glove_300)
        elif target_convertor == "glove_100":
            glove_100 = GloVe(name="6B", dim=100, cache=vector_cache_dir)
            fn = partial(self.text_to_embedding, glove_100)
        elif target_convertor == "vocab":
            self.build_vocab(dataframe)
            fn = self.vocab_tokenizer_fn
        elif target_convertor == "bert":
            tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
            fn = partial(self.bert_tokenizer, tokenizer)
        else:
            raise ValueError(
                f"Invalid value for `traget_convertor`: {target_convertor}"
            )

        self.dataframe = dataframe
        self.maximum_tokens = maximum_tokens
        self.vocab_size = None

        self.X = collections.deque()
        self.Y = collections.deque()

        for i, row in tqdm(dataframe.iterrows()):
            comment_text = row["comment_text"]
            labels = row[target_classes]

            self.X.append(fn(comment_text))
            self.Y.append(torch.FloatTensor(labels))

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def build_vocab(self, dataframe: pd.DataFrame):
        token_counts = collections.Counter()
        for _comment_text in dataframe["comment_text"]:
            tokens = spacy_tokenizer(_comment_text)
            token_counts.update(tokens)

        print("Vocab-size:", len(token_counts))
        self.vocab_size = len(token_counts)

        sorted_by_freq_tuples = sorted(
            token_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        ordered_dict = collections.OrderedDict(sorted_by_freq_tuples)

        vocab_ = vocab(ordered_dict)

        vocab_.insert_token("<pad>", 0)
        vocab_.insert_token("<unk>", 1)
        vocab_.set_default_index(1)

        self.vocab_ = vocab_

    def get_tokens(self, text_: str) -> List[str]:
        tokens_ = [token for token in spacy_tokenizer(text_)]
        if len(tokens_) < self.maximum_tokens:
            tokens_ += [""] * (self.maximum_tokens - len(tokens_))
        else:
            tokens_ = tokens_[: self.maximum_tokens]

        return tokens_

    def bert_tokenizer(self, tokenizer: BertTokenizer, text_: str) -> torch.Tensor:
        tokens = tokenizer.encode(text_, add_special_tokens=True)
        if len(tokens) > self.maximum_tokens:
            tokens = tokens[: self.maximum_tokens - 1] + [tokenizer.sep_token_id]

        return torch.LongTensor(tokens)

    def text_to_embedding(self, glove: GloVe, text_: str) -> torch.Tensor:
        return glove.get_vecs_by_tokens(self.get_tokens(text_), lower_case_backup=False)

    def vocab_tokenizer_fn(self, text_: str) -> torch.Tensor:
        return torch.tensor(
            [self.vocab_[token] for token in self.get_tokens(text_)],
            dtype=torch.int64,
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        return self.X[index], self.Y[index]


#####################################################################


def get_dataloaders(
    target_convertor: str = "glove_300",
    maximum_tokens: int = 50,
    batch_size: int = 64,
    dataset_fraction: float = 1.0,
) -> Tuple[DataLoader, DataLoader]:
    train_df = pd.read_csv("data/train.csv")

    # training on a part of data for speed
    train_df = train_df.sample(frac=dataset_fraction)
    train_df, val_df = train_test_split(train_df, test_size=0.2)

    train_dataset = ToxicCommentsDataset(
        train_df,
        target_convertor=target_convertor,
        maximum_tokens=maximum_tokens,
    )
    val_dataset = ToxicCommentsDataset(
        val_df,
        target_convertor=target_convertor,
        maximum_tokens=maximum_tokens,
    )

    train_sampler = RandomSampler(train_dataset)
    val_sampler = RandomSampler(val_dataset)

    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
    )

    return train_dl, val_dl
