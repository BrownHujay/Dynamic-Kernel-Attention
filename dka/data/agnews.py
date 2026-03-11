"""AG News data loader for topic classification.

AG News: 4 classes (World, Sports, Business, Sci/Tech), ~120k train / ~7.6k test.

Uses HuggingFace datasets for download and a simple word-level tokenizer with
a vocabulary built from the training set. Sequences are truncated/padded to
a fixed length (default 128).

The model receives integer token IDs and is expected to use a learned
embedding table of size (vocab_size, d).

Reference: DKA Build Guide, Sections 4.2, 5.4.
"""

import re
from collections import Counter
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset


# Special token indices
PAD_IDX = 0
UNK_IDX = 1


class Vocabulary:
    """Simple word-level vocabulary built from a corpus.

    Args:
        max_size: Maximum vocabulary size (including special tokens).
        min_freq: Minimum word frequency to include in the vocabulary.
    """

    def __init__(self, max_size: int = 30000, min_freq: int = 2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx: dict[str, int] = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}
        self.idx2word: dict[int, str] = {PAD_IDX: "<pad>", UNK_IDX: "<unk>"}

    def build(self, texts: list[str]) -> None:
        """Build vocabulary from a list of raw text strings.

        Args:
            texts: List of documents (raw strings).
        """
        counter: Counter = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            counter.update(tokens)

        # Sort by frequency (descending), then alphabetically for ties
        sorted_words = sorted(
            counter.items(), key=lambda x: (-x[1], x[0])
        )

        idx = len(self.word2idx)
        for word, freq in sorted_words:
            if freq < self.min_freq:
                continue
            if idx >= self.max_size:
                break
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            idx += 1

    def encode(self, text: str) -> list[int]:
        """Convert a raw text string to a list of token indices.

        Args:
            text: Raw text string.

        Returns:
            List of integer token IDs.
        """
        tokens = self._tokenize(text)
        return [self.word2idx.get(t, UNK_IDX) for t in tokens]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer.

        Lowercases, splits on whitespace and punctuation boundaries.
        """
        text = text.lower()
        # Insert spaces around punctuation so they become separate tokens
        text = re.sub(r"([.,!?;:\"'()\[\]{}\-/])", r" \1 ", text)
        return text.split()

    def __len__(self) -> int:
        return len(self.word2idx)


class AGNewsDataset(Dataset):
    """PyTorch Dataset wrapper for AG News with tokenization and padding.

    Args:
        texts: List of raw text strings.
        labels: List of integer class labels (0-3).
        vocab: Vocabulary for encoding text to indices.
        seq_len: Fixed sequence length (truncate/pad).
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        vocab: Vocabulary,
        seq_len: int = 128,
    ):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return (token_ids, label) for one sample.

        token_ids: LongTensor of shape (seq_len,), padded/truncated.
        label: integer class label.
        """
        token_ids = self.vocab.encode(self.texts[idx])

        # Truncate
        if len(token_ids) > self.seq_len:
            token_ids = token_ids[: self.seq_len]

        # Pad
        padding_len = self.seq_len - len(token_ids)
        if padding_len > 0:
            token_ids = token_ids + [PAD_IDX] * padding_len

        return torch.tensor(token_ids, dtype=torch.long), self.labels[idx]


def get_agnews_loaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    seq_len: int = 128,
    vocab_size: int = 30000,
    min_freq: int = 2,
    num_workers: int = 2,
    pin_memory: bool = True,
    vocab: Optional[Vocabulary] = None,
) -> tuple[DataLoader, DataLoader, Vocabulary]:
    """Create AG News training and test data loaders.

    Downloads the dataset via HuggingFace datasets on first call.

    Args:
        data_dir: Cache directory for dataset download.
        batch_size: Batch size for both loaders.
        seq_len: Fixed sequence length (truncate/pad).
        vocab_size: Maximum vocabulary size.
        min_freq: Minimum word frequency for vocabulary inclusion.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for GPU transfer.
        vocab: Pre-built vocabulary (if None, builds from training data).

    Returns:
        train_loader: DataLoader for training set.
        test_loader: DataLoader for test set.
        vocab: The Vocabulary object (needed for inference / embedding table size).
            Access vocab_size as len(vocab).
    """
    from datasets import load_dataset

    dataset = load_dataset("ag_news", cache_dir=data_dir)

    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    # Build vocabulary from training data
    if vocab is None:
        vocab = Vocabulary(max_size=vocab_size, min_freq=min_freq)
        vocab.build(train_texts)
        print(f"AG News vocabulary size: {len(vocab)}")

    train_dataset = AGNewsDataset(
        texts=train_texts,
        labels=train_labels,
        vocab=vocab,
        seq_len=seq_len,
    )
    test_dataset = AGNewsDataset(
        texts=test_texts,
        labels=test_labels,
        vocab=vocab,
        seq_len=seq_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    return train_loader, test_loader, vocab
