"""WikiText-2 data loader for language modeling.

WikiText-2: ~2M tokens for training, standard benchmark for language models.

Uses HuggingFace datasets for download and a BPE tokenizer (GPT-2 tokenizer
via HuggingFace tokenizers). Text is concatenated into a single stream and
chunked into fixed-length sequences of 256 tokens.

For language modeling, each sample is (input_ids, target_ids) where
target_ids = input_ids shifted by one position (next-token prediction).

Reference: DKA Build Guide, Sections 4.2, 5.4.
"""

from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset


class WikiText2Dataset(Dataset):
    """PyTorch Dataset for WikiText-2 language modeling.

    Concatenates all text into a single token stream, then chunks it into
    non-overlapping sequences of fixed length. Each sample returns:
        input_ids:  tokens[i * seq_len : (i+1) * seq_len]
        target_ids: tokens[i * seq_len + 1 : (i+1) * seq_len + 1]

    Args:
        token_ids: 1D LongTensor of all token IDs (concatenated corpus).
        seq_len: Length of each chunk.
    """

    def __init__(self, token_ids: torch.Tensor, seq_len: int = 256):
        self.seq_len = seq_len
        # Trim to a multiple of (seq_len + 1) so every sample has a full target
        n_tokens = ((len(token_ids) - 1) // seq_len) * seq_len
        self.data = token_ids[: n_tokens + 1]

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input_ids, target_ids) for one chunk.

        input_ids:  LongTensor of shape (seq_len,)
        target_ids: LongTensor of shape (seq_len,), shifted by 1.
        """
        start = idx * self.seq_len
        end = start + self.seq_len
        input_ids = self.data[start:end]
        target_ids = self.data[start + 1 : end + 1]
        return input_ids, target_ids


def _tokenize_split(
    texts: list[str],
    tokenizer,
) -> torch.Tensor:
    """Tokenize a list of text strings and concatenate into a single stream.

    Filters out empty lines and article headers (lines starting with ' = ').

    Args:
        texts: List of text strings (one per line/paragraph).
        tokenizer: HuggingFace tokenizer with an encode() method.

    Returns:
        1D LongTensor of all token IDs concatenated.
    """
    all_ids: list[int] = []
    for text in texts:
        text = text.strip()
        # Skip empty lines and article-level headers
        if len(text) == 0:
            continue
        ids = tokenizer.encode(text)
        all_ids.extend(ids)

    return torch.tensor(all_ids, dtype=torch.long)


def get_wikitext2_loaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    seq_len: int = 256,
    num_workers: int = 2,
    pin_memory: bool = True,
    tokenizer_name: str = "gpt2",
    tokenizer=None,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """Create WikiText-2 train/val/test data loaders for language modeling.

    Downloads the dataset via HuggingFace datasets on first call. Uses a
    pretrained BPE tokenizer (GPT-2 by default) for subword tokenization.

    Args:
        data_dir: Cache directory for dataset download.
        batch_size: Batch size for all loaders.
        seq_len: Sequence length for chunking (default 256).
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for GPU transfer.
        tokenizer_name: Name of HuggingFace pretrained tokenizer.
        tokenizer: Pre-initialized tokenizer (overrides tokenizer_name).

    Returns:
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        test_loader: DataLoader for test set.
        vocab_size: Size of the tokenizer vocabulary (needed for embedding
            table and LM head).
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # Load tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # GPT-2 tokenizer has no pad token by default; set one
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    vocab_size = len(tokenizer)

    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=data_dir)

    # Tokenize each split into a single concatenated stream
    train_ids = _tokenize_split(dataset["train"]["text"], tokenizer)
    val_ids = _tokenize_split(dataset["validation"]["text"], tokenizer)
    test_ids = _tokenize_split(dataset["test"]["text"], tokenizer)

    print(
        f"WikiText-2 token counts — train: {len(train_ids):,}, "
        f"val: {len(val_ids):,}, test: {len(test_ids):,}"
    )
    print(f"Vocabulary size: {vocab_size:,}")

    train_dataset = WikiText2Dataset(train_ids, seq_len=seq_len)
    val_dataset = WikiText2Dataset(val_ids, seq_len=seq_len)
    test_dataset = WikiText2Dataset(test_ids, seq_len=seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
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

    return train_loader, val_loader, test_loader, vocab_size
