from __future__ import annotations

import argparse
import json
import re
import tarfile
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
PAD_ID, UNK_ID, BOS_ID, EOS_ID = range(4)

BASE_URL = "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi"
FILES = ["train.en", "train.vi", "tst2012.en", "tst2012.vi", "tst2013.en", "tst2013.vi"]
MIRROR_URL = "https://github.com/stefan-it/nmt-en-vi/raw/master/data"
ARCHIVES = ["train-en-vi.tgz", "dev-2012-en-vi.tgz", "test-2013-en-vi.tgz"]


def tokenize(text: str) -> list[str]:
    """Tách token nhẹ; dữ liệu Stanford vốn đã được tách từ cơ bản."""
    return re.findall(r"\w+|[^\w\s]", text.strip().lower(), flags=re.UNICODE)


@dataclass
class Vocabulary:
    itos: list[str]

    def __post_init__(self) -> None:
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    def encode(self, text: str, add_boundaries: bool = True) -> list[int]:
        ids = [self.stoi.get(token, UNK_ID) for token in tokenize(text)]
        return [BOS_ID, *ids, EOS_ID] if add_boundaries else ids

    def decode(self, ids: Iterable[int]) -> str:
        tokens: list[str] = []
        for index in ids:
            if index == EOS_ID:
                break
            if index not in (PAD_ID, BOS_ID):
                tokens.append(self.itos[index] if index < len(self.itos) else UNK_TOKEN)
        return re.sub(r"\s+([,.;:!?%])", r"\1", " ".join(tokens))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.itos, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def __len__(self) -> int:
        return len(self.itos)


def build_vocabulary(lines: Iterable[str], max_size: int = 15000, min_freq: int = 2) -> Vocabulary:
    counts: Counter[str] = Counter()
    for line in lines:
        counts.update(tokenize(line))
    words = [word for word, count in counts.most_common() if count >= min_freq]
    return Vocabulary(SPECIAL_TOKENS + words[: max(0, max_size - len(SPECIAL_TOKENS))])


def download_iwslt(data_dir: str | Path) -> Path:
    destination = Path(data_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if all((destination / filename).exists() for filename in FILES):
        return destination

    for archive_name in ARCHIVES:
        archive_path = destination / archive_name
        print(f"Downloading {archive_name} ...")
        urllib.request.urlretrieve(f"{MIRROR_URL}/{archive_name}", archive_path)
        with tarfile.open(archive_path, "r:gz") as archive:
            members = [member for member in archive.getmembers() if Path(member.name).name in FILES]
            if not members:
                raise RuntimeError(f"Archive {archive_name} does not contain expected corpus files.")
            for member in members:
                source = archive.extractfile(member)
                if source is None:
                    continue
                (destination / Path(member.name).name).write_bytes(source.read())
        archive_path.unlink()

    missing = [filename for filename in FILES if not (destination / filename).exists()]
    if missing:
        raise RuntimeError(f"Missing corpus files after extraction: {', '.join(missing)}")
    return destination


def load_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


class ParallelDataset(Dataset):
    def __init__(self, source_lines: Sequence[str], target_lines: Sequence[str], source_vocab: Vocabulary,
                 target_vocab: Vocabulary, max_length: int = 80) -> None:
        if len(source_lines) != len(target_lines):
            raise ValueError("Hai phía của corpus phải có cùng số câu.")
        self.examples: list[tuple[list[int], list[int]]] = []
        self.raw_examples: list[tuple[str, str]] = []
        for source, target in zip(source_lines, target_lines):
            source_ids = source_vocab.encode(source)
            target_ids = target_vocab.encode(target)
            if 2 < len(source_ids) <= max_length and 2 < len(target_ids) <= max_length:
                self.examples.append((source_ids, target_ids))
                self.raw_examples.append((source.strip(), target.strip()))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self.examples[index]


def collate_batch(batch: list[tuple[list[int], list[int]]]) -> tuple[torch.Tensor, torch.Tensor]:
    source = [torch.tensor(item[0], dtype=torch.long) for item in batch]
    target = [torch.tensor(item[1], dtype=torch.long) for item in batch]
    return (
        torch.nn.utils.rnn.pad_sequence(source, batch_first=True, padding_value=PAD_ID),
        torch.nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=PAD_ID),
    )


def prepare_data(data_dir: str | Path, vocab_size: int = 15000) -> tuple[Vocabulary, Vocabulary]:
    root = download_iwslt(data_dir)
    source_vocab_path = root / "vocab.en.json"
    target_vocab_path = root / "vocab.vi.json"
    if source_vocab_path.exists() and target_vocab_path.exists():
        return Vocabulary.load(source_vocab_path), Vocabulary.load(target_vocab_path)
    source_vocab = build_vocabulary(load_lines(root / "train.en"), max_size=vocab_size)
    target_vocab = build_vocabulary(load_lines(root / "train.vi"), max_size=vocab_size)
    source_vocab.save(source_vocab_path)
    target_vocab.save(target_vocab_path)
    return source_vocab, target_vocab


def make_loader(data_dir: str | Path, split: str, source_vocab: Vocabulary, target_vocab: Vocabulary,
                batch_size: int, max_length: int, shuffle: bool = False) -> DataLoader:
    names = {"train": "train", "dev": "tst2012", "test": "tst2013"}
    if split not in names:
        raise ValueError(f"Split không hợp lệ: {split}")
    prefix = Path(data_dir) / names[split]
    dataset = ParallelDataset(load_lines(prefix.with_suffix(".en")), load_lines(prefix.with_suffix(".vi")),
                              source_vocab, target_vocab, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tải và chuẩn bị IWSLT 2015 English-Vietnamese")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--vocab-size", type=int, default=15000)
    args = parser.parse_args()
    source_vocab, target_vocab = prepare_data(args.data_dir, args.vocab_size)
    print(f"Data ready: vocab EN={len(source_vocab):,}, vocab VI={len(target_vocab):,}")


if __name__ == "__main__":
    main()
