"""A standalone candidate-set Transformer for the public competition dataset."""

import json
from pathlib import Path

import click
import numpy as np
import polars as pl
import torch
from loguru import logger
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

TRAIN_CONFIG = {
    "train_feeds": 0,
    "val_feeds": 0,
    "epochs": 5,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "width": 64,
    "layers": 2,
    "dropout": 0.1,
    "seed": 42,
    "threads": 8,
}

FLOAT_FEATURES = ["f1", "f2", "f3", "f4", "f5"]
CANDIDATE_COLUMNS = ["item_id", "category_id", *FLOAT_FEATURES, "item_id__embedding_idx"]


def read_feeds(split: Path, max_feeds: int = 0) -> pl.DataFrame:
    source = pl.scan_parquet(sorted((split / "impressions").glob("*.parquet")))
    columns = CANDIDATE_COLUMNS + (["target"] if "target" in source.collect_schema() else [])
    source = source.select("feed_id", "user_embedding", *columns)
    if max_feeds:
        source = source.head(max_feeds)
    feeds = source.collect(streaming=True)
    return feeds


def category_mapping(categories: pl.Series) -> dict[int, int]:
    mapping = {value: index + 1 for index, value in enumerate(sorted(categories.unique().to_list()))}
    return mapping


class CandidateData(Dataset):
    """Keep candidate tensors in RAM; read int8 item vectors per feed from memmap."""

    def __init__(self, split: Path, max_feeds: int = 0, categories_mapping: dict[int, int] | None = None):
        feeds = read_feeds(split, max_feeds)
        candidate_counts = feeds["item_id"].list.len().to_numpy()
        flat = feeds.drop("user_embedding").explode(CANDIDATE_COLUMNS + (["target"] if "target" in feeds else []))

        self.categories_mapping = (
            category_mapping(flat["category_id"]) if categories_mapping is None else categories_mapping
        )
        category = flat["category_id"].replace_strict(self.categories_mapping, default=0, return_dtype=pl.Int64)
        self.category = torch.tensor(category.to_numpy())
        self.floats = torch.tensor(flat.select(FLOAT_FEATURES).to_numpy(), dtype=torch.float32)
        self.users = F.normalize(torch.tensor(feeds["user_embedding"].to_list(), dtype=torch.float32), dim=1)
        self.targets = torch.tensor(flat["target"].to_numpy(), dtype=torch.float32) if "target" in flat else None
        self.keys = flat.select("feed_id", "item_id")
        self.offsets = np.insert(candidate_counts.cumsum(), 0, 0)
        self.item_index = flat["item_id__embedding_idx"].to_numpy()

        meta = json.loads((split / "item_embeddings" / "data.meta").read_text())
        self.items = np.memmap(
            split / "item_embeddings" / "data.memmap",
            mode="r",
            dtype=meta["dtype"],
            shape=tuple(meta["shape"]),
            order=meta["order"],
        )

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index: int) -> tuple:
        """One feed: features [N,134], category/target [N], user [128]."""
        rows = slice(self.offsets[index], self.offsets[index + 1])
        indices = self.item_index[rows]
        missing = indices == -1
        items = self.items[np.maximum(indices, 0)].astype(np.float32)
        items[missing] = 0
        features = torch.cat(
            (
                self.floats[rows],
                F.normalize(torch.from_numpy(items), dim=1),
                torch.from_numpy(missing)[:, None],
            ),
            dim=1,
        )
        target = self.targets[rows] if self.targets is not None else None
        return features, self.category[rows], self.users[index], target


def collate_feeds(feeds: list[tuple]) -> tuple:
    """Pad to N=max feed length: float32 features [B,N,134], users [B,128],
    int64 categories [B,N], bool padding [B,N], optional float32 targets [B,N].
    """
    features, categories, users, targets = zip(*feeds, strict=True)
    candidate_counts = torch.tensor([len(x) for x in features])
    padding = torch.arange(candidate_counts.max())[None, :] >= candidate_counts[:, None]
    return (
        pad_sequence(features, batch_first=True),
        pad_sequence(categories, batch_first=True),
        torch.stack(users),
        padding,
        pad_sequence(targets, batch_first=True) if targets[0] is not None else None,
    )


class Ranker(nn.Module):
    def __init__(self, category_count, width=64, layers=2, dropout=0.1):
        super().__init__()
        self.category = nn.Embedding(category_count + 1, 8, padding_idx=0)
        self.candidate = nn.Linear(142, width)
        self.user = nn.Linear(128, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=4, dim_feedforward=2 * width, dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.score = nn.Linear(width, 1)

    def forward(
        self, features: torch.Tensor, category: torch.Tensor, users: torch.Tensor, padding: torch.Tensor
    ) -> torch.Tensor:
        """Float32 features [B,N,134], users [B,128]; int64 category [B,N];
        bool padding [B,N] (True=pad). Return float32 logits [B,N].
        """
        candidates = self.candidate(torch.cat((features, self.category(category)), dim=-1))
        tokens = torch.cat((self.user(users)[:, None, :], candidates), dim=1)
        mask = F.pad(padding, (1, 0), value=False)
        context = self.encoder(tokens, src_key_padding_mask=mask)
        return self.score(context[:, 1:]).squeeze(-1)


def save_epoch(
    model, training, validation, output: Path, config: dict, metrics: dict, history: list, best_loss: float
) -> float:
    """Record epoch metrics and save the checkpoint with the lowest val Logloss."""
    history.append(metrics)
    if metrics["val_logloss"] < best_loss:
        best_loss = metrics["val_logloss"]
        torch.save(
            {
                "state_dict": model.state_dict(),
                "categories_mapping": training.categories_mapping,
                "epoch": metrics["epoch"],
                "model_params": {key: config[key] for key in ("width", "layers", "dropout")},
            },
            output / "model.pt",
        )
    (output / "run.json").write_text(
        json.dumps(
            {
                "config": config,
                "history": history,
                "best_val_logloss": best_loss,
                "train_candidates": len(training.keys),
                "val_candidates": len(validation.keys),
                "parameters": sum(p.numel() for p in model.parameters()),
            },
            indent=2,
        )
        + "\n"
    )
    return best_loss


def fit(training, validation, output, config):
    """Train and save the checkpoint with the lowest validation Logloss."""
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])

    device, batch_size = config["device"], config["batch_size"]
    model_params = {key: config[key] for key in ("width", "layers", "dropout")}
    model = Ranker(len(training.categories_mapping), **model_params).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True, collate_fn=collate_feeds)
    val_loader = DataLoader(validation, batch_size=batch_size, collate_fn=collate_feeds)
    history, best_loss = [], float("inf")

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            features, category, users, padding, target = (x.to(device) for x in batch)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, category, users, padding)[~padding]
            loss = F.binary_cross_entropy_with_logits(logits, target[~padding])
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * logits.numel()

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                features, category, users, padding, target = (x.to(device) for x in batch)
                val_loss += F.binary_cross_entropy_with_logits(
                    model(features, category, users, padding)[~padding], target[~padding], reduction="sum"
                ).item()
        val_loss /= len(validation.keys)
        metrics = {"epoch": epoch, "train_logloss": train_loss / len(training.keys), "val_logloss": val_loss}
        best_loss = save_epoch(model, training, validation, output, config, metrics, history, best_loss)

        logger.info(metrics)


@click.group()
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cpu")
def main(device):
    """Train a Transformer or score the public competition bundle."""
    torch.set_num_threads(TRAIN_CONFIG["threads"])


@main.command()
@click.option("--data", type=click.Path(path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.pass_context
def train(ctx: click.Context, data: Path, output: Path):
    """Save the epoch with the lowest validation Logloss."""
    if output.exists():
        raise FileExistsError(output)

    config = {**TRAIN_CONFIG, "data": str(data), "output": str(output), **ctx.parent.params}
    training = CandidateData(data / "train", config["train_feeds"])
    validation = CandidateData(data / "val", config["val_feeds"], training.categories_mapping)

    fit(training, validation, output, config)


@main.command()
@click.option("--data", type=click.Path(path_type=Path), required=True)
@click.option("--model", "model_path", type=click.Path(path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.pass_context
def predict(ctx: click.Context, data: Path, model_path: Path, output: Path):
    """Write feed_id, item_id, score in the original candidate order."""
    if output.exists():
        raise FileExistsError(output)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    device = ctx.parent.params["device"]
    mapping = checkpoint.get("categories_mapping")
    if mapping is None:
        mapping = {value: index + 1 for index, value in enumerate(checkpoint["categories"])}

    model = Ranker(len(mapping), **checkpoint.get("model_params", {})).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    test = CandidateData(data / "test", categories_mapping=mapping)
    test_loader = DataLoader(test, batch_size=TRAIN_CONFIG["batch_size"], collate_fn=collate_feeds)
    scores = []
    with torch.inference_mode():
        for batch in test_loader:
            features, category, users, padding = (x.to(device) for x in batch[:4])
            scores.append(torch.sigmoid(model(features, category, users, padding)[~padding]).cpu().numpy())
    scores = np.concatenate(scores)

    output.parent.mkdir(parents=True, exist_ok=True)
    test.keys.with_columns(pl.Series("score", scores)).write_parquet(output, compression="zstd")
    logger.info("Wrote {} candidates to {}", len(scores), output)


if __name__ == "__main__":
    main()
