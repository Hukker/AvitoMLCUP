"""Standalone CatBoost baseline for the public competition/ bundle."""

import json
from functools import partial
from pathlib import Path

import click
import numpy as np
import polars as pl
from catboost import CatBoostRanker, Pool
from loguru import logger

# Snapshot of src/train_ranker/defaults.py: MODEL_PARAMS and FIT_PARAMS.
MODEL_PARAMS = {
    "boosting_type": "Plain",
    "max_ctr_complexity": 1,
    "objective": "YetiRank",
    "task_type": "GPU",
    "eval_metric": "NDCG:top=30;type=Base",
    "early_stopping_rounds": 100,
    "gpu_ram_part": 0.8,
    "learning_rate": 0.4,
    "num_trees": 2000,
    "random_state": 666,
    "l2_leaf_reg": 5,
    "nan_mode": "Min",
    "allow_writing_files": False,
    "thread_count": 8,
    "devices": "0",
}
FIT_PARAMS = {"verbose": 100, "use_best_model": True, "early_stopping_rounds": 50}
DATA_PARAMS = {"train_feeds": 0, "val_feeds": 0}

BIN_FEATURES = ["f1", "f2", "f3", "f4", "f5"]
CAT_FEATURES = ["category_id"]
FLOAT_FEATURES = [*BIN_FEATURES, "embedding_dot", "embedding_cosine", "embedding_missing"]
FEATURES = [*FLOAT_FEATURES, *CAT_FEATURES]
CANDIDATE_COLUMNS = ["item_id", *CAT_FEATURES, *BIN_FEATURES, "item_id__embedding_idx"]
EMBEDDING_FEATURES = pl.Struct(
    {
        "embedding_dot": pl.List(pl.Float32),
        "embedding_cosine": pl.List(pl.Float32),
        "embedding_missing": pl.List(pl.Float32),
    }
)


def embedding_features(batch: pl.Series, embeddings: np.memmap) -> pl.Series:
    """Compute candidate similarities once per feed; -1 denotes a missing item."""
    features = {name: [] for name in EMBEDDING_FEATURES.to_schema()}
    for row in batch:
        user = np.asarray(row["user_embedding"], dtype=np.float32)
        indices = np.asarray(row["item_id__embedding_idx"])
        missing = indices == -1
        # Convert int8 user and item vectors to float32 before multiplication.
        items = embeddings[np.maximum(indices, 0)].astype(np.float32)
        items[missing] = 0
        dot = np.einsum("ij,j->i", items, user)
        denominator = np.linalg.norm(items, axis=1) * np.linalg.norm(user)
        cosine = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0)
        features["embedding_dot"].append(dot)
        features["embedding_cosine"].append(cosine)
        features["embedding_missing"].append(missing.astype(np.float32))
    result = pl.DataFrame(features, schema=EMBEDDING_FEATURES.to_schema()).to_struct()
    return result


def candidate_dataset(split: Path, max_feeds: int = 0) -> pl.LazyFrame:
    """Limit complete feeds, then expand their candidate arrays using native Polars."""
    source = pl.scan_parquet(sorted((split / "impressions").glob("*.parquet")))
    columns = CANDIDATE_COLUMNS + (["target"] if "target" in source.collect_schema() else [])
    source = source.select("feed_id", "user_embedding", *columns)
    if max_feeds:
        source = source.head(max_feeds)

    meta = json.loads((split / "item_embeddings" / "data.meta").read_text())
    embeddings = np.memmap(
        split / "item_embeddings" / "data.memmap",
        mode="r",
        dtype=meta["dtype"],
        shape=tuple(meta["shape"]),
        order=meta["order"],
    )
    candidates = (
        source.with_columns(
            pl.struct("user_embedding", "item_id__embedding_idx")
            .map_batches(
                partial(embedding_features, embeddings=embeddings),
                return_dtype=EMBEDDING_FEATURES,
                is_elementwise=True,
            )
            .alias("embedding_features")
        )
        .unnest("embedding_features")
        .explode([*columns, "embedding_dot", "embedding_cosine", "embedding_missing"])
        .drop("user_embedding", "item_id__embedding_idx")
    )
    return candidates


def read_pool(split: Path, max_feeds: int) -> Pool:
    data = candidate_dataset(split, max_feeds).collect(streaming=True).rechunk()
    pool = Pool(data.select(FEATURES), label=data["target"], cat_features=CAT_FEATURES, group_id=data["feed_id"])
    return pool


def predict_scores(batch: pl.DataFrame, model: CatBoostRanker) -> pl.DataFrame:
    features = Pool(batch.select(FEATURES), cat_features=CAT_FEATURES)
    scores = model.predict(features, thread_count=MODEL_PARAMS["thread_count"])
    result = batch.select("feed_id", "item_id").with_columns(pl.Series("score", scores, dtype=pl.Float32))
    return result


@click.group()
def main():
    """Train a CatBoost baseline or score the public competition bundle."""


@main.command()
@click.option("--data", type=click.Path(path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def train(data: Path, output: Path):
    """Train YetiRank on train, select by validation NDCG@30, save model.cbm."""
    config = {"data": str(data), "output": str(output), **DATA_PARAMS}
    if output.exists():
        raise FileExistsError(output)
    train_pool = read_pool(data / "train", DATA_PARAMS["train_feeds"])
    val_pool = read_pool(data / "val", DATA_PARAMS["val_feeds"])

    model = CatBoostRanker(**MODEL_PARAMS)
    model.fit(train_pool, eval_set=val_pool, **FIT_PARAMS)

    quality = model.eval_metrics(
        val_pool,
        ["NDCG", "NDCG:top=30", "NDCG:top=10"],
        eval_period=model.tree_count_,
        thread_count=MODEL_PARAMS["thread_count"],
    )
    train_groups = train_pool.get_group_id_hash()
    val_groups = val_pool.get_group_id_hash()
    metrics = {
        "val_ndcg": quality["NDCG:type=Base"][-1],
        "val_ndcg_at_30": quality["NDCG:top=30;type=Base"][-1],
        "val_ndcg_at_10": quality["NDCG:top=10;type=Base"][-1],
        "train_feeds": int(np.count_nonzero(train_groups[1:] != train_groups[:-1]) + 1),
        "train_candidates": train_pool.num_row(),
        "val_feeds": int(np.count_nonzero(val_groups[1:] != val_groups[:-1]) + 1),
        "val_candidates": val_pool.num_row(),
        "trees": model.tree_count_,
    }

    output.mkdir(parents=True, exist_ok=False)
    model.save_model(str(output / "model.cbm"))
    report = {
        "config": config,
        "model_params": model.get_all_params(),
        "features": FEATURES,
        "fit_params": FIT_PARAMS,
        "metrics": metrics,
    }
    (output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info(metrics)


@main.command()
@click.option("--data", type=click.Path(path_type=Path), required=True)
@click.option("--model", "model_path", type=click.Path(path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def predict(data: Path, model_path: Path, output: Path):
    """Write raw ranking scores for the complete test split."""
    if output.exists():
        raise FileExistsError(output)

    model = CatBoostRanker()
    model.load_model(str(model_path))
    output.parent.mkdir(parents=True, exist_ok=True)

    candidates = candidate_dataset(data / "test")
    schema = candidates.select("feed_id", "item_id").collect_schema()
    schema["score"] = pl.Float32
    predictions = candidates.map_batches(
        partial(predict_scores, model=model),
        schema=schema,
        projection_pushdown=False,
        streamable=True,
    )
    predictions.sink_parquet(output, compression="zstd", maintain_order=True)
    logger.info("Predictions saved to {}", output)


if __name__ == "__main__":
    main()
