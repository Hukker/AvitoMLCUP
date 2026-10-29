"""Calculate the same all/public/private metrics as the final model comparison.

Run from baselines/ after uv sync; GPU and sample_submission are not needed:
    python calculate_metrics.py \
        --predictions neural_submission.parquet --answers test_answers.parquet

Predictions: feed_id, item_id, score.
Answers: feed_id, item_id, target (0/1), leaderboard_split (public/private).
Rows are joined by keys, so their input order does not matter.
NDCG uses binary gains, log2 discounts and an unweighted average over feeds.
Equal scores use pessimistic ties; a feed without positives receives NDCG=1.
Every metric is independently verified with NumPy, as in the original evaluator.
"""

import json
from pathlib import Path

import click
import numpy as np
import polars as pl
from catboost.utils import eval_metric

METRICS = [
    ("ndcg", "NDCG:type=Base", None),
    ("ndcg_at_10", "NDCG:top=10;type=Base", 10),
    ("ndcg_at_30", "NDCG:top=30;type=Base", 30),
]


def metrics(frame: pl.DataFrame) -> dict:
    frame = frame.sort("feed_id")
    targets = frame["target"].to_numpy()
    scores = frame["score"].to_numpy().astype(np.float64)
    groups = frame["feed_id"].to_numpy()
    boundaries = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1, len(groups)]
    result = {
        "rows": len(frame),
        "groups": len(boundaries) - 1,
        "positives": int(targets.sum()),
    }
    labels, predictions, group_ids = targets.tolist(), scores.tolist(), groups.tolist()
    for name, metric, _ in METRICS:
        result[name] = eval_metric(labels, predictions, metric, group_id=group_ids, thread_count=8)[0]

    independent = np.zeros((result["groups"], len(METRICS)))
    empty_groups = 0
    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        labels = targets[start:end]
        ranked = labels[np.lexsort((labels, -scores[start:end]))]
        ideal = np.sort(labels)[::-1]
        discounts = 1 / np.log2(np.arange(end - start) + 2)
        empty_groups += int(ideal.sum() == 0)
        for j, (_, _, cutoff) in enumerate(METRICS):
            denominator = np.dot(ideal[:cutoff], discounts[:cutoff])
            independent[i, j] = np.dot(ranked[:cutoff], discounts[:cutoff]) / denominator if denominator else 1.0

    for (name, _, _), value in zip(METRICS, independent.mean(axis=0), strict=True):
        if abs(result[name] - value) >= 1e-10:
            raise RuntimeError(f"Independent {name} check failed: CatBoost={result[name]}, NumPy={value}")
    result["groups_without_positives"] = empty_groups
    result["independent_numpy_check"] = "passed"
    return result


@click.command()
@click.option("--predictions", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--answers", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
def main(predictions: Path, answers: Path):
    """Print JSON metrics for the full test and each leaderboard split."""
    keys = ["feed_id", "item_id"]
    predicted = pl.read_parquet(predictions, columns=[*keys, "score"])
    labeled = pl.read_parquet(answers, columns=[*keys, "target", "leaderboard_split"])
    if not len(labeled) or len(predicted) != len(labeled):
        raise click.ClickException("Predictions and answers must be nonempty and have the same number of rows.")
    if predicted.null_count().to_numpy().any() or labeled.null_count().to_numpy().any():
        raise click.ClickException("Predictions and answers must not contain nulls.")
    if not predicted["score"].is_finite().all() or not labeled["target"].is_in([0, 1]).all():
        raise click.ClickException("Scores must be finite and targets must be binary (0/1).")
    if set(labeled["leaderboard_split"].unique().to_list()) != {"public", "private"}:
        raise click.ClickException("Answers must contain both public and private leaderboard splits.")
    splits_per_feed = labeled.group_by("feed_id").agg(pl.col("leaderboard_split").n_unique())
    if splits_per_feed["leaderboard_split"].max() != 1:
        raise click.ClickException("Every feed_id must belong to exactly one leaderboard split.")

    # Polars enforces unique keys on both sides; equal counts and no missing
    # scores ensure exact key coverage, including the absence of extra keys.
    joined = labeled.join(predicted, on=keys, how="left", validate="1:1")
    if joined["score"].null_count():
        raise click.ClickException("Prediction keys do not match the answer keys.")
    result = {"all": metrics(joined)}
    for split in ("public", "private"):
        result[split] = metrics(joined.filter(pl.col("leaderboard_split") == split))
    click.echo(json.dumps({"predictions": str(predictions), "answers": str(answers), "metrics": result}, indent=2))


if __name__ == "__main__":
    main()
