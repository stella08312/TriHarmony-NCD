"""Post-hoc evaluation: validate every locked prediction before reading GT."""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple
import csv
import re
from pathlib import Path
from .utils import (new_output_directory, read_json, resolve_relative,
                 sha256_file, write_json)

if TYPE_CHECKING:
    import numpy as np

COLUMNS = ["protocol", "dataset", "category", "seed", "n", "known_k",
           "NMI", "ARI", "micro_F1", "Disc", "prediction_sha256"]

def validate_prediction(value):
    if value.get("epoch") != 40 or value.get("checkpoint") != "fixed_epoch_40":
        raise ValueError("Evaluation requires the fixed final epoch-40 prediction")
    for flag in ("target_gt_used", "target_metric_used", "target_normal_used"):
        if value.get(flag) is not False:
            raise ValueError("Prediction must declare %s=false" % flag)
    if value.get("variant") != "learned":
        raise ValueError("Only learned Full predictions belong to the main tables")
    if type(value.get("seed")) is not int or not 0 <= value["seed"] < 2**32:
        raise ValueError("Prediction seed is missing or invalid")
    if value.get("protocol") not in ("table1", "table2"):
        raise ValueError("Unknown prediction protocol")
    if value.get("dataset") not in ("mvtec", "mtd") or not value.get("category"):
        raise ValueError("Missing prediction dataset/category")
    keys, predicted = value.get("sample_keys"), value.get("prediction")
    if not isinstance(keys, list) or not isinstance(predicted, list) or len(keys) != len(predicted):
        raise ValueError("Prediction and sample-key lengths differ")
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Prediction contains empty/duplicate sample keys")
    if any(not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key) for key in keys):
        raise ValueError("Evaluation accepts only opaque SHA-256 sample keys")
    if any(type(label) is not int or label < 0 for label in predicted):
        raise ValueError("Cluster assignments must be nonnegative integers")
    known_k = value.get("known_k")
    if type(known_k) is not int or known_k < 1 or known_k != len(set(predicted)):
        raise ValueError("Occupied cluster count must equal a-priori known_k")
    if value.get("occupied_k") != known_k:
        raise ValueError("Inconsistent occupied_k metadata")
    return value

def lock_predictions(args):
    output = Path(args.output).resolve()
    rows = []
    identities = set()
    for directory in args.runs:
        path = Path(directory).resolve()
        if path.is_dir():
            path = path / "learned_predictions.json"
        value = validate_prediction(read_json(path))
        identity = tuple(value[k] for k in ("protocol", "dataset", "category", "seed"))
        if identity in identities:
            raise ValueError("Duplicate cohort/seed: %r" % (identity,))
        identities.add(identity)
        import os
        rows.append({"predictions": os.path.relpath(str(path), str(output.parent)),
                     "sha256": sha256_file(path)})
    if not rows:
        raise ValueError("No predictions to lock")
    write_json(output, {"schema_version": 1, "runs": rows})
    return {"manifest": str(output), "runs": len(rows)}

def _locked_predictions(path):
    manifest = read_json(path)
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Prediction manifest must contain a nonempty runs list")
    locked, identities = [], set()
    for row in runs:
        prediction_path = resolve_relative(Path(path).resolve().parent, row["predictions"])
        digest = sha256_file(prediction_path)
        if digest != row.get("sha256"):
            raise ValueError("Prediction SHA-256 mismatch: %s" % prediction_path.name)
        value = validate_prediction(read_json(prediction_path))
        identity = tuple(value[k] for k in ("protocol", "dataset", "category", "seed"))
        if identity in identities:
            raise ValueError("Duplicate cohort/seed in locked predictions")
        identities.add(identity)
        locked.append((value, digest))
    return locked

def evaluate(args):
    locked = _locked_predictions(args.predictions_manifest)
    # This is deliberately the first read of the evaluation-label file.
    document = read_json(args.labels)
    labels = document.get("labels")
    if not isinstance(labels, dict) or not labels:
        raise ValueError("Labels JSON must contain a labels map keyed by SHA-256 sample key")
    import numpy as np
    rows = []
    for prediction, digest in locked:
        try:
            truth = [labels[key] for key in prediction["sample_keys"]]
        except KeyError as error:
            raise ValueError("Evaluation label missing for a locked sample") from error
        if any(not isinstance(value, (str, int)) or isinstance(value, bool) for value in truth):
            raise ValueError("Evaluation classes must be string or integer names")
        if len(set(map(str, truth))) != prediction["known_k"]:
            raise ValueError("Evaluation truth and predeclared known_k disagree")
        metrics = clustering_metrics(encode_labels(list(map(str, truth))),
                                     np.asarray(prediction["prediction"], dtype=np.int64))
        rows.append({**{key: prediction[key] for key in
                         ("protocol", "dataset", "category", "seed", "known_k")},
                     "n": len(truth), **metrics, "prediction_sha256": digest})
    output = new_output_directory(args.output)
    with (output / "cohort_metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "evaluation_manifest.json", {
        "schema_version": 1, "evaluation_after_predictions_locked": True,
        "prediction_manifest_sha256": sha256_file(args.predictions_manifest),
        "evaluation_labels_sha256": sha256_file(args.labels),
        "cohort_metrics_sha256": sha256_file(output / "cohort_metrics.csv"),
        "runs": len(rows), "metric_units": "fraction",
        "Disc": "(NMI + ARI + Hungarian-matched micro-F1) / 3",
    })
    return {"output": str(output), "runs": len(rows)}


def encode_labels(values: List[str]) -> np.ndarray:
    import numpy as np
    mapping = {name: index for index, name in enumerate(sorted(set(values)))}
    return np.asarray([mapping[value] for value in values], dtype=np.int64)


def hungarian_micro_f1(truth: np.ndarray, prediction: np.ndarray
                       ) -> Tuple[float, np.ndarray, Dict[int, int]]:
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    true_values = sorted(np.unique(truth))
    predicted_values = sorted(np.unique(prediction))
    table = np.zeros((len(true_values), len(predicted_values)), dtype=np.int64)
    for row, true_value in enumerate(true_values):
        for column, predicted_value in enumerate(predicted_values):
            table[row, column] = int(np.sum(
                (truth == true_value) & (prediction == predicted_value)))
    rows, columns = linear_sum_assignment(-table)
    mapping = {int(predicted_values[column]): int(true_values[row])
               for row, column in zip(rows.tolist(), columns.tolist())}
    aligned = np.asarray([mapping[int(value)] for value in prediction], dtype=np.int64)
    return float(np.mean(aligned == truth)), aligned, mapping


def clustering_metrics(truth: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    nmi = float(normalized_mutual_info_score(truth, prediction))
    ari = float(adjusted_rand_score(truth, prediction))
    micro, _, _ = hungarian_micro_f1(truth, prediction)
    return {"NMI": nmi, "ARI": ari, "micro_F1": micro,
            "Disc": float((nmi + ari + micro) / 3.0)}


def macro_setting(rows: List[Dict[str, object]]) -> Dict[str, float]:
    import numpy as np
    return {metric: float(np.mean([float(row[metric]) for row in rows]))
            for metric in ("NMI", "ARI", "micro_F1", "Disc")}
