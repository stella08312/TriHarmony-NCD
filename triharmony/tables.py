"""Main Tables 1--4 from complete, equally weighted category metrics."""
import csv
import math
import json
import statistics
from pathlib import Path
from .utils import new_output_directory, sha256_file, write_json

CATEGORIES = ("bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
              "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
              "wood", "zipper")
METRICS = ("NMI", "ARI", "micro_F1", "Disc")

def _cohort_contract():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/main_cohorts.json").read_text())


def _write(path, columns, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

def aggregate(args):
    with Path(args.metrics).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Empty cohort metrics")
    by_identity = {}
    cohort_contract = _cohort_contract()
    for row in rows:
        row["seed"] = int(row["seed"])
        row["n"], row["known_k"] = int(row["n"]), int(row["known_k"])
        contract = cohort_contract.get(row["dataset"], {}).get(row["category"])
        if contract is None:
            raise ValueError("Unknown main-table cohort")
        if row["n"] != contract["sample_count"] or row["known_k"] != contract["known_k"]:
            raise ValueError("Main-table cohort size or known_k differs from the published protocol")
        for name in METRICS:
            row[name] = float(row[name])
            if not math.isfinite(row[name]) or not (-1 <= row[name] <= 1):
                raise ValueError("Metric values must be finite fractions")
        if abs(row["Disc"] - sum(row[name] for name in METRICS[:3]) / 3) > 1e-10:
            raise ValueError("Disc is inconsistent with the three metrics")
        identity = tuple(row[key] for key in ("protocol", "dataset", "category", "seed"))
        if identity in by_identity:
            raise ValueError("Duplicate cohort metrics")
        by_identity[identity] = row
    seeds = sorted({row["seed"] for row in rows})
    expected = {(protocol, dataset, category, seed)
                for protocol in ("table1", "table2") for seed in seeds
                for dataset, categories in (("mvtec", CATEGORIES), ("mtd", ("MTD",)))
                for category in categories}
    if set(by_identity) != expected:
        missing, extra = expected - set(by_identity), set(by_identity) - expected
        raise ValueError("Complete main tables require exactly 32 cohorts per seed; "
                         "missing=%s, extra=%s" % (sorted(missing), sorted(extra)))
    output = new_output_directory(args.output)
    summaries = []
    for protocol in ("table1", "table2"):
        for dataset, categories in (("mvtec", CATEGORIES), ("mtd", ("MTD",))):
            for seed in seeds:
                selected = [by_identity[(protocol, dataset, category, seed)]
                            for category in categories]
                summaries.append({
                    "protocol": protocol, "dataset": dataset, "seed": seed,
                    "category_count": len(selected), "samples": sum(row["n"] for row in selected),
                    **{name: sum(row[name] for row in selected) / len(selected) for name in METRICS},
                })
        summary = [row for row in summaries if row["protocol"] == protocol]
        _write(output / (protocol + "_ours.csv"),
               ("protocol", "dataset", "seed", "category_count", "samples") + METRICS, summary)
        number = 3 if protocol == "table1" else 4
        category_rows = [{key: row[key] for key in ("category", "seed", "n", "known_k") + METRICS}
                         for seed in seeds for category in CATEGORIES
                         for row in [by_identity[(protocol, "mvtec", category, seed)]]]
        _write(output / ("table%d_categories.csv" % number),
               ("category", "seed", "n", "known_k") + METRICS, category_rows)
    if len(seeds) > 1:
        means = []
        for protocol in ("table1", "table2"):
            for dataset in ("mvtec", "mtd"):
                selected = [row for row in summaries
                            if row["protocol"] == protocol and row["dataset"] == dataset]
                means.append({"protocol": protocol, "dataset": dataset, "seed_count": len(seeds),
                              **{name: sum(row[name] for row in selected) / len(seeds)
                                 for name in METRICS}})
        _write(output / "main_seed_means.csv", ("protocol", "dataset", "seed_count") + METRICS, means)
        statistic_rows = []
        statistic_columns = tuple(item for name in METRICS for item in
                                  (name + "_mean", name + "_std_sample"))
        def statistics_for(selected):
            return {item: function([row[name] for row in selected])
                    for name in METRICS
                    for item, function in ((name + "_mean", statistics.mean),
                                           (name + "_std_sample", statistics.stdev))}
        for protocol in ("table1", "table2"):
            setting_rows = []
            for dataset in ("mvtec", "mtd"):
                selected = [row for row in summaries
                            if row["protocol"] == protocol and row["dataset"] == dataset]
                row = {"protocol": protocol, "dataset": dataset, "seed_count": len(seeds),
                       **statistics_for(selected)}
                setting_rows.append(row)
                statistic_rows.append(row)
            _write(output / (protocol + "_mean_std.csv"),
                   ("protocol", "dataset", "seed_count") + statistic_columns, setting_rows)
            number = 3 if protocol == "table1" else 4
            category_statistics = []
            for category in CATEGORIES:
                selected = [by_identity[(protocol, "mvtec", category, seed)] for seed in seeds]
                category_statistics.append({
                    "category": category, "seed_count": len(seeds),
                    "n": selected[0]["n"], "known_k": selected[0]["known_k"],
                    **statistics_for(selected),
                })
            _write(output / ("table%d_categories_mean_std.csv" % number),
                   ("category", "seed_count", "n", "known_k") + statistic_columns,
                   category_statistics)
        _write(output / "main_seed_statistics.csv",
               ("protocol", "dataset", "seed_count") + statistic_columns, statistic_rows)
        per_seed_disc = [
            statistics.mean(row["Disc"] for row in summaries if row["seed"] == seed)
            for seed in seeds
        ]
        write_json(output / "overall_mean_disc.json", {
            "seeds": seeds, "per_seed": dict(zip(map(str, seeds), per_seed_disc)),
            "mean": statistics.mean(per_seed_disc),
            "std_sample": statistics.stdev(per_seed_disc),
            "definition": "Within each seed, mean Disc across two protocols x two datasets; then mean and sample SD across seeds",
        })
    write_json(output / "table_manifest.json", {
        "schema_version": 1, "input_metrics_sha256": sha256_file(args.metrics),
        "seeds": seeds, "cohorts_per_seed": 32,
        "sample_standard_deviation_ddof": 1 if len(seeds) > 1 else None,
        "category_aggregation": "unweighted mean across all 15 MVTec categories, including K=1 toothbrush",
        "mtd_aggregation": "one complete MTD cohort per protocol",
        "metric_units": "fraction; multiply by 100 for percent",
        "scope": "Tri-Harmony main Full rows only; no baseline or AUPRO recomputation",
    })
    return {"output": str(output), "seeds": seeds, "cohorts": len(rows)}
