#!/usr/bin/env python3
"""Reconstruct nested, fixed-size and external predictions from measured betas.

Input is a verified array cache, or public signal matrices and sample metadata
from which a new cache is prepared. Reference predictions are never read here.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from array_io import load_cache, sha256, write_json
from atomic_io import prepare_cache_atomic, publish_directory, sync_tree, work_directory_name
from classification_core import (
    UnresolvedPrediction, choose_size, classify, empirical_metrics, rank_and_centroids,
)


def write_csv(frame: pd.DataFrame, path: Path):
    compression = {"method": "gzip", "mtime": 0} if path.suffix == ".gz" else None
    frame.to_csv(path, index=False, compression=compression, float_format="%.17g")


def fit_model(beta, samples, indices, classes, probe_ids, sizes, model_id,
              directory, model_manifest, context, export_full_ranking=False):
    order, scores, means = rank_and_centroids(
        beta, indices, samples.population.to_numpy(), classes, probe_ids)
    maximum = max(sizes)
    selected = order[:maximum]
    selected_means = means[selected].copy()
    artifact = directory / "models" / (model_id + ".npz")
    np.savez_compressed(
        artifact, probe_ids=probe_ids[selected], ranking_scores=scores[selected],
        centroids=selected_means, class_order=np.asarray(classes, dtype=str),
        training_gsm=samples.iloc[indices].gsm.to_numpy(dtype=str))
    membership = samples.iloc[indices]
    meta = {
        "model_id": model_id, **context, "model_file": str(artifact.relative_to(directory)),
        "training_gsm": membership.gsm.tolist(),
        "training_donor_ids": sorted(membership.donor_id.astype(int).unique().tolist()),
        "training_class_counts": {str(k): int(v) for k, v in membership.population.value_counts().items()},
        "probe_universe_size": len(probe_ids), "ranked_features_exported": maximum,
        "ranking_ties_at_evaluated_cutoffs": {
            str(size): int(np.count_nonzero(scores == scores[order[size - 1]])) for size in sizes
        },
    }
    model_manifest.append(meta)
    if export_full_ranking:
        write_csv(pd.DataFrame({
            "probe_id": probe_ids[order], "R_discovery": scores[order],
            "discovery_rank": np.arange(1, len(order) + 1),
        }), directory / "external_discovery_ranking.csv.gz")
        signature = pd.DataFrame({"probe_id": probe_ids[selected[:100]],
                                  "R_discovery": scores[selected[:100]]})
        for column, cell_type in enumerate(classes):
            signature["centroid_" + cell_type] = selected_means[:100, column]
        write_csv(signature, directory / "external_100_signature_and_centroids.csv")
    return {"model_id": model_id, "feature_indices": selected,
            "centroids": selected_means, "classes": list(classes)}


def evaluate_model(beta, samples, indices, model, sizes, methods, context, distance_rows):
    selected_samples = samples.iloc[indices].reset_index(drop=True)
    # No evaluation labels are passed to classify().
    x = np.asarray(beta[np.ix_(model["feature_indices"], np.asarray(indices, dtype=int))])
    results = []
    for size in sizes:
        for method in methods:
            try:
                predictions, cost, margin = classify(
                    x[:size], model["centroids"][:size], model["classes"], method)
            except UnresolvedPrediction as exc:
                raise UnresolvedPrediction(exc.reason, {
                    **exc.details, "model_id": model["model_id"], "n_cpg": size,
                    "method": method, "evaluation_gsm_in_order": selected_samples.gsm.tolist(),
                }) from exc
            for position, sample in enumerate(selected_samples.to_dict("records")):
                row = {
                    **context, "model_id": model["model_id"], "n_cpg": int(size),
                    "method": method, "gsm": sample["gsm"], "sample_name": sample["sample_name"],
                    "donor": sample["donor_id"], "true": sample["population"],
                    "predicted": str(predictions[position]),
                    "correct": bool(predictions[position] == sample["population"]),
                    "minimum_distance": float(cost[position].min()),
                    "distance_margin": float(margin[position]),
                }
                results.append(row)
                for column, cell_type in enumerate(model["classes"]):
                    distance_rows.append({
                        **context, "model_id": model["model_id"], "n_cpg": int(size),
                        "method": method, "gsm": sample["gsm"],
                        "centroid_class": cell_type, "distance": float(cost[position, column]),
                        "distance_definition": "squared_euclidean" if method == "euclidean" else "one_minus_pearson",
                    })
    return results


def check_partition(training, validation, outer_test, expected_train):
    arrays = [np.asarray(x, dtype=int) for x in (training, validation, outer_test)]
    if len(arrays[0]) != expected_train or len(arrays[1]) != 7 or len(arrays[2]) != 7:
        raise ValueError("A nested partition has an unexpected sample count.")
    for a, b in ((0, 1), (0, 2), (1, 2)):
        if np.intersect1d(arrays[a], arrays[b]).size:
            raise ValueError("Training, inner validation and outer test samples overlap.")


def add_membership(rows, model_id, samples, roles, cohort="discovery"):
    for index, role in enumerate(roles):
        sample = samples.iloc[index]
        rows.append({
            "model_id": model_id, "cohort": cohort, "gsm": sample.gsm,
            "sample_name": sample.sample_name, "population": sample.population,
            "donor_id": sample.donor_id, "role": role,
        })


def export_metrics(directory, nested, fixed, external, classes7, classes6):
    metric_rows, confusion_rows, class_rows = [], [], []
    evaluations = [("nested_discovery", None, "euclidean", nested, classes7)]
    for size, frame in fixed.groupby("n_cpg", sort=True):
        evaluations.append(("fixed_size_discovery", int(size), "euclidean", frame, classes7))
    for (size, method), frame in external.groupby(["n_cpg", "method"], sort=True):
        evaluations.append(("external", int(size), str(method), frame, classes6))
    for protocol, size, method, frame, classes in evaluations:
        numbers, cm = empirical_metrics(frame.true.tolist(), frame.predicted.tolist(), classes)
        context = {"protocol": protocol, "n_cpg": size, "method": method}
        metric_rows.append({**context, **numbers})
        for i, true in enumerate(classes):
            tp = int(cm[i, i]); support = int(cm[i].sum()); predicted_count = int(cm[:, i].sum())
            class_rows.append({
                **context, "cell_type": true, "true_count": support,
                "predicted_count": predicted_count, "correct": tp,
                "recall": tp / support, "precision": tp / predicted_count if predicted_count else None,
            })
            for j, predicted in enumerate(classes):
                confusion_rows.append({**context, "true": true, "predicted": predicted, "count": int(cm[i, j])})
        if protocol == "nested_discovery":
            pd.DataFrame(cm, index=classes, columns=classes).rename_axis("true").to_csv(
                directory / "nested_confusion_matrix.csv")
    metrics = pd.DataFrame(metric_rows)
    confusion = pd.DataFrame(confusion_rows)
    write_csv(metrics, directory / "classification_metrics.csv")
    write_csv(confusion, directory / "classification_confusion_counts.csv")
    write_csv(pd.DataFrame(class_rows), directory / "classification_metrics_by_class.csv")
    write_csv(metrics.loc[metrics.protocol.eq("fixed_size_discovery")].drop(columns="protocol"),
              directory / "discovery_fixed_size_curve.csv")
    write_csv(metrics.loc[metrics.protocol.eq("external")].drop(columns="protocol"),
              directory / "external_size_curve.csv")
    write_csv(confusion.loc[confusion.protocol.eq("external")].drop(columns="protocol"),
              directory / "external_confusion_matrices_long.csv")
    return metric_rows


def train_all(cache, protocol, directory):
    dbeta, ebeta = cache["discovery_beta"], cache["external_beta"]
    ds, es = cache["discovery_samples"], cache["external_samples"]
    classes7, classes6 = protocol["discovery"]["classes"], protocol["external"]["classes"]
    sizes = protocol["nested"]["candidate_sizes"]
    if sizes != protocol["fixed_size"]["candidate_sizes"]:
        raise ValueError("Fixed and nested candidate lists must match this specified protocol.")
    donors = ds.donor_id.to_numpy(dtype=int)
    if sorted(np.unique(donors).tolist()) != protocol["discovery"]["donor_ids"]:
        raise ValueError("The cache donor IDs disagree with the fixed protocol.")
    if dbeta.shape != (protocol["discovery"]["expected_eligible_probes"], protocol["discovery"]["n_samples"]):
        raise ValueError("Unexpected discovery beta universe.")
    if ebeta.shape != (protocol["external"]["expected_common_probes"], protocol["external"]["n_samples"]):
        raise ValueError("Unexpected external beta universe.")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "models").mkdir()
    all_inner, all_fixed, all_nested, selections = [], [], [], []
    distances_rows, models, membership = [], [], []
    for outer_fold, outer_donor in enumerate(protocol["discovery"]["donor_ids"], start=1):
        outer_test = np.flatnonzero(donors == outer_donor)
        outer_train = np.flatnonzero(donors != outer_donor)
        if len(outer_test) != 7 or len(outer_train) != 35:
            raise ValueError("An outer donor split is not 35 training plus 7 test samples.")
        if ds.iloc[outer_test].population.value_counts().to_dict() != {name: 1 for name in classes7}:
            raise ValueError("Outer test donor does not have one sample per class.")
        inner_counts = {size: 0 for size in sizes}
        fold_inner = []
        for inner_donor in protocol["discovery"]["donor_ids"]:
            if inner_donor == outer_donor:
                continue
            validation = np.flatnonzero(donors == inner_donor)
            training = np.flatnonzero((donors != outer_donor) & (donors != inner_donor))
            check_partition(training, validation, outer_test, 28)
            if ds.iloc[validation].population.value_counts().to_dict() != {name: 1 for name in classes7}:
                raise ValueError("Inner validation donor does not have one sample per class.")
            context = {
                "phase": "inner_validation", "outer_fold_index": outer_fold,
                "outer_test_donor_id": outer_donor, "inner_validation_donor_id": inner_donor,
            }
            model_id = f"outer_{outer_fold:02d}_inner_donor_{inner_donor:02d}"
            roles = np.where(donors == outer_donor, "outer_test",
                             np.where(donors == inner_donor, "inner_validation", "inner_training"))
            add_membership(membership, model_id, ds, roles)
            model = fit_model(
                dbeta, ds, training, classes7, cache["discovery_ids"], sizes,
                model_id, directory, models, context)
            predictions = evaluate_model(
                dbeta, ds, validation, model, sizes, ["euclidean"], context, distances_rows)
            fold_inner.extend(predictions)
            for row in predictions:
                inner_counts[row["n_cpg"]] += int(row["correct"])
        # Selection is completed using only inner predictions before outer evaluation.
        selected_size = choose_size(inner_counts)
        for size in sizes:
            size_rows = [row for row in fold_inner if row["n_cpg"] == size]
            if len(size_rows) != 35:
                raise ValueError("Each size must have 35 pooled inner validation predictions.")
            selections.append({
                "outer_fold": outer_fold, "outer_test_donor_id": outer_donor,
                "candidate_cpgs": size, "correct_inner_predictions": inner_counts[size],
                "n_inner_predictions": 35, "pooled_inner_balanced_accuracy": inner_counts[size] / 35,
                "selected": size == selected_size,
            })
        all_inner.extend(fold_inner)
        context = {"phase": "outer_test", "outer_fold_index": outer_fold,
                   "outer_test_donor_id": outer_donor, "inner_validation_donor_id": ""}
        model_id = f"outer_{outer_fold:02d}_training"
        add_membership(membership, model_id, ds,
                       np.where(donors == outer_donor, "outer_test", "outer_training"))
        model = fit_model(
            dbeta, ds, outer_train, classes7, cache["discovery_ids"], sizes,
            model_id, directory, models, context)
        fixed_predictions = evaluate_model(
            dbeta, ds, outer_test, model, sizes, ["euclidean"], context, distances_rows)
        all_fixed.extend(fixed_predictions)
        selected_predictions = [dict(row, selected_signature_size=selected_size)
                                for row in fixed_predictions if row["n_cpg"] == selected_size]
        all_nested.extend(selected_predictions)
        print(json.dumps({"stage": "outer_fold_complete", "outer_fold": outer_fold,
                          "outer_donor_id": outer_donor, "selected_size": selected_size,
                          "outer_correct": sum(row["correct"] for row in selected_predictions),
                          "outer_samples": len(selected_predictions)}), flush=True)
    nested = pd.DataFrame(all_nested)
    nested.insert(0, "outer_prediction_index", np.arange(1, len(nested) + 1))
    nested = nested.drop(columns=["n_cpg"])
    fixed = pd.DataFrame(all_fixed).sort_values(["n_cpg", "outer_fold_index"], kind="stable")
    fixed.insert(0, "prediction_index", fixed.groupby("n_cpg", sort=False).cumcount() + 1)
    write_csv(nested, directory / "nested_outer_predictions.csv")
    write_csv(fixed, directory / "discovery_fixed_size_predictions.csv.gz")
    write_csv(pd.DataFrame(all_inner), directory / "nested_inner_predictions.csv.gz")
    write_csv(pd.DataFrame(selections), directory / "nested_inner_size_selection.csv")

    # External labels and external beta values do not enter this training call.
    common_beta = np.asarray(dbeta[cache["common_indices"], :], dtype=np.float64)
    external_training = np.flatnonzero(ds.population.isin(classes6).to_numpy())
    if len(external_training) != 36 or set(ds.iloc[external_training].population) != set(classes6):
        raise ValueError("External-transfer training requires the 36 six-class discovery samples.")
    external_sizes = protocol["external"]["candidate_sizes"]
    context = {"phase": "external_evaluation", "outer_fold_index": "",
               "outer_test_donor_id": "", "inner_validation_donor_id": ""}
    model_id = "external_discovery_six_classes"
    add_membership(membership, model_id, ds, np.where(
        ds.population.isin(classes6).to_numpy(), "external_model_training", "excluded_discovery_cell_type"))
    add_membership(membership, model_id, es, ["external_evaluation"] * len(es), cohort="external")
    model = fit_model(
        common_beta, ds, external_training, classes6, cache["common_ids"], external_sizes,
        model_id, directory, models, context, export_full_ranking=True)
    external_rows = evaluate_model(
        ebeta, es, np.arange(len(es)), model, external_sizes,
        ["euclidean", "correlation"], context, distances_rows)
    external = pd.DataFrame(external_rows)
    write_csv(external, directory / "external_individual_predictions.csv.gz")
    write_csv(pd.DataFrame(distances_rows), directory / "classifier_distances.csv.gz")
    write_csv(pd.DataFrame(membership), directory / "model_sample_membership.csv")
    write_json(directory / "fitted_models.json", models)
    metrics = export_metrics(directory, nested, fixed, external, classes7, classes6)
    return {
        "models_fitted": len(models), "outer_folds": len(protocol["discovery"]["donor_ids"]),
        "nested_predictions": len(nested), "fixed_size_predictions": len(fixed),
        "inner_validation_predictions": len(all_inner), "external_predictions": len(external),
        "selected_sizes_in_outer_donor_order": [
            next(row["candidate_cpgs"] for row in selections if row["outer_fold"] == fold and row["selected"])
            for fold in range(1, 7)],
        "metrics": metrics, "unresolved_distance_ties": 0, "undefined_correlations": 0,
    }


def verify_output_payloads(directory):
    for path in directory.rglob("*"):
        if not path.is_file() or any(part.startswith(".") for part in path.relative_to(directory).parts):
            continue
        if path.suffix == ".npz":
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise ValueError(f"Invalid fitted-model archive: {path.name}")
            with np.load(path, allow_pickle=False) as archive:
                for name in archive.files:
                    value = archive[name]
                    if value.dtype.kind in "fc" and not np.isfinite(value).all():
                        raise ValueError(f"Nonfinite fitted model values: {path.name}/{name}")
        elif path.suffix == ".csv" or path.name.endswith(".csv.gz"):
            pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--protocol", type=Path,
                        default=Path(__file__).resolve().parents[1] / "config/classification_protocol.json")
    parser.add_argument("--discovery-signals", type=Path)
    parser.add_argument("--external-signals", type=Path)
    parser.add_argument("--discovery-metadata", type=Path)
    parser.add_argument("--external-metadata", type=Path)
    args = parser.parse_args()
    raw_args = [args.discovery_signals, args.external_signals, args.discovery_metadata, args.external_metadata]
    protocol = json.loads(args.protocol.read_text())
    protocol_hash = sha256(args.protocol)
    if args.cache_dir.exists():
        if any(value is not None for value in raw_args):
            parser.error("Use the verified existing cache alone, or choose a new cache path for raw inputs.")
    else:
        if not all(value is not None for value in raw_args):
            parser.error("A new cache requires both signal files and both metadata files.")
        prepare_cache_atomic(*raw_args, args.cache_dir, protocol, args.protocol)
    cache = load_cache(args.cache_dir, args.protocol)
    working = work_directory_name(args.out)
    script_names = ["train_classification.py", "classification_core.py", "array_io.py", "atomic_io.py"]
    scripts_dir = Path(__file__).resolve().parent
    code_hashes = {name: sha256(scripts_dir / name) for name in script_names}
    try:
        summary = train_all(cache, protocol, working)
        if sha256(args.protocol) != protocol_hash:
            raise RuntimeError("The protocol changed during model fitting.")
        if any(sha256(scripts_dir / name) != digest for name, digest in code_hashes.items()):
            raise RuntimeError("The model-fitting source changed during execution.")
        verify_output_payloads(working)
        sync_tree(working)
        publish_directory(working, args.out.resolve())
        print(json.dumps({"stage": "training_complete", "out": str(args.out), **{
            key: value for key, value in summary.items() if key != "metrics"
        }}, indent=2), flush=True)
    except Exception as exc:
        if working.exists():
            write_json(working / "training_failure.json", {
                "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                "details": exc.details if isinstance(exc, UnresolvedPrediction) else None,
                "protocol_sha256": protocol_hash, "source_code_sha256": code_hashes,
            })
        raise


if __name__ == "__main__":
    main()
