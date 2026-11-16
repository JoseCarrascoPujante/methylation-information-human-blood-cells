"""Read verified array signals and create an explicit, checksummed beta cache.

The chunked float64 parsing and source-column alignment follow the verified
recompute_common_array_profiles.py and recompute_discovery_distances.py readers.
This module retains sample beta values; it does not train or evaluate models.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


DISCOVERY_TYPES = {
    "CD4+": "CD4", "CD8+": "CD8", "CD14+": "Mono", "CD19+": "B",
    "CD56+": "NK", "Neu": "Neu", "Eos": "Eos",
}
SOURCE_URLS = {
    "discovery": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE35nnn/GSE35069/suppl/GSE35069_Matrix_signal_intensities.txt.gz",
    "external": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE110nnn/GSE110554/suppl/GSE110554_signals.txt.gz",
}
SAMPLE_COLUMNS = [
    "gsm", "sample_name", "population", "donor_id",
    "source_sample_index_zero", "array_id",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_record(path: Path) -> dict:
    return {"sha256": sha256(path), "bytes": path.stat().st_size}


def header(path: Path) -> list[str]:
    with gzip.open(path, "rt") as stream:
        return stream.readline().rstrip("\r\n").split("\t")


def discovery_samples(metadata_path: Path, signal_header: list[str], protocol: dict) -> pd.DataFrame:
    records = json.loads(metadata_path.read_text())
    if len(records) != 60 or len(signal_header) != 181 or signal_header[0] != "ID_REF":
        raise ValueError("Unexpected discovery metadata or intensity-matrix structure.")
    samples = []
    for index in range(18, 60):
        entry = records[index]
        title = entry["Sample_title"][0]
        prefix, suffix = title.rsplit("_", 1)
        if prefix not in DISCOVERY_TYPES or not suffix.isdecimal():
            raise ValueError(f"Unrecognized discovery sample title: {title}")
        samples.append({
            "gsm": entry["gsm"], "sample_name": title,
            "population": DISCOVERY_TYPES[prefix], "donor_id": int(suffix),
            "source_sample_index_zero": index, "array_id": f"SAMPLE {index + 1}",
        })
    frame = pd.DataFrame(samples, columns=SAMPLE_COLUMNS)
    expected = protocol["discovery"]
    counts = frame.groupby(["donor_id", "population"]).size()
    if len(frame) != expected["n_samples"] or len(counts) != expected["n_samples"] or not counts.eq(1).all():
        raise ValueError("Discovery samples are not one observation per donor and class.")
    if sorted(frame.donor_id.unique().tolist()) != expected["donor_ids"]:
        raise ValueError("Discovery donor suffixes do not match the fixed protocol.")
    if set(frame.population) != set(expected["classes"]):
        raise ValueError("Discovery populations do not match the fixed protocol.")
    if frame.gsm.duplicated().any() or frame.sample_name.duplicated().any():
        raise ValueError("Repeated discovery sample identity.")
    return frame


def external_samples(metadata_path: Path, signal_header: list[str], protocol: dict) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    required = {"gsm", "sample_title", "array_id", "population", "included"}
    if not required.issubset(metadata.columns) or len(metadata) != 49 or len(signal_header) != 148:
        raise ValueError("Unexpected external sample map or matrix structure.")
    included = metadata.included.str.lower()
    if not included.isin(["true", "false"]).all():
        raise ValueError("The external included field must be true or false.")
    chosen = metadata[included.eq("true")].copy()
    expected_inclusion = metadata.population.ne("MIX") & metadata.sample_title.ne("Th2535-2")
    if not np.array_equal(expected_inclusion.to_numpy(), included.eq("true").to_numpy()):
        raise ValueError("External exclusions disagree with the protocol.")
    if metadata.population.eq("MIX").sum() != 12 or metadata.sample_title.eq("Th2535-2").sum() != 1:
        raise ValueError("Expected twelve mixtures and the specified technical replicate.")
    raw_order = [name.removesuffix(" Unmethylated signal") for name in signal_header[1::3]]
    if len(raw_order) != 49 or len(set(raw_order)) != 49 or set(raw_order) != set(metadata.array_id):
        raise ValueError("External array identifiers differ between metadata and raw columns.")
    samples = []
    for row in chosen.to_dict("records"):
        samples.append({
            "gsm": row["gsm"], "sample_name": row["sample_title"],
            "population": row["population"], "donor_id": "",
            "source_sample_index_zero": raw_order.index(row["array_id"]),
            "array_id": row["array_id"],
        })
    frame = pd.DataFrame(samples, columns=SAMPLE_COLUMNS).sort_values(
        "source_sample_index_zero", kind="stable").reset_index(drop=True)
    expected = protocol["external"]
    if len(frame) != expected["n_samples"] or frame.population.value_counts().to_dict() != {
        name: 6 for name in expected["classes"]
    }:
        raise ValueError("External evaluation must contain six samples per shared class.")
    if frame.gsm.duplicated().any() or frame.sample_name.duplicated().any():
        raise ValueError("Repeated external sample identity.")
    return frame


def read_eligible_beta(path: Path, samples: pd.DataFrame, cohort: dict, beta_offset: float):
    names = header(path)
    columns = [names[0]]
    raw_indices = samples.source_sample_index_zero.to_numpy(dtype=int)
    if not np.all(np.diff(raw_indices) > 0):
        raise ValueError("Sample metadata must follow source-column order.")
    for sample in samples.to_dict("records"):
        index = sample["source_sample_index_zero"]
        triplet = names[1 + index * 3:1 + (index + 1) * 3]
        expected = [sample["array_id"] + suffix for suffix in (
            " unmethylated signal", " methylated signal", " detection pval")]
        if [name.lower() for name in triplet] != [name.lower() for name in expected]:
            raise ValueError(f"Signal columns disagree for {sample['gsm']}.")
        columns.extend(triplet)
    ids_parts, beta_parts = [], []
    rows_seen = 0
    for chunk in pd.read_csv(
        path, sep="\t", compression="gzip", usecols=columns,
        dtype={names[0]: str}, chunksize=20000
    ):
        ids = chunk.iloc[:, 0].astype(str).to_numpy()
        values = chunk.iloc[:, 1:].to_numpy(dtype=np.float64).reshape(-1, len(samples), 3)
        keep = np.fromiter(
            (value.startswith(cohort["probe_prefix"]) for value in ids), bool, count=len(ids))
        keep &= (values[:, :, 2] < cohort["detection_p_strictly_less_than"]).sum(axis=1) >= cohort["minimum_detected_samples"]
        selected = values[keep]
        beta = selected[:, :, 1] / (selected[:, :, 1] + selected[:, :, 0] + beta_offset)
        if not np.isfinite(beta).all() or np.any((beta < 0.0) | (beta > 1.0)):
            raise ValueError("Invalid beta value after the specified eligibility filter.")
        ids_parts.append(ids[keep])
        beta_parts.append(beta)
        rows_seen += len(chunk)
        if rows_seen % 200000 == 0:
            print(json.dumps({"stage": "read_arrays", "cohort": cohort["accession"],
                              "source_rows_read": rows_seen}), flush=True)
    # Iteration through all chunks reaches gzip EOF, including its integrity check.
    ids = np.concatenate(ids_parts).astype(str)
    beta = np.concatenate(beta_parts, axis=0)
    if len(ids) != cohort["expected_eligible_probes"]:
        raise ValueError(f"Unexpected eligible probe count for {cohort['accession']}: {len(ids)}")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("Duplicate eligible probe identifiers.")
    return ids, beta, {
        "source_rows": rows_seen, "eligible_probes": len(ids),
        "samples": len(samples), "minimum_detected_samples": cohort["minimum_detected_samples"],
        "detection_p_strictly_less_than": cohort["detection_p_strictly_less_than"],
        "gzip_read_to_eof_passed": True,
    }


def prepare_cache(
    discovery_path: Path, external_path: Path, discovery_metadata: Path,
    external_metadata: Path, cache_dir: Path, protocol: dict, protocol_path: Path
) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=False)
    paths = {"discovery": discovery_path, "external": external_path,
             "discovery_metadata": discovery_metadata, "external_metadata": external_metadata}
    source_records = {key: {"filename": path.name, **file_record(path)} for key, path in paths.items()}
    source_stats = {key: (path.stat().st_size, path.stat().st_mtime_ns) for key, path in paths.items()}
    ds = discovery_samples(discovery_metadata, header(discovery_path), protocol)
    es = external_samples(external_metadata, header(external_path), protocol)
    dids, dbeta, dsummary = read_eligible_beta(discovery_path, ds, protocol["discovery"], protocol["beta_offset"])
    np.save(cache_dir / "discovery_beta.npy", dbeta, allow_pickle=False)
    del dbeta
    eids, ebeta, esummary = read_eligible_beta(external_path, es, protocol["external"], protocol["beta_offset"])
    positions = pd.Index(eids).get_indexer(dids)
    di = np.flatnonzero(positions >= 0).astype(np.int64)
    ei = positions[di]
    common_ids = dids[di]
    if len(common_ids) != protocol["external"]["expected_common_probes"]:
        raise ValueError(f"Unexpected shared probe count: {len(common_ids)}")
    np.save(cache_dir / "external_common_beta.npy", ebeta[ei], allow_pickle=False)
    del ebeta
    np.save(cache_dir / "common_discovery_indices.npy", di, allow_pickle=False)
    compression = {"method": "gzip", "mtime": 0}
    for name, ids in [
        ("discovery_probe_ids.csv.gz", dids),
        ("common_probe_ids.csv.gz", common_ids),
        ("external_eligible_probe_ids.csv.gz", eids),
    ]:
        pd.DataFrame({"probe_id": ids}).to_csv(cache_dir / name, index=False, compression=compression)
    ds.to_csv(cache_dir / "discovery_samples.csv", index=False)
    es.to_csv(cache_dir / "external_samples.csv", index=False)
    for key, path in paths.items():
        if (path.stat().st_size, path.stat().st_mtime_ns) != source_stats[key]:
            raise RuntimeError(f"An input changed during cache preparation: {key}")
    manifest = {
        "schema_version": 1, "status": "complete", "array_orientation": "probes_by_samples",
        "precision": "float64", "source_files": source_records, "source_urls": SOURCE_URLS,
        "protocol_sha256": sha256(protocol_path), "protocol": protocol,
        "discovery": dsummary, "external": esummary, "common_probes": len(common_ids),
        "row_order": "eligible discovery source-row order; common rows preserve discovery order",
        "column_order": "included sample source-column order, recorded in sample CSVs",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "reader_sha256": sha256(Path(__file__)),
        "files": {path.name: file_record(path) for path in sorted(cache_dir.iterdir()) if path.is_file()},
    }
    write_json(cache_dir / "cache_manifest.json", manifest)
    return manifest


def load_cache(cache_dir: Path, protocol_path: Path | None = None) -> dict:
    manifest_path = cache_dir / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema_version"), manifest.get("status"),
        manifest.get("array_orientation"), manifest.get("precision")) != (
        1, "complete", "probes_by_samples", "float64"
    ):
        raise ValueError("Unsupported or incomplete beta cache.")
    if protocol_path is not None and manifest["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Cache protocol differs from the requested frozen protocol.")
    for name, record in manifest["files"].items():
        path = cache_dir / name
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Cache checksum mismatch: {name}")
    dbeta = np.load(cache_dir / "discovery_beta.npy", mmap_mode="r", allow_pickle=False)
    ebeta = np.load(cache_dir / "external_common_beta.npy", mmap_mode="r", allow_pickle=False)
    di = np.load(cache_dir / "common_discovery_indices.npy", allow_pickle=False)
    dids = pd.read_csv(cache_dir / "discovery_probe_ids.csv.gz").probe_id.to_numpy(dtype=str)
    cids = pd.read_csv(cache_dir / "common_probe_ids.csv.gz").probe_id.to_numpy(dtype=str)
    ds = pd.read_csv(cache_dir / "discovery_samples.csv", keep_default_na=False)
    es = pd.read_csv(cache_dir / "external_samples.csv", keep_default_na=False)
    if dbeta.dtype != np.float64 or ebeta.dtype != np.float64:
        raise ValueError("The cache matrices must be float64.")
    if dbeta.shape != (len(dids), len(ds)) or ebeta.shape != (len(cids), len(es)):
        raise ValueError("Cache dimensions do not agree with identifiers and samples.")
    if di.dtype != np.int64 or not np.array_equal(dids[di], cids) or len(np.unique(di)) != len(di):
        raise ValueError("The common-probe index does not align with discovery rows.")
    return {"manifest": manifest, "discovery_beta": dbeta, "external_beta": ebeta,
            "discovery_ids": dids, "common_ids": cids, "common_indices": di,
            "discovery_samples": ds, "external_samples": es}
