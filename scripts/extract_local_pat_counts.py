#!/usr/bin/env python3
"""Extract one sample's actual PAT counts; Python standard library, no network.

Example:
  python3 extract_local_pat_counts.py --gsm GSM5652281 --population CD4 \
    --donor-id 49 --pat /path/sample.hg38.pat.gz --out /path/results

Input: local gzip/BGZF PAT, ordered by chr1 starting CpG index. Coordinates
are hg38 CpG indices, not base pairs. A complete interval requires a real chr1
record starting beyond 120000. The whole input gzip is validated to EOF.
Outputs: candidate site/window CSVs, a valid gzip of original prefix lines
and a per-sample count summary. No common-sample filter or population estimates are applied.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

START, END, WIDTH = 100001, 120000, 4
N_SITES = END - START + 1
N_WINDOWS = N_SITES // WIDTH
READ_BYTES = 1024 * 1024
PATTERNS = [format(i, "04b").replace("0", "T").replace("1", "C") for i in range(16)]


def validate_gzip(path, hash_uncompressed=False):
    n = 0
    digest = hashlib.sha256() if hash_uncompressed else None
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(READ_BYTES), b""):
            n += len(chunk)
            if digest is not None:
                digest.update(chunk)
    return n, digest.hexdigest() if digest is not None else None


def extract_region(source, prefix):
    source, prefix = Path(source).resolve(), Path(prefix).resolve()
    if source == prefix or (prefix.exists() and source.samefile(prefix)):
        raise ValueError("Input PAT and output prefix refer to the same file.")
    methylated, total = [0] * N_SITES, [0] * N_SITES
    windows = [[0] * 16 for _ in range(N_WINDOWS)]
    prefix_digest = hashlib.sha256()
    metrics = Counter()
    first_start, previous, boundary = None, -1, None
    prefix.parent.mkdir(parents=True, exist_ok=True)
    staging = None
    try:
        with gzip.open(source, "rb") as reader, tempfile.NamedTemporaryFile(
            mode="wb", dir=prefix.parent, prefix=prefix.name + ".", suffix=".tmp", delete=False
        ) as output:
            staging = Path(output.name)
            with gzip.GzipFile(filename="", mode="wb", compresslevel=6, fileobj=output, mtime=0) as writer:
                for line in reader:
                    writer.write(line)
                    prefix_digest.update(line)
                    metrics["prefix_uncompressed_bytes"] += len(line)
                    metrics["original_lines_retained"] += 1
                    if not line.strip() or line.startswith(b"#"):
                        continue
                    fields = line.rstrip(b"\r\n").split(b"\t")
                    if len(fields) < 4:
                        raise ValueError("Malformed PAT row: fewer than four tab-separated fields.")
                    if fields[0] not in (b"chr1", b"1"):
                        continue
                    start = int(fields[1])
                    metrics["chr1_records_examined"] += 1
                    if first_start is None:
                        first_start = start
                        if first_start > START:
                            raise ValueError("Input begins after the regional start; missing early sites cannot be coded as zero.")
                    if start < 1 or start < previous:
                        raise ValueError("Invalid or decreasing chr1 CpG start.")
                    previous = start
                    if start > END:
                        boundary = start
                        break
                    pattern, multiplicity = fields[2].upper(), int(fields[3])
                    if not pattern or multiplicity <= 0 or any(c not in b"CT." for c in pattern):
                        raise ValueError("Invalid PAT pattern or multiplicity.")
                    fragment_end = start + len(pattern) - 1
                    if fragment_end < START:
                        continue
                    metrics["overlapping_pat_records"] += 1
                    metrics["overlapping_pat_multiplicity_sum"] += multiplicity
                    if start < START:
                        metrics["left_overlapping_records"] += 1
                        metrics["left_overlapping_multiplicity_sum"] += multiplicity
                    for cpg in range(max(START, start), min(END, fragment_end) + 1):
                        call = pattern[cpg - start]
                        if call in (67, 84):
                            total[cpg - START] += multiplicity
                            if call == 67:
                                methylated[cpg - START] += multiplicity
                    first_window = max(START, START + ((start - START + WIDTH - 1) // WIDTH) * WIDTH)
                    last_window = min(END - WIDTH + 1, fragment_end - WIDTH + 1)
                    for block in range(first_window, last_window + 1, WIDTH):
                        fragment = pattern[block - start:block - start + WIDTH]
                        if len(fragment) == WIDTH and all(call in (67, 84) for call in fragment):
                            code = 0
                            for call in fragment:
                                code = (code << 1) | int(call == 67)
                            windows[(block - START) // WIDTH][code] += multiplicity
        if first_start is None or boundary is None:
            raise ValueError("Incomplete interval: no real chr1 record with start >120000 was found.")
        prefix_n, prefix_sha = validate_gzip(staging, hash_uncompressed=True)
        if prefix_n != metrics["prefix_uncompressed_bytes"] or prefix_sha != prefix_digest.hexdigest():
            raise ValueError("The generated prefix does not preserve the original PAT lines.")
        os.replace(staging, prefix)
        staging = None
    finally:
        if staging is not None:
            staging.unlink(missing_ok=True)
    log = dict(metrics)
    log.update(first_chr1_start=first_start, first_chr1_start_after_interval=boundary,
               chr1_starts_sorted_through_boundary=True, complete_interval=True,
               prefix_uncompressed_sha256=prefix_sha, prefix_valid_gzip_to_eof=True)
    return methylated, total, windows, log


def validate_counts(methylated, total, windows):
    if len(methylated) != N_SITES or len(total) != N_SITES or len(windows) != N_WINDOWS:
        raise ValueError("Unexpected candidate dimensions.")
    if any(not isinstance(m, int) or not isinstance(n, int) or not 0 <= m <= n
           for m, n in zip(methylated, total)):
        raise ValueError("Invalid methylated/total site counts.")
    for j, counts in enumerate(windows):
        if len(counts) != 16 or any(not isinstance(c, int) or c < 0 for c in counts):
            raise ValueError("Invalid pattern counts.")
        depth = sum(counts)
        for pos in range(WIDTH):
            index = j * WIDTH + pos
            mc = sum(c for k, c in enumerate(counts) if (k >> (WIDTH - pos - 1)) & 1)
            if depth > total[index] or mc > methylated[index] or depth - mc > total[index] - methylated[index]:
                raise ValueError("Complete-pattern counts are inconsistent with per-site counts.")
    return {"candidate_dimensions_correct": True, "counts_are_nonnegative_integers": True,
            "site_methylated_never_exceeds_total": True,
            "complete_pattern_marginals_are_subsets_of_site_counts": True}


def write_counts(out, gsm, population, donor, methylated, total, windows):
    counts_dir = out / "counts"
    counts_dir.mkdir(parents=True, exist_ok=True)
    site_file = counts_dir / f"{gsm}_candidate_site_counts.csv"
    window_file = counts_dir / f"{gsm}_candidate_window_counts.csv"
    with site_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gsm", "population", "donor_id_from_metadata", "cpg_index",
                         "methylated_count", "total_count", "sample_depth_ge_10"])
        for j, (m, n) in enumerate(zip(methylated, total)):
            writer.writerow([gsm, population, donor, START + j, m, n, n >= 10])
    with window_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gsm", "population", "donor_id_from_metadata", "block_start_index",
                         "block_end_index", "complete_depth", "sample_depth_ge_20"] + ["count_" + p for p in PATTERNS])
        for j, counts in enumerate(windows):
            block, depth = START + WIDTH * j, sum(counts)
            writer.writerow([gsm, population, donor, block, block + WIDTH - 1, depth, depth >= 20] + counts)
    return site_file, window_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsm", required=True)
    parser.add_argument("--population", required=True)
    parser.add_argument("--donor-id", required=True, type=int)
    parser.add_argument("--pat", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if not args.gsm.startswith("GSM") or not args.gsm[3:].isdigit():
        parser.error("--gsm must be a GSM accession.")
    source, out = args.pat.resolve(), args.out.resolve()
    if not source.is_file():
        parser.error("Local PAT gzip does not exist.")
    prefix = out / "prefixes" / f"{args.gsm}.pat.gz"
    if source == prefix:
        parser.error("The input must not be the output prefix path.")
    initial_stat = source.stat()
    compressed_bytes = source.stat().st_size
    uncompressed_bytes, _ = validate_gzip(source)
    print(json.dumps({"event": "full_source_gzip_validated", "gsm": args.gsm,
                      "compressed_bytes": compressed_bytes, "uncompressed_bytes": uncompressed_bytes}), flush=True)
    methylated, total, windows, _ = extract_region(source, prefix)
    validate_counts(methylated, total, windows)
    final_stat = source.stat()
    if (initial_stat.st_size, initial_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
        raise ValueError("Source file changed during processing.")
    site_file, window_file = write_counts(out, args.gsm, args.population, args.donor_id, methylated, total, windows)
    depths = [sum(c) for c in windows]
    summary = {"source_sample": args.gsm, "population": args.population, "donor_id": args.donor_id,
               "region_cpg_indices": [START, END], "candidate_sites": N_SITES, "candidate_windows": N_WINDOWS,
               "sites_with_observed_calls": sum(n > 0 for n in total),
               "sites_depth_ge_10_this_sample_only": sum(n >= 10 for n in total),
               "windows_with_observed_complete_patterns": sum(n > 0 for n in depths),
               "windows_depth_ge_20_this_sample_only": sum(n >= 20 for n in depths),
               "methylated_calls": sum(methylated), "total_calls": sum(total),
               "complete_pattern_observations_across_windows": sum(depths),
               "common_nine_sample_filter_applied": False, "population_entropy_reconstructed": False}
    summary_file = out / "counts" / f"{args.gsm}_extraction_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "completed", **summary, "prefix_bytes": prefix.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
