# Reproduce the array analyses from public intensities

Run commands from the root of the extracted repository.
The commands create new output directories and keep the downloaded inputs
read only.

## Inputs

Download the two compressed signal matrices identified in
`data/metadata/array_analysis_parameters.json` into a directory named `raw`.
That metadata file provides their exact public URLs and SHA256 checksums.
The two raw downloads together contain approximately 472 MB of compressed
data. The generated beta cache occupies approximately 300 MB plus metadata.

- `GSE35069_Matrix_signal_intensities.txt.gz`
- `GSE110554_signals.txt.gz`

Sample selection is defined by the included GEO metadata and the explicit
external array sample map. The settings in `config/classification_protocol.json`
state the eligible sample sets, detection thresholds, candidate signature
sizes, ranking rule and distance rules.

## Prepare the individual sample beta cache

```bash
python scripts/prepare_array_cache.py \
  --discovery-signals raw/GSE35069_Matrix_signal_intensities.txt.gz \
  --external-signals raw/GSE110554_signals.txt.gz \
  --discovery-metadata data/metadata/discovery_series_samples.json \
  --external-metadata data/metadata/external_array_sample_map.csv \
  --cache-dir beta_cache
```

The cache contains float64 beta values, probe IDs and sample identities.
It retains 481,737 eligible discovery probes across 42 samples and aligns
448,510 common probes to the 36 external samples. Its cache_manifest.json records file
hashes and matrix orientation. The model-fitting and spectral commands check
the required cache payloads before using them.

## Fit and evaluate the classifiers

```bash
python scripts/train_classification.py \
  --cache-dir beta_cache \
  --out fitted_classification
```

The training command evaluates nested donor-held-out discovery, the stated
fixed discovery signature sizes and the external signature sizes. Every
inner split recomputes feature ranking and class centroids from its training
donors. The outer test donor is excluded from all inner operations. External
training uses the 36 discovery samples from the six shared populations.
The 100-CpG Euclidean evaluation is primary; the correlation-distance
comparison is exploratory. External labels are used to evaluate predictions.

Outputs include the sample identities and individual predictions, confusion
matrices, accuracy, balanced accuracy, empirical label mutual information,
inner score grids, selected sizes, probe memberships and fitted centroids.
The training command uses measured beta values and the stated protocol; it
does not use the supplied reference predictions to select or fit models.

## Calculate aggregation and spectral quantities

```bash
python scripts/recompute_array_aggregation_spectra.py \
  --cache-dir beta_cache \
  --out array_quantities
```

The script calculates donor and population transformations directly from
the eligible discovery beta values. It computes R7 for the descriptive
seven-population analysis and R6 for selecting the shared-population
signatures. It calculates the correlation spectra, participation ratios,
effective ranks and class-balanced marginal-profile sums for the nine
signature sizes. Spectral dimensions are dimensionless; marginal-profile
sums are in bits under the operational definitions in the manuscript.

For the complete per-probe aggregation export, add
`--write-probe-aggregation`. This optional table is larger than the summary
outputs.

## Distances and cross-cohort profiles

The additional raw-input commands in `COMMANDS.txt` reproduce all 861
discovery sample-pair distances, the common-probe class-mean profiles,
variance-ratio comparisons, pattern correlations and ranked-set overlap.
They use the same public source matrices and the included sample metadata.
Their output files identify the sample and probe sets for each quantity.

## Regional sequencing

The regional calculations use the eighteen included candidate-count CSVs:
one site-count and one complete-pattern-count file for each of nine samples.

```bash
python scripts/recompute_regional_statistics.py \
  --samples-root regional/samples \
  --out regional_quantities
```

The masks are recalculated from all nine samples. The script retains 18,531
sites and 58 four-CpG windows under the stated coverage thresholds and
calculates the site and window quantities with equal donor weights.
`COMMANDS.txt` also gives the extraction command for independently obtained
public PAT files. Sample identities and public PAT URLs are listed in
`regional/metadata/sample_metadata.csv`.
