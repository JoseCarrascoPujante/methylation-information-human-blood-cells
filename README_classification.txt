RECONSTRUCTING ARRAY CLASSIFICATION
==================================

This module builds donor-held-out nested predictions, fixed-size discovery
predictions and external predictions directly from array signal intensities.
The computations use NumPy and pandas. The algorithm is deterministic and
uses no random split, random seed or stochastic model fit.

Install the versions listed in requirements.txt. The detailed
definitions are in docs/CLASSIFICATION_METHODS.txt; the machine-readable
parameters are in config/classification_protocol.json.

INPUTS

Download the two public intensity files identified by config/array_sources.json,
or provide local copies with the stated SHA256 digests. The sample metadata
are included under data/metadata. Keep intensity files in a directory such
as raw/. The commands below make no network requests.

CREATE A CACHE AND FIT THE MODELS

From the repository root, run:

  python scripts/train_classification.py --discovery-signals raw/GSE35069_Matrix_signal_intensities.txt.gz --external-signals raw/GSE110554_signals.txt.gz --discovery-metadata data/metadata/discovery_series_samples.json --external-metadata data/metadata/external_array_sample_map.csv --cache-dir cache/arrays --out results/model_fits

The cache and result directories must be new. The command reads the two
intensity files, validates sample-column alignment, applies the stated global
eligibility filters, saves a checksummed float64 cache, and fits the models.
All outputs are written beneath the requested new directories.

ALTERNATIVE: PREPARE THE CACHE SEPARATELY

  python scripts/prepare_array_cache.py --discovery-signals raw/GSE35069_Matrix_signal_intensities.txt.gz --external-signals raw/GSE110554_signals.txt.gz --discovery-metadata data/metadata/discovery_series_samples.json --external-metadata data/metadata/external_array_sample_map.csv --cache-dir cache/arrays

Then fit from that verified cache:

  python scripts/train_classification.py --cache-dir cache/arrays --out results/model_fits

Cache input hashes and matrix/metadata alignment are checked before fitting.
The cached data contain measured beta values and sample identities. The
model-fitting command does not take prediction tables as inputs.

OUTPUTS

  nested_outer_predictions.csv
      The 42 outer test predictions and the size selected for each donor.
  nested_inner_predictions.csv.gz
      Individual predictions for all inner folds and candidate sizes.
  nested_inner_size_selection.csv
      The 48 outer-fold-by-size scores and selected candidates.
  discovery_fixed_size_predictions.csv.gz
      The 336 predictions from eight fixed-size evaluations.
  external_individual_predictions.csv.gz
      The 648 predictions from nine sizes and two distance rules.
  external_discovery_ranking.csv.gz
      The discovery ranking of all 448,510 common CpGs.
  external_100_signature_and_centroids.csv
      The primary 100-CpG signature and its six discovery class centroids.
  model_sample_membership.csv and fitted_models.json
      Training, validation and test membership, identified by GSM.
  models/*.npz
      Ranked features, ranking scores, training GSMs and class centroids.
  classifier_distances.csv.gz
      Distances to every class for each evaluated prediction.
  classification_metrics.csv and classification_confusion_counts.csv
      Evaluation metrics and confusion counts by protocol, size and rule.
  classification_metrics_by_class.csv
      True counts, predicted counts, correct counts, recall and precision.
  discovery_fixed_size_curve.csv and external_size_curve.csv
      Size-specific aggregate performance.

For the nested procedure, signature size is selected using the inner
validation predictions before any outer test prediction is evaluated.
External signature sizes are evaluated descriptively; the external outcomes
do not select the primary size or fit discovery centroids.
