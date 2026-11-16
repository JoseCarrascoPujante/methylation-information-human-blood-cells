# DNA methylation profiles and blood cell recognition

This repository contains numerical data, sample metadata and computational code
for the study "Quantifying DNA methylation as epigenetic information in bits
across human blood-cell methylomes". It supports Figures 1–4, S1–S6 Figs and
the supplementary numerical tables. Paths below are relative to this directory. Original public
array intensity matrices and PAT files can be obtained from the accessions and
URLs recorded in data/metadata and regional/metadata.

## Contents and computational scope

data/arrays
  Discovery summaries describe 481,737 eligible cg probes in 42 samples from
  seven cell populations, with six donors per population. The common-probe
  table has 448,510 identified cg probes and two descriptive R columns based on
  the six shared cell types. The finite pattern correlations are Pearson
  correlations between the two six-component vectors of cell-type mean beta
  values. A constant external profile has an undefined correlation and is
  represented by an empty CSV field; 448,509 common probes have finite values.
  The pairwise-distance table contains all 861 unordered discovery-sample
  pairs, with sample identities, group labels and distances on 481,737 probes.

data/spectral
  Identified discovery-ranked probes, positive eigenvalues and summaries for
  nine signature sizes are provided. The spectra concern CpG values centered
  and standardized across 36 discovery samples from six shared cell types.
  The code recomputes participation ratio and spectral effective rank from
  these eigenvalues. The marginal-profile sums are numerical inputs; the
  plotting code does not reconstruct a beta matrix from those sums.
  scripts/recompute_array_aggregation_spectra.py calculates the aggregation
  summaries, rankings, signature memberships, marginal sums and spectra
  directly from the individual-sample beta cache.

regional/samples
  Each of nine samples has one candidate-site count CSV and one candidate-
  window count CSV. These are the complete numerical inputs for the regional
  calculation: 20,000 sites and 5,000 non-overlapping four-CpG windows per
  sample, on chr1, hg38 CpG indices 100,001–120,000 inclusive. Coordinates are
  CpG indices, not physical base-pair positions. C denotes methylated and T
  denotes unmethylated calls. Counts include the PAT record multiplicity.

regional/results
  Common-coverage masks, retained indices, complete four-CpG pattern counts,
  window entropies and donor/population summaries are supplied. The masks
  retain 18,531 sites with at
  least 10 C/T calls in every sample and 58 windows with at least 20 complete
  C/T fragments in every sample. The 16 window-pattern counts are ordered from
  TTTT to CCCC according to binary order with T=0 and C=1.
  The command for regional calculations additionally writes the full donor-
  by-site fractions/entropies and site-by-population aggregation tables from
  the included candidate counts, so these large derived tables need not be
  duplicated in the archive.

regional/metadata
  Sample accession, sample name, population, published donor identifier,
  public PAT URL and coverage summaries are provided. CD4 T cells, B cells and
  monocytes are represented for each of the same three donors, 54, 51 and 49.
  The published sample-name, donor and cell-type information is identified by
  its source Table S1 and spreadsheet row.

plot_data and FIGURE_DATA_MAP.csv
  The figure map links every panel to the CSV file(s) and plotting command
  used for it. The binary-entropy curve and evaluation-design parameters also
  have CSV representations. Large observation tables are stored once and may
  support several panels. DATA_DICTIONARY.csv lists columns and measurement
  units for the numerical files.

data/model_fits
  The complete intensity-based model fit outputs include identified inner,
  nested, fixed-size and external predictions; score grids; the full common
  discovery ranking; fitted centroids and probe memberships; and all 37 model
  archives. Sample identifiers are GSM accessions, with donor identifiers for
  the paired discovery design. README_classification.txt describes every
  output and the commands that regenerate it. The plotting scripts recompute
  confusion matrices, accuracy, balanced accuracy and empirical mutual
  information from the individual held-out predictions in this folder; they
  do not fit classifiers. Intensity-based model fitting is implemented by
  scripts/prepare_array_cache.py and scripts/train_classification.py, with
  the protocol in config/classification_protocol.json. The complete commands
  are in docs/RAW_ARRAY_REPRODUCTION.md.

Additional array metadata
  data/arrays/donor_entropy_sample_identities.csv identifies the 42 donor
  summaries. data/spectral/signature_membership.csv.gz identifies every
  probe in every spectral signature; signature_population_marginal_sums.csv
  supplies all six population sums for each size. The numerical rank settings
  and diagnostics are explicit. data/metadata/array_quantity_definitions.json
  records the equations, units, input scopes and software versions.


data/donor_comparisons and data/illustrative_examples
  The donor comparison uses 42 individual mean h(beta) values in six matched
  donor blocks. It supplies ranks, the Friedman statistic and its Monte Carlo
  null frequency table, donor-stratified distances and all six within-donor
  summaries. Exact probability examples distinguish aggregation, conditional
  class information, linked-pattern entropy and spectral dimensions. The
  R7 histogram has 50 fixed bins and is supplied with its source counts.
  docs/DONOR_COMPARISONS_AND_EXAMPLES.txt records definitions and commands.

data/model_fits/ranked_cpg_identifiers_by_fit.csv.gz
  All 190,000 exported ranked identifiers from 37 training fits are listed
  with model, partition, rank and R value. evaluated_rank_cutoffs_by_fit.csv
  identifies all 297 evaluated prefixes and their minimum retained scores.
  Each top-k signature consists of ranks 1 through k within its own fit.
  Training samples and selected nested sizes are recorded separately in
  the model-design and inner-selection outputs. The identification utility is
  scripts/export_signature_identifiers.py and takes a fitted-model directory.

## Definitions and units

The binary-entropy function is h(p)=-p log2(p)-(1-p) log2(1-p), with h(0)=h(1)=0.
For arrays, h is applied operationally to beta=M/(M+U+100). For population c,
sum_A_bits is the sum over eligible probes of h of the equally weighted
donor-mean beta. sum_B_bits averages the donor-specific h(beta) values before
summing over probes. median_A_across_CpGs is the median of the first, per-probe
quantity; it is not the mean of donor entropies. These profile transformations
do not by themselves determine a joint molecular methylation distribution.

R is the sample variance among cell-type mean beta values divided by that
variance plus the mean within-cell-type sample variance among donors. R and
spectral dimensions are dimensionless. MAD is the mean absolute difference
between two beta profiles. RMS is the square root of their mean squared
difference. These distances have beta-fraction units. Pairwise observations
share samples; their interquartile ranges are descriptive, not confidence
intervals. The overlap reference m^2/N is an expectation for independently
and uniformly selected sets of size m, used as a descriptive reference.

Regional single-site fractions are p=methylated_count/total_count. At each
retained site, donor fractions have equal weights. The two population
summaries are A=mean_site h(mean_donor p) and
B=mean_site mean_donor h(p), in bits per CpG. Their difference is a Jensen
difference. Counts are not pooled across donors for these summaries.

Regional joint entropy and the sum of four marginal binary entropies are
calculated from the same 16 complete-fragment pattern counts, in bits per
four-CpG window. The depth-20 values are conditional expectations of plug-in
entropies after sampling 20 complete fragments without replacement from the
observed count vector. They are evaluated analytically with finite
hypergeometric probabilities; there is no simulation, seed or repetition
count. Window quantities are averaged equally within each donor and then
equally over the three donors. The individual donor points are descriptive;
the package does not calculate confidence intervals or genomic P values.

## Running the code

Tested with Python 3.10 and numpy 1.23.0, pandas 1.3.5, matplotlib 3.10.0 and Pillow 8.4.0 (requirements.txt).
Install those packages, then run from this directory:

```bash
python scripts/rebuild_all.py --out rebuilt
```

This command also reproduces the donor-block comparison, histogram and
exact probability examples. It regenerates regional statistics from all 18 count CSVs,
array/spectral summaries from the supplied numerical exports, classification
metrics from the individual predictions, and all ten figures. It writes only
to the new output directory. It performs no network access. Existing input
files are not modified. COMMANDS.txt provides separate commands for each
calculation and figure, and commands for optional extraction from public raw
inputs downloaded independently by the reader.

## Reference analysis results

| Evaluation | Samples correctly assigned | Classes | Empirical label information in bits |
| --- | ---: | ---: | ---: |
| Nested discovery | 42/42 | 7 | 2.807354922 |
| External 100-CpG Euclidean rule | 33/36 | 6 | 2.355388542 |
| External 100-CpG correlation rule exploratory | 36/36 | 6 | 2.584962501 |

The selected sizes in the six nested outer partitions are 100, 250, 100,
100, 100 and 5,000 CpGs. The primary and exploratory external rules are
reported separately. The complete intensity-based calculations and the
regional count-based calculations reproduce the numerical results at the
precision presented in the manuscript and figures. Full precision outputs
are provided by the corresponding commands.

## Figure assignment

  Fig1: information framework and binary-entropy curve.
  Fig2: discovery profile summaries, distances and recognition.
  Fig3: cross-cohort agreement and external transfer.
  Fig4: distinct profile, spectral and label-information quantities.
  S1 Fig: signature-size evaluations and nested selection.
  S2 Fig: nested discovery and primary external confusion matrices.
  S3 Fig: regional sequencing summaries from nine samples.
  S4 Fig: evaluation design.
  S5 Fig: donor aggregation and spectral dimensionality.
  S6 Fig: full distribution of the seven-class discovery variance ratio.

Scientific input
metadata, masks, units and sample sets are retained in the numerical files and
their descriptions. The figures/ directory includes the ten reference PNG figures. The plotting
commands regenerate those scientific panels from the supplied numerical inputs.
The intensity-based commands also fit the classifiers and calculate the array
and spectral quantities from individual-sample measurements.

## Public sources

GSE35069: discovery array data, NCBI Gene Expression Omnibus.
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE35069

GSE110554: external array data, NCBI Gene Expression Omnibus.
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE110554

Loyfer N, Magenheim J, Peretz A, et al. A DNA methylation atlas of normal human
cell types. Nature. 2023;613:355–364. doi:10.1038/s41586-022-05580-6.
The regional sample accessions and individual public PAT URLs are listed in
regional/metadata/sample_metadata.csv. Published donor and cell-type labels
are from Table S1, 41586_2022_5580_MOESM4_ESM.xlsx; spreadsheet row identifiers
are provided in the sample metadata.

GSE186458: source sequencing dataset for the regional analysis, NCBI GEO.
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186458

## Version citation and reuse

Citation metadata are in CITATION.cff. Cite the authors and persistent
identifier of the public repository record, together with the original studies
and datasets relevant to the analysis. Code is released under the MIT License
and data, figures and documentation under CC BY 4.0 (LICENSE); original public
sources and third-party software retain their own terms.
