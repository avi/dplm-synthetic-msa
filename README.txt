DPLM synthetic MSA benchmark — release candidate v0.1.0
=====================================================

Open index.html for the ten-protein comparison and links to every prediction.

RESULT
50 unconstrained DPLM variants at 40% masking improve mean reference TM-score
versus no MSA in 9 of 10 tested proteins. This compares the same query and the
same OpenFold-3 model across inputs. It does not establish variant function,
evolutionary homology, or generalization to unseen proteins. Higher masking
helps some cases but strongly harms the shallow-family antifreeze case.

CONTENTS
scripts/       Generation, OpenFold wrapper, input preparation, analysis, checks.
examples/      Myoglobin FASTA and complete 40% / 60% generation configurations.
data/          Ten query FASTAs, reference CIFs, MSAs and generation audit files.
results/       305 predictions and confidence sidecars; sample and summary CSVs.
provenance/    Original experiment protocols and untouched wrapper snapshots.
third_party/   Upstream license notices.
tests/         CPU-only configuration and input-contract tests.
manifest.json  Exact protein, reference, condition and file mappings.
SHA256SUMS.json Integrity hashes for the release payload.

275 predictions belong to the main unconstrained comparison. Another 15 test
250 variants, and 15 retain earlier constrained trials as exploratory results.
Exploratory trials are excluded from the headline. Cyclophilin's exploratory
trial used R55 protection, identity selection and temperature3 vanilla sampling.
Main runs have no permanently protected residues or identity filtering.
Failed/partial generation attempts and the excluded CspB depth-screen candidate
are not included as benchmark results. Every completed folding condition is
included once. Some pools share sequences: the 250-variant pools extend the
corresponding 50-variant pools; these are not independent generation replicates.

SETUP (from this folder)
Use Python 3.13 for the exact local dependency versions below.

  python3.13 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt
  python -m unittest discover -s tests
  python scripts/verify_release.py

Inspection, tests and metric recomputation use no GPU or Modal account.
Generation and folding need your own Modal account; set it up with:

  python -m modal setup

Cloud jobs create model-cache/result volumes in that account and incur GPU
charges. Model weights download into the cloud cache on first use; weights are
not distributed here. The bundled DPLM wrapper uses A100-80GB by default and the
OpenFold wrapper uses A100; GPU environment overrides are available in source.
Cloud Python, torch and model dependencies are defined in each wrapper.

EXAMPLE 1 — GENERATE 50 VARIANTS
Inspect/edit examples/generate_50_mask40.json. Paths inside the config are
relative to this release folder. This command only prints the submission:

  python scripts/run_generation.py --config examples/generate_50_mask40.json

Run the generation with:

  python scripts/run_generation.py --config examples/generate_50_mask40.json --execute

For 60% masking use examples/generate_50_mask60.json. Choose a fresh out_dir for
each rerun. Outputs include variants.fasta, synthetic.a3m, metadata.json and a
candidate audit. No fixed residue positions, temperature increase, identity
filter or adaptive masking is used by either sample configuration.

Direct CLI equivalent (also exposes optional constraint/filter controls):

  python -m modal run --detach scripts/modal_dplm_homologs.py --fasta examples/query.fasta --out-dir out/myoglobin_direct --num-variants 50 --no-filter-identity --no-adaptive-mask --mask-fraction 0.4 --sampling-strategy gumbel_argmax --temperature 1 --max-iter 100 --seed 42 --model-revision 7a7e651baa667d094aba05e9dc1cf52a3332110a

All available options:
  python -m modal run scripts/modal_dplm_homologs.py --help

IMPORTANT: the underlying legacy CLI defaults enable identity filtering and
adaptive masking. Use the sample configs or explicitly pass both --no-* flags
above to reproduce this benchmark. Gumbel sampling uses upstream's fixed noise
scale; its temperature argument does not increase diversity. The vanilla
sampler has different temperature behavior. We make substitution-only pools,
so the generated A3M is already aligned. Unique variants are not necessarily
independent evolutionary observations or functional proteins.

ITERATIVE MUTATION PILOT
  python scripts/run_generation.py --config examples/generate_50_iterative.json --execute

This makes 50 terminal descendants from independent query-started lineages.
Five rounds each keep 90% of the mutable residues of the immediate parent,
resampling a fresh random 10% (rounded, minimum one position). Mutations may
accumulate, stay unchanged, change again, or revert. This is iterative model
sampling, without selection for fitness. The final mutation rate is measured,
not imposed. The example uses the same pinned model, seed and diffusion steps
as the one-shot runs, with identity filtering and adaptive masking disabled.

keep_fractions is a comma-separated schedule: "0.9,0.9,0.9,0.9,0.9" for constant
retention, or e.g. "0.8,0.85,0.9,0.95" for increasing retention. Fractions apply
to mutable positions; fixed positions always retain their query residues.
An empty schedule preserves one-shot behavior. max_iter remains the number of
DPLM denoising steps PER ROUND. Iterative mode requires adaptive_mask=false;
if identity filtering is enabled it applies only to terminal descendants.
mask_fraction is ignored in iterative mode. max_candidates counts lineages,
not intermediate descendants. Final pools exclude the query and duplicates.

An invalid intermediate sequence terminates and rejects that lineage while
other lineages continue. Its failed output and partial history are retained;
earlier valid ancestors are never substituted into the final pool.

candidates.jsonl records every round's sequence, masked positions, changes
from its parent, query identity and reversions. rounds/round_NN.a3m provides
the same final-selected lineages at earlier depths; those earlier pools can
contain duplicate/query rows. synthetic.a3m contains query + terminal variants.
Compare final pools at matched depth against existing single-pass pools, then
fold the original query with each pool and score against the same reference.
A fixed number of rounds does not guarantee matched final identity, and this
pilot uses more model sampling than a one-shot run. Summarize a completed pilot:

  python scripts/analyze_iterative.py --protein myoglobin --generation out/myoglobin_iterative_keep90_r5 --out out/myoglobin_iterative_keep90_r5/analysis.json

Add --predictions PATH after folding to include reference structure scores.
Original release checksums
remain a record of the original payload and will flag these source edits.

EXAMPLE 2 — FOLD THE QUERY WITH MATCHED INPUTS
This works immediately with the included example results; no regeneration is
needed. It creates no-MSA, query-only, synthetic, and natural-MSA conditions:

  python scripts/prepare_comparison.py --fasta examples/query.fasta --synthetic-msa data/myoglobin/generation/synthetic_40/synthetic.a3m --natural-msa data/myoglobin/msas/natural_50/custom_database_hits.a3m --out-dir out/example_inputs --name myoglobin

To use newly generated variants, replace --synthetic-msa with
out/myoglobin_mask40/synthetic.a3m. Omit --natural-msa if unavailable.

Submit predictions:

  python -m modal run --detach scripts/modal_openfold3.py --query-json out/example_inputs/queries.json --msa-mode disabled --num-diffusion-samples 5 --num-model-seeds 1 --out-dir out/example_predictions --run-name example

Here 'msa-mode disabled' disables server retrieval; explicit MSA files remain
active according to each query's use_msas/use_main_msas flags. Templates and
ligands are omitted. With --num-model-seeds 1 the runner produces model seed
2746317213 from its seed42 RNG, as in the released results.

The optional wrapper mode --msa-mode precomputed calls the bundled
precompute_openfold3_msa.py and additionally needs Boltz installed in a suitable
Python environment. Set LOCAL_BOLTZ_PYTHON to that interpreter. It is not needed
for any example above; all benchmark MSAs are provided.

EXAMPLE 3 — PREPARE THE FULL PACKAGED BENCHMARK
  python scripts/prepare_benchmark.py --out out/benchmark_queries.json

Defaults to 55 main conditions / 275 structures. Add --include-depth250 for
three depth conditions; --include-exploratory adds three historical conditions.
Use --protein myoglobin (repeatable) to select a smaller run. Pass the resulting
JSON to the same Modal folding command. Input preparation alone is CPU-only.

ANALYSIS AND REPRODUCIBILITY
  python scripts/verify_release.py
  python scripts/make_report.py

The verifier checks hashes, all 305 structures' metrics, query sequences and
pool contracts. Report regeneration rewrites derived files; their bytes may
differ across plotting-library versions. Checksums describe the original
release, so verify before regenerating. manifest.json and per-pool metadata
record model revisions, sequences, settings and reference details. Historical protocols retain their original relative filenames; manifest.json
is the canonical mapping for this release. Old
wrapper snapshots are retained because earlier constrained experiments used
different controllers. The public DPLM wrapper also contains a later validation
fix for ignored identity targets; its inference path is unchanged.

METHOD AND LIMITS
Same original query folded in every condition; variants only supply MSA rows.
DPLM-650M checkpoint revision 7a7e651baa667d094aba05e9dc1cf52a3332110a,
DPLM source 8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d.
OpenFold-3 0.4.0, of3-p2-155k, source
af111e167d3035fb10a048d2c1608243b660b443. Five diffusion samples per condition,
one model seed and one generated pool; samples are correlated, not independent
replicate experiments. Convenience panel; training overlap is possible.
No random-mutation MSA control or experimental validation was performed.

No-MSA supplies a gap row and zero profile in this OpenFold version. Query-only
preserves query-derived MSA features. Query-only controls were added for the
seven newer proteins, not the original three. Main input arms have 51 rows for
50 variants (including query). Natural depth is distinct aligned nonquery hits;
the embedding cap is 1024 rows, while profile calculation precedes subsampling.
Top7 is designed; its four search hits are not verified natural homologs.
DHFR reference 1RX2 has an N37D difference from the query. CI2 and Top7 use
resolved contiguous segments; Top7 MSE maps to methionine. Acylphosphatase uses
the first deposited NMR model. Other construct/ligand details are in manifest.
Whole-chain Cα RMSD, reference-normalized TM-align score, Cα lDDT, pLDDT and pTM
are reported separately. Confidence scores are not experimental accuracy.

ATTRIBUTION AND LICENSING
Original project code and documentation: MIT (see LICENSE).
See THIRD_PARTY_NOTICES.txt and LICENSE_STATUS.txt for upstream terms. Model
weights are not bundled.
This folder is prepared locally; nothing has been published or uploaded.
