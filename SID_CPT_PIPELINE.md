# Semantic + Collaborative SID and CPT Pipeline

This extension keeps MiniOneRec's existing `SFT -> GRPO` stages and adds:

1. multimodal product normalization;
2. collaborative embedding alignment;
3. RFSQ SID construction and tokenizer diagnostics;
4. SID continual pre-training with prefix-aware negative ranking.

The stages write normal `.npy`, `.index.json`, CSV and Hugging Face checkpoint
artifacts, so the existing `convert_dataset.py`, `sft.py`, `rl.py` and
`evaluate.py` remain the downstream entry points.

## 1. Product image-text joint embeddings

### Current SID dataset: chronological ~5,000-product prefix

The current SID input dataset is **Office_Products_chrono5000**, not the
historical popularity-selected top3000 benchmark. Starting at
`2018-10-01T00:00:00+08:00`, count distinct `parent_asin` values in raw reviews
ordered by their original **millisecond** timestamps. The time at which the
5,000th distinct product first appears is the cutoff. Retain **all** reviews
between start and cutoff inclusive, including all cutoff-time ties, duplicate
reviews, image-less products, missing metadata, and singleton users. Do not
select by popularity, apply user/item 5-core, or add interactions after cutoff.

```bash
python data/build_amazon23_chronological_subset.py \
  --reviews_file data/Amazon23/raw/Office_Products.jsonl.gz \
  --source_catalog data/Amazon23/Office_Products/Office_Products.item.jsonl \
  --output_dir data/Amazon23/Office_Products_chrono5000 \
  --start_time 2018-10-01T00:00:00 --timezone Asia/Shanghai --items 5000
python data/download_amazon23_product_images.py \
  --catalog data/Amazon23/Office_Products_chrono5000/Office_Products.item.jsonl \
  --output_catalog data/Amazon23/Office_Products_chrono5000/Office_Products.local.item.jsonl \
  --image_dir data/Amazon23/Office_Products_chrono5000/images \
  --allow_missing_images --allow_unavailable_images --workers 32
```

The output directory must be new. Item IDs follow first appearance (ties by
ASIN). Stable interaction ordering is timestamp then original source line;
same-time records are all retained. Global 80/10/10 time splits assign each
boundary timestamp entirely to the earlier split. Full user histories are in
`.inter.json` and `.user_histories.jsonl`; `.samples.jsonl` and `.train/valid/test.inter`
retain **every** interaction, including first interactions with an empty
history. `--max_history 0` preserves full prior history. No later interactions
are introduced into sample histories. Keep singleton/empty-history counts
visible; this dataset may have many cold-start users.

This dataset intentionally uses **mixed embedding inputs**:

- readable product image: image + structured attributes, `embedding_mode=joint`,
  `image_used=true`;
- no usable image: attributes from metadata using the same Instruct model, then
  **the same Qwen3-VL-Embedding-8B** encoder in text-only mode,
  `embedding_mode=text`, `image_used=false`;
- no usable product text: record an exception and retain its catalog ID and
  interactions. Never replace it with a later product or fabricate a vector.

Missing image URLs and unavailable/download-failed images are separate recorded
categories. `--allow_unavailable_images` explicitly accepts those failures as
text-only inputs and preserves their error details. Without that flag, download
failures fail preparation. An unreadable image declared available still causes
attribute/embedding failure, never a silent text fallback. Text usability is
assessed from product title/description/features/details, not category/brand
alone. Metadata/image availability does not affect the cutoff.

After starting an Ollama service with the Instruct Q4_K_M model, run the
two-stage resumable pipeline (replace paths with real local paths):

```bash
python data/run_amazon23_chronological_embeddings.py \
  --catalog data/Amazon23/Office_Products_chrono5000/Office_Products.local.item.jsonl \
  --output_dir output/amazon23_chrono5000 \
  --ollama_model benchmark-qwen3-vl:8b-instruct-q4_K_M \
  --ollama_url http://127.0.0.1:11435 --num_ctx 8192 --workers 1 \
  --model models/Qwen3-VL-Embedding-8B/Qwen3-VL-Embedding-8B-Q4_K_M.gguf \
  --mmproj models/Qwen3-VL-Embedding-8B/mmproj-Qwen3-VL-Embedding-8B-f16.gguf \
  --llama_server tools/llama.cpp/build/bin/llama-server \
  --matrix_output data/Amazon23/Office_Products_chrono5000/Office_Products.emb-qwen-attributes-mixed-td.npy
```

Only one stage's model is loaded at a time. Successful v2 attributes resume by
ID; the orchestrator verifies their ASINs before resuming, retries failures,
and preserves raw attempts separately from final validated attributes. It starts
the embedding server only after attributes are valid and Ollama is unloaded.
The per-item `embedding_mode` array in mixed NPZs reports the actual input, not
just the requested dataset-level mode. Export checks coverage, ASINs, image
provenance, shape, finite/nonzero vectors, and saves a `.npy.item_ids.npy`
row-to-original-ID sidecar. RFSQ export honors those IDs; SID input loading
rejects non-finite vectors rather than converting them to zero placeholders.
Exceptions do not remove interactions. If any product has no valid vector,
complete SID coverage is unresolved and must be handled explicitly downstream.

Once the full pipeline reports `status=complete`, the SID stage must use the
new matrix and the new interaction root, not the top3000/full-k-core artifacts.
For example, the later RFSQ run would use:

```bash
python rq/rqfsq.py \
  --data_path data/Amazon23/Office_Products_chrono5000/Office_Products.emb-qwen-attributes-mixed-td.npy \
  --ckpt_dir output/Office_Products_chrono5000-rfsq
python rq/generate_rfsq_indices.py \
  --checkpoint output/Office_Products_chrono5000-rfsq/<run>/best_collision_model.pth \
  --output_file data/Amazon23/Office_Products_chrono5000/Office_Products.index.json \
  --repair_collisions
```

These are subsequent training commands, not part of embedding/sequence
preparation. After SID generation, use
`--data_dir data/Amazon23/Office_Products_chrono5000` in `convert_dataset.py`
and leave `--keep_longest_only` disabled to retain all prepared samples. If
collaborative alignment is later run, use the same new `.train.inter` and
verify complete dense vector coverage first; it must not consume old histories
or a compact matrix with missing item IDs as if it were dense.

This run's cutoff is `2018-10-03T06:26:21.072+08:00`, yielding 5,000 products,
5,576 users and 6,465 interactions (train/valid/test 5,172/647/646). There are
4,978 downloaded usable product images, 21 products without image URLs and one
product whose original 26 image URLs all fail. Thus **4,978 joint + 22 text-only**
inputs are expected; no product lacks usable text metadata. Final actual
counts, exceptions, validation and commands are saved in
`output/amazon23_chrono5000/REPORT.md` after the pipeline completes. Do not label
this representation as an all-joint embedding dataset or claim recommendation
quality from successful vector generation.

### Two-stage multimodal product representation

The following image-complete top3000 workflow is retained for reproducing the
historical strict-joint benchmark. The current chronological dataset uses the
explicit mixed policy above.

Stage 1 uses the existing Qwen3-VL Instruct model to extract target-product
attributes from metadata and product images. The prompt excludes people, hands,
backgrounds, lighting and camera composition. Stage 2 jointly embeds the product
image and the structured product text using Qwen3-VL-Embedding; no new embedding
model is trained. The intended method is image-text joint embedding, not
image-only embedding or text-only embedding. Attribute normalization aims to
focus the textual description on the product; it does not prove that the joint
encoder ignores visual backgrounds or shooting conditions.

The catalog embedding CLI defaults to `mixed`; the processed-item alternative
defaults to `joint`. Explicitly pass `--embedding_mode joint` for this historical
image-complete benchmark, or `--embedding_mode mixed` for the current chronological dataset. Separate
text-only comparison runs must not be reported as all-joint experiments.

The v2 schema includes product type, brand, model, compatible model, function,
material, color, pattern, specifications, structural features and intended use.
Code fixes field ordering, normalizes Unicode/case/whitespace, maps a small set of
unambiguous aliases, and sorts/deduplicates list fields. This reduces nuisance
variation but does not guarantee identical descriptions of unverified products.

The raw Office_Products metadata has 710,503 entries. The project's 2018-10
to 2023-09, 5-core interaction filter yields 38,944 products. Build that
intermediate catalog from the downloaded Amazon23 files:

```bash
python data/build_amazon23_catalog.py \
  --metadata_file data/Amazon23/raw/meta_Office_Products.jsonl.gz \
  --output_file data/Amazon23/Office_Products/Office_Products.item.jsonl \
  --index_file data/Amazon23/Office_Products/Office_Products.item2id
python data/count_amazon23_kcore.py \
  --metadata_file data/Amazon23/raw/meta_Office_Products.jsonl.gz \
  --reviews_file data/Amazon23/raw/Office_Products.jsonl.gz \
  --database data/Amazon23/Office_Products/kcore.sqlite
python data/export_amazon23_kcore_catalog.py \
  --database data/Amazon23/Office_Products/kcore.sqlite \
  --source_catalog data/Amazon23/Office_Products/Office_Products.item.jsonl \
  --output_catalog data/Amazon23/Office_Products/Office_Products.kcore.item.jsonl \
  --index_file data/Amazon23/Office_Products/Office_Products.kcore.item2id
```

For the compact experiment, select the 3,000 most interacted products with
image URLs, reapply 5-core to their interactions, and rebuild user histories
and chronological train/valid/test samples. The current result has 2,995
surviving products; outputs and `subset.stats.json` are under
`data/Amazon23/Office_Products_top3000/`.

```bash
python data/build_amazon23_top_items.py \
  --database data/Amazon23/Office_Products/kcore.sqlite \
  --source_catalog data/Amazon23/Office_Products/Office_Products.kcore.item.jsonl \
  --reviews_file data/Amazon23/raw/Office_Products.jsonl.gz \
  --output_dir data/Amazon23/Office_Products_top3000 \
  --items 3000
```

Download one actual product image per selected product into the project and
create a catalog whose `images` fields contain local file paths:

```bash
python data/download_amazon23_product_images.py \
  --catalog data/Amazon23/Office_Products_top3000/Office_Products.item.jsonl \
  --output_catalog data/Amazon23/Office_Products_top3000/Office_Products.local.item.jsonl \
  --image_dir data/Amazon23/Office_Products_top3000/images
```

Generate image-derived v2 summaries with the local Ollama model:

```bash
python rq/summary/generate_structured_summaries.py \
  --item_file data/Amazon23/Office_Products_top3000/Office_Products.local.item.jsonl \
  --output_file output/Office_Products_top3000.attributes-v2.jsonl \
  --ollama_model qwen3-vl:8b-instruct-q4_K_M
```

Start the local embedding server with
`D:/sofaware/Qwen3-VL-Embedding-8B-Q4_K_M/Start-Embedding.ps1`, then run:

```bash
python rq/text2emb/amazon23_summary_image_embed.py \
  --catalog data/Amazon23/Office_Products_top3000/Office_Products.local.item.jsonl \
  --summary_path output/Office_Products_top3000.attributes-v2.jsonl \
  --output_dir output/amazon23_top3000_attributes_joint \
  --model D:/sofaware/Qwen3-VL-Embedding-8B-Q4_K_M/Qwen3-VL-Embedding-8B-Q4_K_M.gguf \
  --mmproj D:/sofaware/Qwen3-VL-Embedding-8B-Q4_K_M/Qwen3-VL-Embedding-8B.mmproj-f16.gguf \
  --embedding_mode joint
```

Use matching `--start` and `--limit` for bounded runs. Stage 1 requires a
readable image; failures are recorded and retried on restart. Only explicit
`--allow_missing_images` permits metadata-only extraction. Resume accepts only
v2 successful records; use a new summary file for migration from old summaries.
Stage 2 in `--embedding_mode joint` loads both product images and product text.
The GGUF backend requires the matching vision projector and a multimodal server.
`--embedding_mode text` is an optional text-only comparison without a projector.

Use a fresh embedding output directory. Each `.npz` stores `item_id`,
`parent_asin`, `embedding`, `embedding_mode`, `image_used` (stage 2 only),
`summary_image_used` (stage 1) and `canonical_summary_item_id`.
Only text-mode runs share an embedding request for identical summary texts;
joint runs must also account for the product image.

For text-only comparison runs with externally verified same-product variants,
optionally supply
`--verified_product_map verified_products.json`, e.g. `{"17": "3", "3": "3"}`.
Item 17 uses item 3's description and cached vector while keeping its own item ID
and parent ASIN. Both summaries must exist. The map must point directly to the
canonical item (no chains/cycles). Do not create this map from category alone or
assume parent ASIN establishes cross-merchant identity.

Regenerate downstream aligned embeddings, collaborative outputs and SID indices
in fresh locations before a new recommendation experiment. Do not reuse old SID
checkpoints as if the representation were unchanged. Evaluate held-out same-product
pairs and visually similar different-product pairs before claiming robustness.

On this Windows checkout, the default `python` has no inference dependencies.
The Ollama/GGUF path was verified using `D:/sofaware/anaconda/python.exe`;
PyTorch is loaded only for the Hugging Face path. Run focused checks in PowerShell:

```powershell
& D:/sofaware/anaconda/python.exe -m unittest discover -s tests -p test_product_attributes.py -v
```

A real one-product smoke run on 2026-10-05 produced
`output/scene_invariant_smoke/attributes-v2.jsonl` and
`output/scene_invariant_smoke/vectors/items_0000000_0000001.npz`.
The vector has shape `(1, 4096)`, finite values and unit norm. Its flags are
`summary_image_used=true`, `image_used=false`, `embedding_mode=text`.
This historical text-only smoke run verifies that path's local inference
plumbing, not joint image-text inference or cross-merchant retrieval gains.

### Alternative processed-item JSON workflow

The official Amazon Reviews 2023 Office_Products raw metadata and reviews are
stored in `data/Amazon23/raw/`. To reproduce the processed item file:

```bash
python data/amazon23_data_process.py \
  --dataset Office_Products \
  --metadata_file data/Amazon23/raw/meta_Office_Products.jsonl.gz \
  --reviews_file data/Amazon23/raw/Office_Products.jsonl.gz \
  --user_k 5 --st_year 2018 --st_month 10 \
  --ed_year 2023 --ed_month 9 \
  --output_path data/Amazon23
```

The input is then the `{dataset}.item.json` produced by this command.
Generation is JSONL and resumable; successful
item IDs are skipped when the command is restarted.

```bash
python rq/summary/generate_structured_summaries.py \
  --item_file data/Amazon23/Office_Products/Office_Products.item.json \
  --output_file output/Office_Products.attributes-v2.jsonl \
  --ollama_model qwen3-vl:8b-instruct-q4_K_M
```

The summary model receives metadata and up to two readable product images. Failed/invalid model
responses are recorded with an `error` field and retried on restart.
The local Ollama model above is an Instruct model. It cannot replace the
separate Qwen3-VL-Embedding checkpoint required by the next command.

Embed each validated v2 summary jointly with its product image using the official
Qwen3-VL-Embedding implementation. Put its repository root on
`PYTHONPATH`. Explicitly set `--embedding_mode joint` (also its default).
The output remains aligned to numeric item IDs. Use a fresh `--plm_name` to avoid
overwriting previous vectors.

```bash
python rq/text2emb/amazon_summary_vl2emb.py \
  --dataset Office_Products \
  --root data/Amazon23/Office_Products \
  --plm_name qwen-attributes-joint \
  --plm_checkpoint /models/Qwen3-VL-Embedding-8B \
  --summary_path output/Office_Products.attributes-v2.jsonl \
  --embedding_mode joint
```

For the locally downloaded Q4_K_M GGUF, start the installed server with
`D:/sofaware/Qwen3-VL-Embedding-8B-Q4_K_M/Start-Embedding.ps1`.
The server loads the main GGUF and its vision projector on port 8912.
Replace the example's `--plm_checkpoint` with the local GGUF path,
then add these arguments:

```bash
  --backend gguf \
  --plm_checkpoint D:/sofaware/Qwen3-VL-Embedding-8B-Q4_K_M/Qwen3-VL-Embedding-8B-Q4_K_M.gguf \
  --mmproj D:/sofaware/Qwen3-VL-Embedding-8B-Q4_K_M/Qwen3-VL-Embedding-8B.mmproj-f16.gguf \
  --server_url http://127.0.0.1:8912
```

The bundled `data/Amazon/index/Office_Products.item.json` is an older Amazon18
text-only catalog. Use the processed Amazon23 item file above for this pipeline.

## 2. Collaborative residual projection

`.inter` and converted MiniOneRec CSV inputs are both supported. Pair support
is counted once per user, even when the source contains overlapping sliding
windows. The default weight is `support / sqrt(pop_i * pop_j)`.

```bash
python rq/collab/train_projection.py \
  --embedding_path data/Amazon23/Office_Products/Office_Products.emb-qwen-vl-summary-td.npy \
  --train_inter data/Amazon23/Office_Products/Office_Products.train.inter \
  --output_embedding_path output/Office_Products.emb-qwen-summary-collab.npy \
  --checkpoint_path output/Office_Products.collab-mlp.pt \
  --window_size 2 \
  --min_user_support 3 \
  --popularity_alpha 0.5 \
  --residual_weight 0.2
```

The semantic embedding table is frozen. The shared MLP is residual by default,
which limits semantic drift while InfoNCE introduces the collaborative signal.
Each positive pair is trained in both directions, and the two InfoNCE losses
are averaged. Random negatives exclude the anchor and all of its known positive
neighbors in either direction. Unobserved positive relationships may still be
sampled as negatives.

## 3. RFSQ tokenizer

### Learning-rate strategy ablation

```bash
.venv/bin/python data/run_amazon23_rfsq_lr_trial.py --detach
```

This keeps the Office chrono5000 embeddings, sequences, quantizer and network
fixed, and tests `lr=0.0003` with linear decay after 50 warmup epochs, compared
with the original `lr=0.001` constant schedule. It changes the complete LR
strategy, so it does not isolate the two components individually. Runs 5000
epochs and compares raw collisions, reconstruction and non-singleton purity.
Experimental SIDs remain under the output directory; no canonical SID is
replaced. The generic benchmark exposes `--lr`, `--lr_scheduler_type` and
`--no_publish_index`. Representation and MPS benchmark pipelines share an
exclusive local GPU lock; other model phases wait while image downloads may
continue. Progress and final comparison are under
`output/Office_Products_chrono5000-rfsq/lr-linear-0003`.

### Industrial & Scientific official 5-core trial

The current follow-up trial uses full official deduplicated Industrial &
Scientific 5-core data, with the official absolute-timestamp train/valid/test
files. It retains 25,848 products, 50,985 users and 412,947 interactions
(297,922 / 44,128 / 70,897). All source pairs are unique, every user/product
has at least five interactions, and every official history matches the
preceding full user history. Histories include prior evaluation events as in
the official event-by-event protocol, never future events.

Preparation: `data/prepare_amazon23_official_5core.py`; output:
`data/Amazon23/Industrial_and_Scientific_5core`.
Start the complete local trial with:

```bash
.venv/bin/python data/run_amazon23_industrial_trial.py --detach
```

The existing Office experiment remains intact. Progress is recorded in
`output/amazon23_industrial_5core/trial.status.json`, with representation
progress under `representations/`. The new trial keeps the same models,
mixed image/text policy and RFSQ configuration to provide a baseline
comparison. Exceptions remain in histories; incomplete vector coverage stops
SID training. It does not run collaborative alignment or recommender training.

The chrono5000 end-to-end run uses the prepared mixed-modal matrix and the
unchanged chronological sequence samples:

```bash
.venv/bin/python data/run_amazon23_rfsq_benchmark.py --epochs 5000 --device mps
```

This trains with `tanh`, scalar-level bounds, scale conditioning and three
2048-cell stages. It records commands/timings, exports both raw and repaired
SIDs, audits coverage and reconstruction, reports training-only popularity
and category purity, and joins the existing samples to SID sequences. Output
is a fresh `output/Office_Products_chrono5000-rfsq/benchmark-*` directory.
The canonical `Office_Products.index.json` is published only after coverage,
token-range and uniqueness checks; `.index.raw.json` retains raw assignments.
The best-collision checkpoint is selected by raw excess-assignment collision
rate, breaking ties by lower training reconstruction loss. The best-loss
checkpoint has a separate report, so uniqueness is not treated as the sole
representation-quality measure. No collaborative alignment or recommender
training is included.

`rq/models/rfsq.py` follows *Robust Residual Finite Scalar Quantization for
Neural Compression* (arXiv:2508.15860). Every stage uses a fixed implicit
Cartesian-product codebook. `scale` conditioning is the default because its
parameters are shared and require no per-item normalization side information.
`layernorm` is provided for ablation.

```bash
python rq/rqfsq.py \
  --data_path output/Office_Products.emb-qwen-summary-collab.npy \
  --ckpt_dir output/Office_Products-rfsq \
  --conditioning scale \
  --epochs 5000 \
  --batch_size 2048
```

RFSQ has **no learned centroid table**. Each residual stage has a fixed implicit
Cartesian product of scalar levels. Its nominal discrete capacity is
`product(levels_per_stage)`: `4,4,4,2,2,2,2,2` gives 2,048 cells. Usage metrics
measure assignments to those implicit cells, rather than learned vectors.

The quantizer exposes `benchmark(indices, item_ids, ...)`; RFSQ training calls
it at each `--eval_step` and at the final epoch. It scans the complete embedding
dataset in fixed order, preserving original item IDs even when the training
loader shuffles or drops its tail. Raw per-stage/prefix/full-SID reports are
saved to `<checkpoint_run>/sid_benchmark/epoch_XXXXXX.json`, `latest.json` and
`history.jsonl`. Occupancy/exclusivity are also logged. Metrics do not modify
the loss or checkpoint-selection objective.

To include category purity and training-popularity concentration:

```bash
python rq/rqfsq.py \
  --data_path data/Amazon23/Office_Products_chrono5000/Office_Products.emb-qwen-attributes-mixed-td.npy \
  --ckpt_dir output/Office_Products_chrono5000-rfsq \
  --metrics_item_file data/Amazon23/Office_Products_chrono5000/Office_Products.item.json \
  --metrics_interaction_file data/Amazon23/Office_Products_chrono5000/Office_Products.interactions.jsonl \
  --metrics_category_level leaf \
  --metrics_hot_cluster_percent 1 --metrics_hot_cluster_multiplier 5
```

New training defaults to `--scalar_bound clamp`: each rounded scalar level ID
is saturated to `[0, L_i-1]` before radix packing, so every stage remains a
finite Cartesian-product space even with learnable residual scaling. The
straight-through gradient is retained. This is an explicit correction to the
previous unbounded quantizer, not a post-hoc collision repair. Old checkpoints
without this setting retain their original behavior on export; use
`--scalar_bound legacy` only to reproduce the old unbounded implementation.
Diagnostics still report out-of-range IDs in legacy assignments rather than
silently fixing them. Occupancy means implicit-cell usage, not utilization of
a learned centroid table.

The default is three stages of `4,4,4,2,2,2,2,2`: eight scalar dimensions and
2048 codes per stage. Input embeddings are standardized per dimension, and the
encoder output defaults to `tanh` to bound the first-stage input. The previous
unbounded `none` setting collapsed on the chrono5000 training run; it remains
available as an explicit ablation. Use `--no-standardize` to disable input
standardization. Standardization is repeated when
indices are exported from the checkpoint. Generate three-token
SIDs from the best collision checkpoint:

```bash
python rq/generate_rfsq_indices.py \
  --checkpoint output/Office_Products-rfsq/<run>/best_collision_model.pth \
  --output_file data/Amazon23/Office_Products/Office_Products.index.json \
  --repair_collisions
```

`--repair_collisions` keeps the three-level format and reassigns only duplicate
last-layer codes to unused codes under the same two-token prefix. This is a
post-quantization uniqueness constraint: repaired last codes are valid but are
not raw RFSQ assignments. Always report raw and repaired results separately.

Tokenizer diagnostics:

```bash
python rq/sid_metrics.py \
  --index_file data/Amazon23/Office_Products_chrono5000/Office_Products.index.json \
  --codebook_sizes 2048 2048 2048 \
  --item_file data/Amazon23/Office_Products_chrono5000/Office_Products.item.json \
  --interaction_file data/Amazon23/Office_Products_chrono5000/Office_Products.interactions.jsonl \
  --category_level leaf --hot_cluster_percent 1 --hot_cluster_multiplier 5 \
  --sid_variant repaired \
  --raw_index_file data/Amazon23/Office_Products_chrono5000/Office_Products.index.raw.json \
  --output_file output/Office_Products_chrono5000.sid-metrics.json
```

The report computes metrics for **each layer, each prefix depth, and full SIDs**:

| Metric | Definition |
|---|---|
| `singleton_rate` | Single-item codes/clusters divided by used codes/clusters |
| `exclusive_item_rate` | Items assigned an exclusive code/cluster divided by all items |
| `occupancy_rate` | Used codes divided by declared capacity; prefix capacity is the product of preceding layer capacities |
| `collision_rate` | Excess assignments `(items - distinct full SIDs) / items` |
| `collision_item_rate` | All items in colliding clusters divided by items |
| `hot_clusters_above_mean.cluster_rate` | Used clusters whose summed training interactions reach at least `--hot_cluster_multiplier` times the mean used-cluster interaction mass, divided by used clusters |
| `hot_clusters.popularity_share` | Training interaction share of the hottest `--hot_cluster_percent` percent of used clusters; also reports their item share |
| `purity.item_weighted_purity` | Sum of majority-label counts per cluster divided by labeled items |
| `purity.non_singleton_cluster_item_weighted_purity` | The same calculation restricted to original clusters containing multiple items |

Both observed-code normalized entropy and capacity-normalized entropy are
reported, together with effective code count, maximum/mean cluster size,
label coverage and macro purity. Missing labels are excluded from purity and
reported through coverage; missing heat data does not produce a fabricated
popularity statistic. Zero total training heat produces null heat fractions.
Use the **raw interaction JSONL**, not sliding-window histories, to count each
training record once and avoid repeated-history counts or test leakage.

Category purity defaults to the leaf (last) category rather than the broad
root; for multiple taxonomy paths it uses the first path. A dotted field such
as `--category_field summary.product_type` can evaluate attributes JSONL, but
model-generated product types are proxy labels, not independent ground truth.
Singleton clusters have purity 1 by construction, especially after collision
repair; inspect non-singleton purity and compare raw/repaired metrics. Full-SID
capacity grows multiplicatively, so low full-space occupancy alone is not an
indication of poor layer utilization.

`generate_rfsq_indices.py` now automatically saves `.metrics.json` alongside
its index, using the checkpoint's actual codebook sizes. With
`--repair_collisions` it also preserves `.raw.json` and includes raw metrics.
Pass `--item_file` and `--interaction_file` to that command to include purity
and training heat immediately. Without those files, capacity and exclusivity
diagnostics still run, while unavailable label/heat metrics are omitted.
Downstream Recall/NDCG must still decide the final tokenizer choice.

If raw collisions remain, an optional second stage can fine-tune an RFSQ
checkpoint with a margin loss on pairs of different item vectors that currently
share the same complete SID:

```bash
python rq/finetune_rfsq_collisions.py \
  --checkpoint output/Office_Products-rfsq/<run>/best_collision_model.pth \
  --output_dir output/Office_Products-rfsq-collision \
  --collision_weight 0.3 \
  --margin 1.0
```

The original reconstruction loss remains active. The script re-mines colliding
pairs every 20 epochs and saves the checkpoint with the lowest *raw* collision
rate. Identical input vectors are excluded because deterministic quantization
cannot distinguish them. Export indices from the new `best_collision_model.pth`
with `rq/generate_rfsq_indices.py`, then compare raw tokenizer diagnostics and
downstream recommendation metrics against the original checkpoint.

## 4. Convert and run CPT

After replacing the index file, regenerate every converted split so item/SID
alignment stays consistent:

```bash
python convert_dataset.py \
  --dataset_name Office_Products \
  --data_dir data/Amazon23/Office_Products \
  --output_dir data/Amazon-collab-rfsq
```

Run CPT with one negative from each difficulty level:

```bash
torchrun --nproc_per_node 8 cpt.py \
  --base_model /models/Qwen2.5-1.5B \
  --train_file data/Amazon-collab-rfsq/train/Office_Products_5_2016-10-2018-11.csv \
  --eval_file data/Amazon-collab-rfsq/valid/Office_Products_5_2016-10-2018-11.csv \
  --sid_index_path data/Amazon23/Office_Products/Office_Products.index.json \
  --output_dir output/Office_Products-cpt \
  --rank_loss_weight 0.2 \
  --hard_negatives 1 \
  --medium_negatives 1 \
  --easy_negatives 1
```

The objective is:

```text
L = L_causal_NTP + rank_loss_weight * L_prefix_InfoNCE
```

Candidate scores are sums of teacher-forced SID-token log probabilities. Hard
negatives share all but the final SID token; medium negatives share a shorter
prefix; easy negatives differ at the first token. Every negative comes from
the catalog SID set, so it is legal under constrained decoding.

Finally point the unchanged SFT stage at the CPT checkpoint:

```bash
python sft.py \
  --base_model output/Office_Products-cpt/final_checkpoint \
  ...
```

## Evaluation boundary

The implementation does not claim Recall@10 or NDCG@10 improvements before a
controlled experiment. At minimum compare:

- original text embedding + RQ-VAE + SFT;
- structured-summary embedding only;
- summary + collaborative MLP;
- RQ-VAE, RQ-KMeans+ and RFSQ on the same aligned embeddings;
- CPT causal-only versus CPT plus prefix ranking;
- raw versus collision-repaired RFSQ indices.

Use the same data split, base model, decoding constraints and seed for every
downstream comparison. For the ranking objective, also log the fraction of
examples for which a hard negative exists and the positive-vs-hard inversion
rate on validation data.
