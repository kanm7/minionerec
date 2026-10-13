# Semantic + Collaborative SID and CPT Pipeline

This extension keeps MiniOneRec's existing `SFT -> GRPO` stages and adds:

1. multimodal product normalization;
2. collaborative embedding alignment;
3. RFSQ SID construction and tokenizer diagnostics;
4. SID continual pre-training with prefix-aware negative ranking.

The stages write normal `.npy`, `.index.json`, CSV and Hugging Face checkpoint
artifacts, so the existing `convert_dataset.py`, `sft.py`, `rl.py` and
`evaluate.py` remain the downstream entry points.

## 1. Image-derived product attributes and text embeddings

### Two-stage scene-independent product representation

Stage 1 uses the existing Qwen3-VL Instruct model to extract target-product
attributes from metadata and product images. The prompt excludes people, hands,
backgrounds, lighting and camera composition. Stage 2 embeds only canonical text
using the existing Qwen3-VL-Embedding model; no new model is trained.

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
  --output_dir output/amazon23_top3000_attributes_text \
  --model D:/sofaware/Qwen3-VL-Embedding-8B-Q4_K_M/Qwen3-VL-Embedding-8B-Q4_K_M.gguf
```

Use matching `--start` and `--limit` for bounded runs. Stage 1 requires a
readable image; failures are recorded and retried on restart. Only explicit
`--allow_missing_images` permits metadata-only extraction. Resume accepts only
v2 successful records; use a new summary file for migration from old summaries.
Stage 2 never loads images in its default `--embedding_mode text`. It needs no
vision projector. `--embedding_mode joint` is retained for baseline comparisons.

Use a fresh embedding output directory. Each `.npz` stores `item_id`,
`parent_asin`, `embedding`, `embedding_mode`, `image_used` (stage 2 only),
`summary_image_used` (stage 1) and `canonical_summary_item_id`.
Identical summary texts share an embedding request within a run.

For externally verified same-product variants only, optionally supply
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
This verifies local inference plumbing, not cross-merchant retrieval gains.

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

Embed each validated v2 summary as text with the official Qwen3-VL-Embedding
implementation. Put its repository root on `PYTHONPATH`. This entry point also
defaults to text embeddings; add `--embedding_mode joint` only for comparisons.
The output remains aligned to numeric item IDs. Use a fresh `--plm_name` to avoid
overwriting previous vectors.

```bash
python rq/text2emb/amazon_summary_vl2emb.py \
  --dataset Office_Products \
  --root data/Amazon23/Office_Products \
  --plm_name qwen-attributes-text \
  --plm_checkpoint /models/Qwen3-VL-Embedding-8B \
  --summary_path output/Office_Products.attributes-v2.jsonl
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

The default is three stages of `4,4,4,2,2,2,2,2`: eight scalar dimensions and
2048 codes per stage. Input embeddings are standardized per dimension, and the
encoder output has no `tanh` activation. Use `--no-standardize` or
`--latent_activation tanh` for ablations. Standardization is repeated when
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
  --index_file data/Amazon23/Office_Products/Office_Products.index.json \
  --codebook_sizes 2048 2048 2048 \
  --item_file data/Amazon23/Office_Products/Office_Products.item.json \
  --output_file output/Office_Products.sid-metrics.json
```

The report includes full-SID uniqueness/collision, per-layer occupancy and
entropy, high-frequency code share, prefix cluster sizes and optional category
purity. Downstream Recall/NDCG must still decide the final tokenizer choice.

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
