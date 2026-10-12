# MemAct source release

This is a standalone source snapshot without Git history or development artifacts. Original project files and datasets are not modified. Model weights and upstream datasets are not included. This copy is **not yet a complete reproduction of the paper**: see the implementation coverage below before using it for an experiment.

## Directory layout

- `generation/`: metadata and QA generation, rendered Action graphs, and dataset assembly.
- `reconstruction/`: rebuild real videos and exact frame references from upstream media.
- `training/`: prepare SFT/replay manifests, index RL tasks, and run SFT or RL training.
- `evaluation/`: prepare evaluation episodes, run main QA/Timing evaluation, and check Action coverage.
- `pipeline/`: shared implementation modules imported by the entry points above; these are runtime dependencies, not old branch copies.
- `tests/`: maintained correctness tests and bounded GPU smoke checks; these do not generate benchmark results.
- `tools/`: annotation export, unpacking, and integrity utilities.
- `configs/`, `data/`, `docs/`: configuration, upstream acquisition instructions, and environment information.

There are no Git objects, refs, branches, caches, training outputs, or development logs in this snapshot. Validation reports are distributed with the dataset release separately from executable source.

## Contents

- `training/train_sft.py`: assistant-only, whole-sequence LoRA training with single-GPU or `torchrun` execution, gradient accumulation, AdamW, and gradient clipping.
- `generation/generate_extra.py`: additional spatial and historical QA candidate generation.
- `generation/run.py`: metadata preparation, Current/Memory QA generation, and Action export.
- `evaluation/evaluate.py`: main QA evaluation, separate from generation and training.
- `pipeline/plan26_spatial_qa_memory_action/memory_runtime.py`: deterministic memory storage and retrieval.
- `pipeline/plan26_spatial_qa_memory_action/multiround_rl_v1/`: cumulative-return graph evaluation and reward functions.
- `tools/export_release.py`: checksum-checked, portable annotation export from existing manifests. Private media-location lists must not be published.
- `tests/smoke_native.py`: one real-image LoRA update; no weights are saved.
- `configs/paper.json`: the paper's declared numerical settings, separate from implementation validation.
- `training/prepare_training.py`: converts portable annotation shards into native SFT and replay manifests.
- `tests/`: focused tests for memory/QA causality, media loading, and cumulative-return rewards.

Historical experiments, report generators, broad debug suites, caches, checkpoints, result estimates, and machine-specific launch controllers are excluded.

## Environment

Use Python 3.11 or a compatible Python version. Install `requirements-core.txt` for data and CPU checks. Install PyTorch, Transformers with `Qwen3_5ForConditionalGeneration`, and PEFT for model runs. The tested versions and Transformers commit are recorded in `docs/environment.json` and `requirements-training.txt`. The native checks used PyTorch 2.5.1 with CUDA 12.1; select the appropriate PyTorch CUDA build for your system.

Blender's `bpy` and `mathutils` are supplied by Blender, not the ordinary Python environment. `decord` is used by optional video-reading paths. The renderer expects prepared scene assets, camera poses, and rendering configurations; it does not download third-party assets.

```bash
python -m pip install -r requirements-core.txt
python -m pytest -q tests
python generation/run.py generate --help
python generation/run.py generate-batch --help
python evaluation/evaluate.py --help
```

Data-generation inputs preserve timestamps, intrinsics, poses, object identities, geometry, and visibility provenance. These private fields construct targets and do not enter model prompts. `MEMACT_DATA_ROOT` configures the default data root; explicit command-line paths take precedence.

## Supervised training

A UTF-8 JSONL manifest contains one row per complete serialized episode:

```json
{"directory":"episodes/example"}
```

Paths are relative to the manifest directory or absolute. Each episode contains a `bundle.json` with `events`, `media`, `schedule`, `answers`, and optional cached actions. Media paths can be relative to the episode directory. The loader rejects relative paths escaping that directory.

```bash
torchrun --standalone --nproc_per_node=8 training/train_sft.py \
  --model /path/to/downloaded/model \
  --manifest /path/to/prevalidated/episodes.jsonl \
  --output outputs/sft --accumulation 8
```

The effective batch is `world_size * accumulation` complete sequences. The default is one sequence per GPU and eight accumulation steps. Loss is averaged over assistant tokens in each sequence, then over items. Base/visual weights are frozen; LoRA rank is 8, alpha 16, dropout 0, on query/value projections. AdamW uses learning rate `1e-5`, with gradient norm capped at 1.

Prepare a manifest whose sequences fit the 180,000-token limit. This entry point rejects unfiltered oversize items rather than silently truncating them. It drops an incomplete final optimizer batch and records that count. Assistant events with `loss=false` are rejected because they require an event-aware masking extension. It saves a final adapter and optimizer; exact interrupted-run resume is not implemented in this entry point. Use `--no-save --max-updates 1 --accumulation 1` only for a bounded smoke run.

## Reinforcement learning coverage

The graph core implements the five camera candidates plus answer-now, four continuation samples by default, private reward computation, historical best evidence, invalid-action costs, and the detached candidate-value objective. The reward tests exercise pixel, framing, joint, and historical variants.

`training/train_rl.py` is the native Qwen3.5 LoRA training entry point. It supports single-GPU and torchrun launches, six candidate decisions, cumulative rollout values, a frozen SFT reference, and action/QA/memory item sampling with probabilities 0.50/0.25/0.25. Normal training requires an SFT adapter and validated task and replay manifests. `--smoke-base` permits runtime checks using a fresh adapter; such checks do not reproduce trained-model results. The retained `core.train` is an algorithm test utility.

Rendered RL tasks are loaded through `task_adapter.py`, which checks causal history, required-object measurements, image availability, and branch coverage. A recorded SFT successor or geometry-only Action draft does not supply a counterfactual graph. Image existence checks alone do not certify successful decoding or content hashes. Final release validation reports must distinguish these checks from GPU runtime tests.

## Data selection and provenance

The selected training release contains 1,000,000 QA answer turns: 302,661 Current, 402,662 Memory, and 294,677 Action. Current/Memory membership replaces 42,802 turns whose scenes occur in the separate evaluation bank. Action membership also excludes evaluation scenes and has been replenished to its target. The Action package passed frame hashes, image decoding, timestamps, and the 180,000-token limit. The separate RL package contains 14,001 unique tasks after cross-host deduplication and graph/media validation.

The independent evaluation bank contains 20,000 main QA turns and 2,000 Timing questions. Exact source-qualified recording IDs are excluded from the new training selection. A stricter pre-upload audit identified 3,082 ScanNet QA turns in 173 training streams sharing 41 physical spaces with evaluation through different scan suffixes. This release retains the audited membership and does not claim strict physical-scene separation. Historical `legacy_split` fields are provenance only; the final `split` and manifest membership determine training. Memory-write and memory-retrieval events are preserved and are not counted as QA answer turns.

Annotations reference observations by frame and timestamp. A video alone does not replace the mapping from commands and QA evidence to individual observations. Upstream data should be downloaded through its original provider; source IDs, clip ranges, and reconstruction steps belong in the final release manifest. Generated videos and Action observations require separate media verification before distribution.

This directory contains no experimental scores presented as reproduced measurements. Original MemAct code uses Apache-2.0, and original annotation contributions use CC BY 4.0. Third-party content retains its source terms; see docs/license-scope.md.

## Evaluation coverage

Native streaming and offline Action evaluation now support stateful three-attempt graphs and separate memory/action budgets. The extended frozen evaluation package supplies separate branch graphs for its 1,245 Action questions. The package coverage report verifies all reachable commands under a three-attempt budget while retaining the original QA and passive images. See the multistep workflow below. The native `evaluate` CLI covers main QA; `evaluation/evaluate_timing.py` runs the separate wait/answer protocol.

The native model callback recomputes the retained causal context on each call and uses KV caching within that call. It does not implement the manuscript's persistent cross-call compact KV cache or visual-token eviction. Its reports record `persistent_cross_call_kv=false`; use it for the supplied functional QA/tool protocol, not to reproduce the paper's streaming latency. These prepared data and runtime checks do not establish which data was used in historical experiments, and do not reproduce paper scores.

## Portable annotation shards

The annotation exporter supports both full bundles and training-only `events.json`/`media.json` episodes. Media identifiers use recorded content hashes when available, or opaque source-path hashes otherwise; identifiers alone do not certify media integrity. After preparing the corresponding media tree, `tools/unpack_episode.py --shard SHARD --id ID --media-root DATA_ROOT --output EPISODE_DIR` creates an episode accepted by the SFT loader. It verifies recorded checksums and rejects missing media.

For a full Action package, use:

```bash
python training/prepare_training.py --shards /data/action/shards/*.jsonl.gz \
  --media-root /data/action --output /data/prepared_action --verify-hashes
```

Use the resulting `sft.jsonl` for `train_sft.py`. Apply the same command to passive annotation shards once their media are available. It also writes `qa_replay.jsonl` and `memory_replay.jsonl`; an Action-only package has no memory replay items. Relative image paths in the release are resolved into local runtime paths during preparation. The original downloaded archives remain unchanged.

RL archives extract under `rl/ali/` and `rl/baidu/`. Each package's `manifest.jsonl` contains task paths relative to that package. When combining manifests, prefix task paths with the corresponding package directory; keep tasks, graphs, and assets together. `task_adapter.py` resolves their relative references without using the generation machine's paths.

## Media distribution and reconstruction

The release includes selected self-rendered simulation videos, the generated Action/RL observations, scene GLBs and their referenced external dependencies. Keep video/frame mappings alongside the QA and memory records. Derived camera, geometry and visibility annotations are separate from model-visible inputs. Original real-world recordings and model weights are obtained from their providers.

Upstream acquisition instructions:

- ScanNet: follow the [official data access instructions](https://github.com/ScanNet/ScanNet/blob/master/README.md). Reconstruct RGB video from each selected recording's `.sens` file.
- ScanNet++: use the [official dataset documentation](https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation) to obtain the selected scenes' iPhone RGB videos.
- ARKitScenes: use the [official download instructions](https://github.com/apple-aiml-research/ARKitScenes/blob/main/DATA.md) to obtain the selected recordings' `lowres_wide.zip`. The supplied per-episode mapping selects the RGB members for video reconstruction. Original dataset folds identify download locations; release manifests determine training/evaluation membership. See the physical-space split limitations below.

Install the reconstruction environment:

```bash
python -m pip install -r requirements-reconstruction.txt
```

Rebuild one prepared ScanNet chunk (zero-based `--chunk-index`; `_part3` means index 2):

```bash
python reconstruction/reconstruct_real_video.py --profile scannet \
  --input /data/ScanNet/scene0222_00/scene0222_00.sens \
  --chunk-index 2 --output /data/rebuilt/scene0222_00_part3/output.mp4 \
  --report /data/rebuilt/scene0222_00_part3/reconstruction.json
```

For ScanNet++, use `--profile scannetpp --input /data/scene/iphone/rgb.mkv` and the zero-based chunk index (`_chunk_1` means index 1). ScanNet and ScanNet++ reproduce the historical 24 FPS, 2160-frame chunking rules. ScanNet uses the historical index-based 30 FPS source clock; ScanNet++ follows decoded video timestamps. They preserve the original resize and MPEG-4 encoding settings.

For ARKitScenes, use the released episode mapping:

```bash
python reconstruction/reconstruct_real_video.py --profile arkit \
  --input /data/ARKitScenes/41007603/lowres_wide.zip \
  --mapping /data/metadata/41007603_part002/frame_mapping.json \
  --output /data/rebuilt/41007603_part002/output.mp4 \
  --report /data/rebuilt/41007603_part002/reconstruction.json
```

The script writes a new video and a provenance report, verifies that the output decodes, and refuses to overwrite existing outputs. It reconstructs RGB media using supplied derived annotations; it does not recompute QA labels or object geometry.

Then restore training frames using a relative-path JSONL manifest:

```json
{"video":"scene0222_00_part3/output.mp4","frames":[{"index":0,"output":"media/IMAGE_SHA256.jpg","sha256":"IMAGE_SHA256","width":640,"height":480}]}
```

```bash
python reconstruction/reconstruct_frames.py --manifest /data/frame_jobs.jsonl \
  --video-root /data/rebuilt --output-root /data/passive \
  --report /data/frame_reconstruction_report.json --workers 8
```

Frame indices are zero-based indices into the prepared video, not indices into the upstream recording. Extraction uses JPEG quality 95. Existing outputs are checksum-verified and reused; mismatched files are not overwritten. Incorrect hashes stop the affected video job and are listed in the report. Preserve the decoder/encoder environment when exact byte identity is required; do not silently replace checksums after a mismatch.

Validation so far: prepared-video extraction matches all eight sampled training JPEGs across the four sources. An ARKit RGB-archive-to-video-to-JPEG sample also matches its recorded training JPEG hash. Public raw ScanNet and ScanNet++ samples passed end-to-end reconstruction: respectively 2,160/1,453 prepared frames and 90/61 sampled JPEGs. Synthetic tests also verify ScanNet's failed-decode sampling behavior and truncated-input handling. These checks do not claim full-corpus raw-media reproduction.

A released `video_jobs.jsonl` drives batch reconstruction. Place downloaded inputs under the relative paths declared in each job (`ScanNet/SCENE/SCENE.sens`, `ScanNet++/SCENE/iphone/rgb.mkv`, or `ARKitScenes/SCENE/lowres_wide.zip`). Keep `metadata/` beside the manifest. For example:

```bash
python reconstruction/reconstruct_media.py --manifest /data/reconstruction/video_jobs.jsonl \
  --raw-root /data/upstream --output-root /data/memact \
  --report /data/reconstruction/videos_report.json --workers 4
python reconstruction/reconstruct_frames.py --manifest /data/reconstruction/frame_jobs.jsonl \
  --video-root /data/memact --output-root /data/memact \
  --report /data/reconstruction/frames_report.json --workers 8
```

The batch runner verifies completed video receipts before reusing them. `--profile arkit --limit 1` runs one selected reconstruction for a small environment check. Full-scene upstream video decoding can be repeated when several chunks from one recording are selected; avoid very high worker counts on network storage. The current reconstruction manifest covers passive training media; evaluation media reconstruction is tracked separately during release preparation.

## Frozen evaluation package preparation

After extracting evaluation archives and reconstructing their real frames, run:

```bash
python evaluation/prepare_evaluation.py --package /data/evaluation --output /data/evaluation_runtime
```

This validates media checksums and writes native episode bundles plus separate main and Timing scoring files. `--episode ID` supports a bounded check. Scoring records are evaluator-only inputs. Run the three-attempt coverage checker on prepared Action episodes before an active comparison. Both training and evaluation reconstruction job manifests are provided in the release.

## Physical-space split gate

`tools/check_physical_splits.py` checks source/scene JSONL manifests using ScanNet space IDs (not scan suffixes), ScanNet++ scene IDs (not chunks), and ARKitScenes official visit IDs (not video IDs). It returns nonzero on overlaps or unknown groups. Its input schema is documented by `--help`.

The pre-upload audit found 3,082 ScanNet QA turns and 5,197 ARKitScenes QA turns crossing these stricter group boundaries, plus 755 ARKit QA turns with unknown visit IDs. The audited training membership is retained; physical-group remediation is outside the current release scope. It is not certified physically disjoint. Action/RL simulation membership is unaffected.

## Multistep Action evaluation and generation

Camera commands are fixed at 0.5 m forward/left/right or 45 degrees left/right.
`action_graph.py` tracks the camera state separately for each question. Invalid
physical actions consume an attempt and keep the same state. Missing branches
raise a coverage error. The evaluation runtime has independent memory-call and
camera-attempt limits and requests a final answer after the last camera attempt.
No graph geometry, rewards, or unchosen images enter the model context.

Convert an existing rendered task (including packaged RL tasks) to a portable
bundle with images:

```bash
python generation/prepare_action_graph.py --task /path/to/task.json --output /new/graph_bundle
```

Use only graphs belonging to the evaluated question and scene. Training graphs
must not be substituted for frozen evaluation questions. A native episode can
contain `action_graph` in `bundle.json` or a separate `action_graph.json`.
A manifest row may also provide `action_graph_bundle`, relative to its episode
root or absolute, to add a separately prepared graph and its media. For multiple
questions, the graph's optional `question_roots` maps schedule indices (strings)
to initial nodes. Selection masks remap those indices.

```bash
python evaluation/check_action_coverage.py --manifest episodes.jsonl --steps 3 --report coverage.json
python evaluation/evaluate.py --manifest episodes.jsonl --model /path/to/model \
  --mode streaming --max-actions 3 --max-calls 4 --output /new/evaluation
python evaluation/evaluate.py --manifest episodes.jsonl --model /path/to/model \
  --mode offline32 --max-actions 3 --output /new/offline_evaluation
```

`--adapter` is optional for base-model comparisons. Offline Action evaluation
starts from up to 32 causal frames and appends each acquired action observation.
Legacy one-step caches are rejected for multistep budgets; installing this code
does not create missing evaluation images. Use the extended evaluation package rather than an older one-step-cache release.
Coverage failures are reported explicitly.

Generate new graphs from GLB geometry and a causal camera history:

```bash
python generation/generate_action_graph.py --config render_input.json \
  --blender /path/to/blender --gpu 0 --steps 3 --output /new/generated_graph
# Restart an interrupted run without changing its inputs:
python generation/generate_action_graph.py --config render_input.json \
  --blender /path/to/blender --gpu 0 --steps 3 --output /new/generated_graph --resume
```

The JSON input contains `assets` (parent of `<scene_id>/front3d_scene.glb`),
`state`, `history`, and `question`. Paths resolve relative to this JSON file.
`state` contains `scene_id`, `state_id`, `frame_id`, `timestamp_sec`,
`focal_length_mm`, `poses.base` (`location`, `rotation_euler_deg`), object records
with categories and centers, and `source_render_contract`. The latter contains
floor/wall texture and HDRI paths and `actual_render_settings`
(`samples`, `bounces`, `exposure`, `denoise`, width 640 and height 480).
`history` contains chronological `frame_id`, `timestamp_sec`, and `poses.base`
records, ending at the question anchor. `question` specifies a `template`
(`object_size_m`, `object_to_object_distance_m`, or `object_to_object_distance_bucket`;
`object_spatial_proximity` is supported for legacy frozen distance-range questions) and one or two object `targets`.
Initial-view referents and answers are computed from visibility and geometry.
This entry point does not perform LLM wording diversification.

The generator renders causal history, expands all five commands from reachable
nonterminal states, checks collisions, and records invalid self-loops. Equivalent
poses are reused within the graph. Completed layer receipts are hash-checked on
resume; interrupted directories are preserved. Outputs include a relative-path
`bundle.json`, an RL-compatible `task.json` plus private graph, and
`demonstrations.json` and `demonstrations.jsonl.gz`. The latter is accepted by
`training/prepare_training.py --shards ... --media-root /new/generated_graph
--output /new/prepared_training`. Demonstrations terminate at the earliest qualifying
prefix on each path (pixel fraction 0.005, framing 0.80, occlusion support 0.70,
historical gain ratio 1.5 and absolute gain 0.002). Zero accepted demonstrations
is a valid result; a complete RL graph need not yield an SFT demonstration.
Image-quality rules are evidence proxies, not guarantees of RGB answerability.

To preserve a frozen evaluation question, put its original public payload and
answer in `question.payload` and `question.answer`. The generator checks the
answer against geometry and preserves option order and wording. An incompatible
answer is an error; it is never silently relabeled.


## Timing evaluation

Use the `timing.scoring.jsonl` produced by `evaluation/prepare_evaluation.py`:

```bash
python evaluation/evaluate_timing.py --scoring /prepared/timing.scoring.jsonl \
  --model /path/to/model --output /new/timing_results
# Or score previously saved predictions without loading a model:
python evaluation/evaluate_timing.py --scoring /prepared/timing.scoring.jsonl \
  --predictions /results/predictions.jsonl --output /new/timing_scores
```

The optional `--adapter` loads a trained adapter. Each question runs independently
with its original chronological 1 FPS prefix. The initial checkpoint may precede
the first image; it then contains the question only. The model emits exactly
`{"type":"wait"}` or `{"type":"answer","answer":value}`. Its first answer ends
the question. No memory or action execution is exposed. The private answer,
first-answerable time, deadline, and expected decision sequence are never sent
to the model. An answer at the deadline is eligible; waiting there is unanswered.
T, JA, and JM retain all applicable questions in their denominators, including
failed, early, and unanswered cases. `--limit` is for bounded runtime checks and
must not be confused with a full evaluation.

## Frozen-question rendering inputs

The release supplies `generation/frozen_action/configs/` and an input manifest
for all 1,245 frozen Action questions. Extract these inputs, simulation GLBs,
and rendering textures/HDRIs into the same data root. Paths in each config
resolve relative to its own location. `dependencies.json` records required
asset paths and SHA-256 hashes. Inputs contain evaluator-only geometry and
answers; they are never model prompts. Several questions can share the same
rendered camera graph.

```bash
python generation/generate_action_graph.py \
  --config DATA/generation/frozen_action/configs/QUESTION_ID.json \
  --blender /path/to/blender --gpu 0 --steps 3 --output WORK/new_graph
```

Released graphs preserve the original starting images and labels. Re-rendering
may show small pixel differences across GPU/Blender builds; do not overwrite
frozen reference images or relabel an incompatible question.
