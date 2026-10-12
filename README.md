# MemAct

**Streaming Spatial Understanding with Memory and Active Exploration**

[Project website](https://vfishc.github.io/memact/) · [Paper PDF](docs/assets/memact-paper.pdf) · [Code guide](CODE_GUIDE.md) · [Data & downloads](#data--downloads)

Yuxi Wei¹, Wei Huang¹, Qirui Chen³, Lu Hou²*, Xiaojuan Qi¹*

¹ The University of Hong Kong  
² Yinwang Intelligent Technology Co. Ltd  
³ Shanghai Jiao Tong University  
\* Corresponding authors

MemAct combines incremental memory writing, historical evidence retrieval, and active camera exploration in a shared vision–language model for streaming spatial understanding.

## Resources

| Resource | Location / status |
| --- | --- |
| Paper | [Non-anonymous preprint PDF](docs/assets/memact-paper.pdf) |
| arXiv | Pending — the identifier will be added after submission |
| Project page | https://vfishc.github.io/memact/ |
| Code | https://github.com/vfishc/memact |
| Hugging Face dataset | [vfishc/memact](https://huggingface.co/datasets/vfishc/memact) — public repository; upload in progress |
| Model weights | Not included in this release |

The dataset repository is `vfishc/memact`. It is public; data upload is in progress. Use the completed release manifest and checksums to verify downloaded contents.

## Code organization

| Directory | Purpose |
| --- | --- |
| `generation/` | Metadata, Current/Memory QA, rendered Action graphs, and dataset assembly |
| `reconstruction/` | Reconstruct upstream real videos and referenced frames |
| `training/` | Manifest preparation, SFT, cumulative-return RL, and replay |
| `evaluation/` | Main QA, streaming/offline inference, Timing, and Action coverage |
| `pipeline/` | Shared runtime, memory router, model interfaces, and reward implementation |
| `tools/` | Export, unpacking, and integrity utilities |
| `tests/` | Maintained correctness tests and bounded GPU smoke entry points |
| `configs/` | Method and runtime configuration |
| `docs/` | Project website, paper, environment and data documentation |

This source snapshot contains no prior Git history, checkpoints, caches, or development logs. The October 12 source validation passed 81 tests and 25 command-line help checks. These are implementation checks, not a rerun of the paper's experimental results.

### Implementation scope

The release implements the causal QA/tool protocol, memory management, SFT/RL entry points, and stateful multi-step Action generation and evaluation. Its native model callback recomputes retained causal context between calls and uses KV caching within each call. It does not implement persistent cross-call compact KV eviction, and does not reproduce the paper's streaming latency. See [the code guide](CODE_GUIDE.md) for environment, commands, tested coverage, and remaining runtime limitations.

## Quick start

```bash
python -m pip install -r requirements-core.txt
python -m pytest -q tests
python generation/run.py generate --help
python training/train_sft.py --help
python training/train_rl.py --help
python evaluation/evaluate.py --help
python evaluation/evaluate_timing.py --help
```

Model training additionally requires the GPU dependencies described in [CODE_GUIDE.md](CODE_GUIDE.md). Obtain model weights and upstream datasets separately.

## Data & downloads

The prepared data release is separate from this Git repository. All download entries below will point into `vfishc/memact` once uploaded. Published manifests, not paper-era intermediate totals, define this release.

| Component | Prepared contents | Dataset path / download status |
| --- | --- | --- |
| Current and Memory QA | 302,661 Current + 402,662 Memory answer turns | `data/train/current_memory/` — pending |
| Memory supervision | 519,948 memory writes and 556,635 retrieval calls, additional to QA turn counts | Included in the Current/Memory streams — pending |
| Action SFT | 294,677 trajectories; 534,752 camera transitions | `data/train/action/` — pending |
| RL | 14,001 tasks with branch observations and reward metadata | `data/rl/` — pending |
| Main evaluation | 20,000 QA turns, including complete graphs for 1,245 Action questions | `data/evaluation/` — pending |
| Timing evaluation | 2,000 wait/answer questions, separate from main QA | `data/evaluation/` — pending |
| Simulation media and assets | Selected rendered videos and associated scene GLBs | `media/simulation/` — pending |
| RL rendering assets | Scene dependencies, textures/HDRIs and render configurations | `assets/rl/` — pending |
| Frozen Action generation | Render inputs for the frozen evaluation questions | `generation/frozen_action/` — pending |
| Temporal and spatial metadata | Frame IDs, timestamps, camera/object information and evidence mappings | `metadata/episodes/` — pending |
| Real-media reconstruction | Training/evaluation reconstruction jobs and mappings | `reconstruction/` — pending |

The final preparation manifest records 206 archive artifacts totaling 589,652,365,064 bytes (about 589.7 GB), plus supporting metadata. Existing archive paths are preserved so references remain valid. Checksums and release statistics accompany the data.

Original ScanNet, ScanNet++, and ARKitScenes recordings are obtained from their providers; they are not redistributed here. Reconstruction instructions and scripts recover the prepared videos and referenced frames. Model weights are excluded. Derived supervision, evaluator-only answers/geometry, and model-visible inputs remain distinct.

Training and evaluation use separate selected source-recording IDs. Strict physical-space disjointness is not claimed: some real sources contain multiple recordings of the same space. Existing audit findings are retained in release documentation.

See [release statistics](docs/release-statistics.json) and [Hugging Face publication instructions](docs/huggingface-upload.md). Original MemAct code is licensed under [Apache-2.0](LICENSE). Original MemAct annotation contributions are licensed under CC BY 4.0; third-party code, geometry, textures, media and upstream-derived content retain their applicable source terms. See [license scope](docs/license-scope.md).

## Local website preview

```bash
python3 -m http.server 8765 --directory docs
```

Open http://localhost:8765/. GitHub Pages should publish the `main` branch's `/docs` directory. No website build step is required.
