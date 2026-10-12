# Publishing the StreaMemAct dataset

## Repository identity

Proposed dataset repository: `vfishc/memact`.

Owner: vfishc. Dataset name: MemAct (case-sensitive). Visibility: public, authorized by the owner on 2026-10-12. License scope: original annotations CC BY 4.0, original code Apache-2.0, upstream assets retain their terms. Model weights are outside this release.

Create a **Dataset** repository (not a Model repository or Space) using https://huggingface.co/new-dataset. Confirm the account's storage allowance for approximately 590 GB before starting; do not enable paid storage without the owner's approval. Public and private storage allowances differ.

## Authentication

On the upload host, use the Hugging Face CLI's interactive authentication:

```bash
export HF_HOME=/mnt/cpfs/weiyuxi/.cache/huggingface
mkdir -p "$HF_HOME"
chmod 700 "$HF_HOME"
hf auth login
hf auth whoami
```

Use an account token with write access to the selected dataset repository. Enter it locally in the authentication prompt; never put it in this repository, a shell command, or a chat message.

## Upload

The administrator will bind `RELEASE_DIR` to the validated final staging directory and fill in the real repository ID. Do not upload a parent development directory.

```bash
hf upload vfishc/memact /path/to/validated/release --repo-type dataset
```

Use the installed client's documented resumable large-upload mode. Current Hub documentation recommends `hf upload`; older versions may expose `hf upload-large-folder`. Check `hf --version` and `hf upload --help` before choosing the command. Preserve upload progress/cache state between retries. Upload each release artifact once; don't push original real recordings or model weights.

After upload, verify remote file inventory, sizes, checksums/manifests, and sample downloads. Update README and website download links only after the repository contents are available. Keep evaluator-only labels and metadata distinct from model inputs.

References:
- https://huggingface.co/docs/hub/datasets-adding
- https://huggingface.co/docs/huggingface_hub/en/guides/upload
- https://huggingface.co/docs/huggingface_hub/en/guides/cli
