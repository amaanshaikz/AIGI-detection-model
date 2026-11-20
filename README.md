# aigi-detloc

A lightweight model that says whether an image is AI-generated or AI-edited, and exactly where it was edited
(a pixel mask). It follows the design doc "Lightweight AIGI Detection + Localization Model", simplified: one MAC
detection head (no deep-supervision heads) and no edit-type head.

About 100M parameters at inference (86M of them a frozen DINOv3 ViT-B/16), one backbone pass per tile.

```
image ─┬─ DINOv3 ViT-B/16 + LoRA ───────────────┐
       └─ frozen forensic stream                 ├─ gated fusion ─┬─ MAC head ───────────── p_global ─┐
          (fingerprint + noise print) ───────────┘                └─ coarse head ─► refine head ─► mask M ─┤
          (edge prior) ───────────────────────────────────────────────────────────┘                       │
                                                                       score = max(p_global, top 0.5% of M) ┘
```

## One command

```bash
pip install -r requirements.txt
huggingface-cli login                       # DINOv3 and some datasets are gated; accept their licenses first
python scripts/run_all.py --preset laptop   # download + convert + merge + audit, train, calibrate, test reports
```

`run_all.py` = `prepare_data.py` (data) + the three forensic-CNN trainers + `train.py` + `calibrate.py` + `report.py`,
each also usable on its own. Use `--preset cluster` for the 1.5M-image mix. The forensic stages are optional: if one
fails it warns and training uses a random frozen network in its place (much weaker). See "The three forensic CNNs".

### The data step

```bash
python scripts/prepare_data.py list    --preset laptop   # what it fetches, caps, licenses, how each schema was checked
python scripts/prepare_data.py inspect sidset_train      # a dataset's REAL columns (do this for any UNVERIFIED one)
python scripts/prepare_data.py run     --preset laptop   # stream from the Hub, convert, merge, audit
```

- **Hugging Face sources** (SID-Set, So-Fake-Set, Community Forensics, OpenFake, OpenSDI, So-Fake-OOD) are streamed:
  only the rows kept are downloaded. A source that fails is reported and skipped, never silent.
- **Manual sources** (CASIAv2, IMD2020, SAGI-D, TGIF2, GIM, BR-Gen, real photos, and test sets like Chameleon, WildRF,
  Synthbuster, CocoGlide) have no Hugging Face copy. Download them, put images under `data/raw/<id>/<type>/` and masks
  under `data/raw/<id>/<type>_masks/` (matched by file name; `<type>` = real, full, edited or trad). `run` picks them up.
- **Held-out data:** So-Fake-Set's `flux.1-dev` generator is excluded from training and used for validation. Test
  sources are never merged into training.
- The merge is audited for format/size shortcuts (`audit_bias.py`); add `--with-vae` for VAE copies of the real images.

Schemas marked UNVERIFIED in `list` were not checked against the real dataset. Nothing here has been run against the
real Hub datasets (tests use local stand-ins with the documented schemas).

## The three forensic CNNs

The model reads low-level traces through three small CNNs that are trained once and then frozen. Each has its own
script; `run_all.py` runs all three.

| CNN | What it learns | Script | Data it needs |
| --- | --- | --- | --- |
| Edge prior | Where editing seams are | `pretrain_edge.py` | Images with masks (any merged manifest) |
| Generator fingerprint | Traces of generators that survive a VAE round trip | `build_fp_pairs.py` then `pretrain_fingerprint.py` | Your train manifest plus a VAE from the Hub (`pip install diffusers`) |
| Camera noise print | Camera and processing noise that a splice breaks | `pretrain_noiseprint.py`, or `convert_noiseprint.py` | Camera-labeled photos (below) |

```bash
# fingerprint: pairs = a crop and its VAE reconstruction (real, generated, and edited-inside/outside the mask)
python scripts/build_fp_pairs.py --manifest data/train.csv --out data/fp_pairs --n 6000 --vae sd15 sdxl
python scripts/pretrain_fingerprint.py --pairs data/fp_pairs/pairs.csv --out runs/fingerprint.pt

# noise print, option A: train it (needs camera labels; see below)
python scripts/pretrain_noiseprint.py --manifest data/train.csv --out runs/noiseprint.pt
python scripts/pretrain_noiseprint.py --camera-dirs data/raw/cameras --labels folders --out runs/noiseprint.pt
# noise print, option B: convert the published Noiseprint weights you download yourself (non-profit license)
python scripts/convert_noiseprint.py --src noiseprint-pytorch/pretrained_weights/model_qf101.pth --out runs/noiseprint.pt
```

Then set `model.fingerprint_ckpt`, `model.noiseprint_ckpt` and `model.edge_prior_ckpt` in your config (`run_all.py`
does this for you). The CNN sizes must match the config: real runs keep `forensic_width: 64`, `forensic_depth: 17`.

### Guide: getting camera data for the noise print

The noise print is only as good as its camera labels: it learns "same camera model" versus "different camera model".

1. **Check what you already have.** `pretrain_noiseprint.py` reads EXIF Make + Model from your real photos and prints how
   many camera models it found. Many web datasets strip EXIF, so this often finds nothing; the script then falls back
   to a weaker mode (two patches from the same photo count as "same") and says so.
2. **Add camera-labeled photos** if you want the better mode. The Noiseprint paper trained on cameras from the Dresden
   Image Database, the Socrates dataset and VISION (125 cameras, 70 models). Download any of them, or your own phones and
   cameras, and put one sub-folder per camera model under one folder:
   ```
   data/raw/cameras/Nikon_D70/*.jpg   data/raw/cameras/iPhone_12/*.jpg   data/raw/cameras/Canon_EOS_5D/*.jpg ...
   ```
   Aim for at least 20 camera models and 20+ photos each. Run `--labels folders --camera-dirs data/raw/cameras`. The
   last lines print retrieval accuracy on held-out cameras next to the chance level; it should be clearly above chance.
   I did not verify these datasets' download pages or licenses: search for each name and read its terms.
3. **Or skip training** and convert the published weights (option B above). The weight files come per JPEG quality
   factor (`qf51` to `qf100`, plus `qf101` for uncompressed images). The converter checks its result against the original
   network on every run and refuses to save if they differ. Their license allows non-profit use only and requires citing
   the authors; this repo does not include the weights.
4. **If a script cannot find cameras at all**, you can run without the noise print: leave `noiseprint_ckpt` unset (a
   random frozen network is used) or use `--skip-noiseprint` in `run_all.py`.

## Layout

| Path | What it holds |
| --- | --- |
| `aigi/model.py` | Backbone wrapper, fusion gate, MAC / coarse / refine heads, output rule |
| `aigi/forensic.py` | DnCNN (fingerprint and noise print), EDGER-style edge prior |
| `scripts/build_fp_pairs.py`, `pretrain_fingerprint.py`, `pretrain_noiseprint.py`, `convert_noiseprint.py`, `pretrain_edge.py` | The three forensic-CNN trainers and the Noiseprint converter |
| `aigi/data.py` | Manifest dataset, crop-not-rescale sampling, label rules, soft edge targets |
| `aigi/degrade.py` | The four-level degradation curriculum |
| `aigi/losses.py` | Focal, Dice, SupCon and the total loss |
| `aigi/infer.py` | Tiled native-resolution inference, Hann blending, decision rules |
| `scripts/` | One script per training stage, plus calibration and inference |
| `configs/default.yaml` | Every setting, with the source of each default in comments |
| `tests/` | End-to-end tests on toy data with a tiny random backbone |

## What you need

1. **Python 3.10+ and a GPU.** Everything runs on CPU for the tests, but real training needs a GPU.
2. **DINOv3 access.** The weights are gated: accept Meta's license on the
   [model page](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) and run `huggingface-cli login`.
   Check the license allows your use (research, challenge entry, or commercial).
3. **Data**, listed in the design doc's training data plan, written as manifests (format below).
4. **Optional:** camera-labeled photos for the noise print (see "Guide: getting camera data"), and `diffusers`
   for the fingerprint's VAE.

```bash
pip install -r requirements.txt
pytest -q          # about 3 minutes on CPU
```

## Data format

One CSV per split. Paths are relative to the CSV's folder.

```csv
path,type,mask,alts
real/0001.jpg,real,,recon/sd15/00000000.png;recon/flux/00000000.png
gen/flux_0042.png,full,,
edits/qwen_0007.png,edited,edits/qwen_0007_mask.png,
casia/Tp_0003.jpg,trad,casia/Tp_0003_gt.png,
```

- `type`: `real`, `full` (fully generated), `edited` (AI edit), or `trad` (classic splice, copy-move, removal).
- `mask`: white = altered pixels. Required for `edited` and `trad`, the same size as the image.
- `alts`: VAE reconstructions of a real image, added by `scripts/make_vae_recon.py`. They are treated as real.
- For fully regenerated edits, mark only the edited region. The untouched background went through the VAE but stays real.

Keep a separate **held-out** manifest whose generators and editors never appear in training. It is used for
validation and threshold calibration, and stands in for the hidden test set.

Before training, apply the bias controls from the design doc: the same JPEG and size handling for real and fake
images, and no metadata.

## Training, stage by stage

```bash
# 0. Add VAE reconstructions of the real images (labeled real)
python scripts/make_vae_recon.py --manifest data/train.csv --out-dir data/vae_recon \
    --out-manifest data/train_alts.csv --vaes sd15 sdxl flux

# 1a. Generator fingerprint: build the pairs, then train
python scripts/build_fp_pairs.py --manifest data/train.csv --out data/fp_pairs --n 6000 --vae sd15 sdxl
python scripts/pretrain_fingerprint.py --pairs data/fp_pairs/pairs.csv --out runs/fingerprint.pt

# 1b. Edge prior, on spliced edits
python scripts/pretrain_edge.py --config configs/default.yaml --manifest data/edge_train.csv --out runs/edge_prior.pt

# 2-3. Joint training with the degradation curriculum (set the *_ckpt paths and manifests in the config first)
python scripts/train.py --config configs/default.yaml        # add --resume runs/main/last.pt to continue

# 4. Thresholds on held-out data, then a robustness sweep with those thresholds
python scripts/calibrate.py --ckpt runs/main/best.pt --manifest data/val_heldout.csv
for L in 2 3 4; do
  python scripts/calibrate.py --ckpt runs/main/best.pt --manifest data/test.csv \
      --thresholds runs/main/thresholds.json --level $L
done
```

The fingerprint pairs are described at the top of `scripts/build_fp_pairs.py`.

## Inference

```bash
python scripts/infer.py --ckpt runs/main/best.pt --thresholds runs/main/thresholds.json --out out/ photo.jpg
# {"image": "photo.jpg", "fake": true, "score": 0.93, "mask": "out/photo_mask.png"}
```

Images are never downscaled. Anything over 1024 px per side is tiled and the masks are blended. A real verdict
always comes with an empty mask. There is no edit-type output: the result is a fake score and a mask.
For SAFE, wrap `aigi.infer.predict` + `decide` in the HTTP service format the organizers specify.

## Notes and limits

- The tests check wiring and data rules, not accuracy. Accuracy depends on the data.
- What was and was not shown: the fingerprint trainer learned on synthetic traces (its generator head went from chance
  to 59% in 50 steps), the noise-print trainer learned to separate clearly different synthetic cameras (retrieval
  accuracy 0.54 to 1.00), and the Noiseprint converter matches the published network to 4e-6 on the real qf101
  weights. None has been run on real camera or generator data, and the noise-print trainer did NOT separate subtly
  different synthetic cameras at toy scale: how well it generalizes to unseen real cameras depends on your camera data.
- Noiseprint++ (TruFor) weights are a different file format and are not covered by `convert_noiseprint.py`.
- Learning rates and loss weights are starting points from the cited papers, marked "start" in the config.
- Evaluation of the ablations in the design doc is up to you; each one is a config change or a one-line edit.

## Laptop first, then a cluster

- **Laptop:** `configs/laptop.yaml` uses DINOv3 ViT-S/16 (32.5M parameters in total, 9.7M trainable), crops of
  256-384 px and batch 4. On CPU a batch-4 step at 256 px took about 12 s in one measurement, so use CPU only for
  `pytest` and smoke tests. Use a GPU for real runs.
- **Cluster:** `train.py` runs on one GPU in one process. Multi-GPU (DDP via `torchrun`) and gradient accumulation
  are not implemented yet; add them before scaling. Pre-download the DINOv3 and VAE weights on a login node
  (`HF_HUB_OFFLINE=1` on compute nodes) and copy the data to node-local fast storage.

## Turning public datasets into training data

See the "Training Data Plan" doc for which datasets to use and why. Three tools do the work:

| Script | What it does |
| --- | --- |
| `scripts/build_manifest.py folder ...` | One folder of images of one type (+ masks matched by name) → a manifest CSV |
| `scripts/build_manifest.py hf ...` | A Hugging Face dataset (image, label, mask columns) → images on disk with their original bytes + a manifest. `--streaming --limit N` downloads only N rows |
| `scripts/merge_manifests.py out.csv a.csv:5000 b.csv:all` | Merge per-source manifests with a cap per source; paths rewritten; duplicates dropped |
| `scripts/audit_bias.py train.csv` | Can file format/size/JPEG quality alone predict real vs fake? Near 0.5 AUC = good; above 0.65 = fix the data first |

```bash
python scripts/build_manifest.py hf --dataset saberzl/SID_Set --split train --streaming --limit 6000 \
    --label-map 0=real,1=full,2=edited --mask-col mask --source sidset \
    --export-dir data/raw/sidset --out data/manifests/sidset.csv
python scripts/merge_manifests.py --out data/train.csv data/manifests/sidset.csv:all data/manifests/casia2.csv:all
python scripts/audit_bias.py data/train.csv
```

## Demo and fixes in this version

- `configs/demo.yaml` + `scripts/run_demo.py`: an offline synthetic benchmark (`scripts/make_demo_data.py`) with
  held-out photos and a held-out generator style, then pretraining, training, calibration and HTML reports.
- `model.full_finetune: true` (tiny/demo backbones only): a RANDOM backbone must train fully. LoRA on it froze 92% of
  the weights, including the patch embedding, and blocked learning. Real pretrained DINOv3 runs keep LoRA.
- `train.py --run-epochs N`: stop after N epochs and `--resume` later (job time limits); writes `history.csv`.
# AIGI-detection-model
