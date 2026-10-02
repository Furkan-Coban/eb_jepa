# Video JEPA: Bottom-Up First-Run Plan

## Goal

The main focus of this study is Video JEPA. I-JEPA is included only as brief
background for the representation-learning idea.

The bottom-up sequence is:

1. Predict video dynamics in latent space without actions (`video_jepa`).
2. Inspect one-step and multi-step future-latent predictions.
3. Add action conditioning (`ac_video_jepa`).
4. Compare prediction errors for correct and shuffled actions.

The first goal is not benchmark reproduction. The goal is to trace the JEPA
data flow in code, run a small training job, and collect controlled evidence of
what the model learned.

> Terminology note: the Moving MNIST `video_jepa` example in this repository is
> not a small reproduction of Meta's original V-JEPA architecture. It is an
> educational future-latent prediction example. It does not implement the
> original masked spatiotemporal prediction and EMA target-encoder setup.

---

## Experiment 1 — State-Only Video JEPA (Primary Run)

### Question

Can the model predict the representation of a future state using only past
video frames?

```text
past frames
    |
    v
encoder (ResNet5)
    |
    v
past latents
    |
    v
state-only predictor (ResUNet)
    |
    v
predicted future latent
    |
    v
compare with the latent of the real future frame
```

Simplified computation:

```text
z_t       = encoder(o_t)
z_target  = encoder(o_t+1)
z_pred    = predictor(previous_latents)
loss_pred = distance(z_pred, z_target)
```

This example has no separate teacher or EMA target encoder. The same encoder
also produces the future target representation.

### Data

- Dataset: Moving MNIST.
- Two digits move inside an image and bounce at the boundaries.
- The repository downloads about 800 MB on first use.
- The source is split into 9,000 training and 1,000 validation sequences.
- Each sequence is split temporally into two shorter clips, so the DataLoaders
  see 18,000 training and 2,000 validation clips by default.

### Model and objectives

- Encoder: `ResNet5`.
- Predictor: `ResUNet` with two latent frames of context.
- Projector: MLP.
- Prediction loss: MSE between predicted and target future latents.
- VC regularizer:
  - variance/std loss discourages collapse;
  - covariance loss discourages redundant latent dimensions.
- Auxiliary evaluation heads:
  - pixel decoder;
  - digit-location detection head.

The decoder and detection head are used to interpret and visualize latent
rollouts. JEPA itself predicts representations rather than pixels.

### Two-stage execution

#### A. Smoke test

The smoke test only verifies that:

- data loads;
- forward and backward passes run;
- losses are finite;
- the GPU is used;
- a checkpoint is written.

Use the subset settings added to the Video JEPA config:

```yaml
data:
  train_size: 512
  val_size: 128
```

Target smoke configuration:

```text
train_size=512
val_size=128
batch_size=32
model.steps=1
epochs=1
wandb=False
```

#### B. Mini learning run

After the smoke test succeeds:

```text
train_size=2,000–4,000
val_size=256–512
batch_size=32 or 64
model.steps=2
epochs=5–10
wandb=False
```

Evidence of learning should include:

- initial versus final prediction loss;
- training versus validation metrics;
- decoded rollout from a trained model;
- error as the prediction horizon grows.

### Colab command

```bash
uv run python -m examples.video_jepa.main \
  --fname examples/video_jepa/cfgs/default.yaml \
  --folder /content/eb_jepa_runs/video_mini \
  logging.log_wandb=False \
  logging.log_every=1 \
  data.train_size=4000 \
  data.val_size=512 \
  data.batch_size=64 \
  data.num_workers=2 \
  model.steps=2 \
  optim.epochs=5
```

If a T4 runs out of memory, reduce the batch size to 32 and then 16.

### Outputs to retain

- exact command and resolved config;
- GPU model (`nvidia-smi`);
- epoch and total runtime;
- training and validation metrics;
- checkpoint;
- loss plot;
- ground-truth / predicted-rollout / detection GIF;
- tensor shapes for input, encoded state, and predicted state.

---

## Experiment 2 — Action-Conditioned Video JEPA

### Question

When the future depends on an external action, does the model use that action
correctly?

```text
observation o_t
    |
    v
Impala encoder
    |
    v
latent z_t + action a_t
    |
    v
GRU predictor
    |
    v
predicted z_(t+1)
```

```text
z_t      = encoder(o_t)
z_next   = encoder(o_t+1)
pred     = predictor(z_t, a_t)
pred_err = distance(pred, z_next)
```

This example also has no separate target or teacher encoder.

### Data and objectives

- Dataset/environment: Two Rooms, generated online.
- Observation: agent and wall image.
- Action: two-dimensional movement vector.
- Encoder output: global 512-dimensional latent.
- Predictor: action-conditioned GRU.
- Objectives:
  - latent prediction;
  - variance;
  - covariance;
  - temporal similarity;
  - inverse dynamics (IDM).

### Small run

Planning evaluation, W&B, and compilation are disabled for the first run:

```bash
uv run python -m examples.ac_video_jepa.main \
  --fname examples/ac_video_jepa/cfgs/train.yaml \
  --folder /content/eb_jepa_runs/ac_video_mini \
  logging.log_wandb=False \
  meta.load_model=False \
  meta.enable_plan_eval=False \
  model.compile=False \
  training.dtype=float16 \
  data.size=2048 \
  data.val_size=128 \
  data.batch_size=32 \
  data.num_workers=2 \
  model.nsteps=2 \
  optim.epochs=3
```

`float16` is selected for a T4. `bfloat16` may be tested on newer GPUs.

### Action-sensitivity check

Different actions producing different outputs is not sufficient: an untrained
network can also do that. Instead, shuffle actions across the batch:

```text
error_correct = distance(P(z_t, a_t),        z_(t+1))
error_wrong   = distance(P(z_t, shuffled_a), z_(t+1))
```

Expected result:

```text
error_correct < error_wrong
```

The notebook also plots ground-truth, correct-action, and shuffled-action
trajectories.

---

## Google Colab Environment

### Recommended runtime

- Runtime: Python 3.
- Hardware accelerator: NVIDIA GPU.
- Practical minimum: T4-class GPU with about 15 GB VRAM.
- Faster option: L4 or A100 when available.
- TPU is not required.
- A high-RAM runtime is not required for the first run.

Record the assigned hardware at the beginning:

```bash
!nvidia-smi
!python --version
```

### VS Code workflow

Use the official Google Colab extension for VS Code:

```text
Open notebooks/video_jepa_colab.ipynb
Select Kernel
Colab
New Colab Server (or Auto Connect)
GPU
```

The notebook clones this fork:

```text
https://github.com/Furkan-Coban/eb_jepa.git
```

The local workspace and Colab VM are different filesystems. Push local changes
to GitHub, then pull or clone them inside Colab.

### Installation

```bash
%cd /content
!git clone https://github.com/Furkan-Coban/eb_jepa.git
%cd /content/eb_jepa
!pip -q install uv
!uv python install 3.12
!uv sync
```

Run project commands through `uv run python ...`.

### Persistence

The Colab VM is temporary. `/content` may be deleted when the runtime ends.

- Train under `/content` for better I/O performance.
- Cache Moving MNIST in Google Drive after its first download.
- Copy configs, metrics, images, and checkpoints to Drive after training.

The notebook uses:

```text
MyDrive/eb_jepa_data/
MyDrive/eb_jepa_artifacts/
```

### Planning-time estimates

These are planning ranges, not measured benchmarks. Hardware assignment,
network speed, batch size, and validation frequency can change them.

| Task | T4 estimate | L4/A100 estimate |
|---|---:|---:|
| Repository and environment setup | 5–15 min | 5–15 min |
| First Moving MNIST download | 2–15 min | 2–15 min |
| Video JEPA subset smoke test | 2–6 min | 1–4 min |
| Video JEPA mini run, 2k–4k clips, 5 epochs | 15–45 min | 8–25 min |
| AC-JEPA mini run, 2k samples, 3 epochs | 10–30 min | 5–20 min |
| Prediction and visual sanity checks | 2–10 min | 2–8 min |

Use the first epoch as the runtime estimate for the remaining epochs. The first
epoch may be slower because of downloading, CUDA warm-up, and caching.

### Is Colab Free sufficient?

Usually yes for these reduced experiments. GPU access, GPU type, and usage quota
are not guaranteed, so checkpoint each epoch and copy final artifacts to Drive.
A paid tier becomes useful only if free GPU access is unavailable, sessions are
repeatedly interrupted, or larger runs and sweeps are required.

---

## Short Research Summary

1. JEPA predicts in representation space rather than generating pixels.
2. State-only Video JEPA predicts future latents from past frames.
3. Representation regularization discourages collapse.
4. Action-conditioned JEPA extends the transition to
   `state + action -> next state`.
5. The decoded video rollout and correct-versus-shuffled action check make the
   mechanisms observable.

Present this work as a bottom-up inspection and small-scale validation of the
JEPA/V-JEPA mechanism, not as reproduction of the original V-JEPA benchmark.

---

## Definition of Done

### Video JEPA

- [ ] Colab GPU and environment verified.
- [ ] Smoke run completed.
- [ ] Mini learning run completed.
- [ ] Metrics and runtimes recorded.
- [ ] Checkpoint copied to Drive.
- [ ] Future-latent prediction executed.
- [ ] Decoded rollout inspected.
- [ ] Important tensor shapes recorded.

### Action-Conditioned JEPA

- [ ] Small training run completed.
- [ ] Checkpoint loaded.
- [ ] `observation + action -> future latent` executed.
- [ ] Correct-action and shuffled-action errors compared.
- [ ] Trajectory comparison inspected.

### Explanation

- [ ] I can explain the state-only and action-conditioned data flows.
- [ ] I can explain why collapse regularizers are needed.
- [ ] I can distinguish this educational example from the original V-JEPA.
