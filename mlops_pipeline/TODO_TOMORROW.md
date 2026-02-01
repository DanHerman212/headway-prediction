# TODO: Fix TensorBoard & Visualization (High Priority)

**Context:** The MLOps pipeline runs successfully, but Vertex AI TensorBoard is empty.
**Root Cause:** The pipeline runs in a custom container, so the `gs://` logs written by TensorFlow are **not** automatically synced to the Managed TensorBoard instance. We must implement `tb-gcp-uploader` as a sidecar process.

## 1. Prerequisites (The Sync)
- [ ] **Install `tb-gcp-uploader`**: Ensure `google-cloud-aiplatform[tensorboard]` is installed in `requirements.txt`.
- [ ] **Update `src/train.py`**:
    - Start `tb-gcp-uploader` as a `subprocess.Popen` before training starts.
    - Arguments: `--logdir <GS_PATH> --tensorboard_resource_name <TB_RESOURCE> --one_shot=False`.
    - Ensure it is killed/flushed after training completes.
- [ ] **Update `src/eval.py`**:
    - Same sync logic is needed here if we want Eval metrics/plots to appear in the same dashboard.

## 2. Address Specific TensorBoard Panels

### Histograms & Distributions
- [ ] **Config Check**: Ensure `histogram_freq > 0` in `config.yaml`.
- [ ] **Code**: In `src/train.py`, confirm `keras.callbacks.TensorBoard(..., histogram_freq=cfg.freq)` is passed.
- [ ] **Note**: These only generate if validation data is provided to `model.fit`.

### HParams (Hyperparameters)
- [ ] **Issue**: Keras callback does not log the HParams domain automatically.
- [ ] **Fix (`src/train.py`)**:
    - Import `from tensorboard.plugins.hparams import api as hp`.
    - Define the experiment configuration (HParam objects) and write them to `log_dir`.
    - Log the specific trial values: `hp.hparams(config_dict)`.

### Text
- [ ] **Goal**: Log the Model Architecture and Raw Config.
- [ ] **Fix**:
    - Use `tf.summary.text("Model Summary", ...)` to log `model.to_json()` or string summary.
    - Log the flattened Hydra config as a Markdown table.

### Images
- [ ] **Training**: Set `write_images=True` in the Keras callback (visualizes weights/biases).
- [ ] **Evaluation (`src/eval.py`)**:
    - *Critical:* The `eval_op` calculates MAE but visuals are missing.
    - **Action**: Generate a "Predicted vs Actual" Scatter Plot using Matplotlib.
    - **Action**: Convert plot to PNG buffer -> `tf.image.decode_png` -> `tf.summary.image("Prediction Analysis", ...)`.

### Graphs
- [ ] **Fix**: Should appear automatically if Sync is fixed.
- [ ] **Check**: Ensure `write_graph=True` is in the Keras callback.

### Profile (Performance)
- [ ] **Fix**: Enable the Profiler in `src/train.py`.
- [ ] **Code**: Set `profile_batch='500..520'` in the Keras callback to capture GPU/CPU performance traces during a specific batch range.

---

## 3. Next Step: Phase 1 Strategy
Once the above visuals are working:
- [ ] Create `conf/model/gru.yaml`.
- [ ] Create `conf/model/lstm.yaml`.
- [ ] Refactor `src/model.py` to use a Factory pattern (`build_model` delegating to `build_gru` / `build_lstm`).
