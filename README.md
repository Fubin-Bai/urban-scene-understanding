# Urban Scene Understanding

Pixel-level urban scene analysis with PyTorch and ResNet50 backbones. This project compares semantic segmentation architectures and investigates which road users are hardest to separate as individual instances.

![Semantic segmentation comparison](assets/semantic-comparison.png)

## Local image demo

A working CPU demo is included. No retraining and no Cityscapes dataset download are required for predicting a single image. Supply the S0 checkpoint at `checkpoints/S0/best.pt`; it is already present in this local copy and is excluded from Git.

On this Mac, double-click **启动街景.command**, or run from this folder:

```bash
python3 app.py
```

The browser opens at **http://127.0.0.1:8765** after loading the model. Upload a JPG, PNG, or WebP image and click the prediction button. Switch between original, overlay, and segmentation views; download the overlay, color segmentation, grayscale class-ID mask, or result JSON. The color-strength slider takes effect on the next prediction. Images are processed locally in memory, with no upload to an external service. Press Ctrl+C in the terminal to stop.

If dependencies are missing, create a virtual environment and install only the demo requirements (the existing Mac environment already has them):

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-demo.txt
python3 app.py
```

On macOS, the launcher uses `python3` on PATH. Set `URBAN_SCENE_PYTHON` to a different Python executable if needed. If Finder prevents opening the launcher, use the terminal command above. To use another port: `python3 app.py --port 8766`.

Command-line inference without a browser:

```bash
python3 app.py --image /absolute/path/to/street.jpg --output outputs/my-prediction
```

Outputs are `original.png`, `segmentation.png`, `overlay.png`, `mask.png`, and `result.json`. Class IDs are Cityscapes trainIds 0–18 in the order listed in `segmentation.py`; masks are not raw Cityscapes labelIds. The model uses the original 512×256 resize and ImageNet normalization, then bilinearly resizes logits before class selection. For memory control, uploaded images are limited to 20 MB and 24 megapixels and their long edge is reduced to 1600 pixels when needed; exports match that processed image size. EXIF rotation is respected.

The demo performs semantic segmentation, not instance separation or object counting. Class percentages measure predicted pixels, not accuracy or confidence. CUDA and MPS are not required; inference uses CPU. Recorded validation scores below have not been recomputed by this demo.

## Models and experiments

| ID | Model | Task | Horizontal flip |
|---|---|---|---|
| S0 | FCN-ResNet50 | Semantic segmentation | Yes |
| S1 | DeepLabV3-ResNet50 | Semantic segmentation | Yes |
| S2 | DeepLabV3-ResNet50 | Semantic segmentation | No |
| I0 | Mask R-CNN-ResNet50-FPN | Instance segmentation | Yes |

Semantic models use ImageNet initialization; Mask R-CNN uses COCO initialization with replacement box and mask prediction heads. Training uses AdamW, a learning rate of 1e-4, batch size 2, and 256 × 512 inputs for five epochs.

## Recorded results

| Experiment | Validation mIoU | Pixel accuracy | Instance AP | AP50 |
|---|---:|---:|---:|---:|
| S0 | 0.4899 | 0.9045 | — | — |
| S1 | 0.4522 | 0.8960 | — | — |
| S2 | 0.4852 | 0.9076 | — | — |
| I0 | — | — | 0.1240 | 0.2614 |

These are historical experiment records, not measurements from a fresh run of this repository. Validation data was used for both checkpoint selection and reporting. Experiments use a single seed; no test-server scores are available. See [experiment details](docs/experiments.md).

![Training curves](assets/training-curves.png)

## Run in Google Colab

1. Obtain Cityscapes access and accept its dataset terms at [Cityscapes](https://www.cityscapes-dataset.com/).
2. Download `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` to your own Google Drive folder, by default `MyDrive/cityscapes-archives`.
3. Upload `project.ipynb` to Colab and select a GPU runtime. Execute the setup cells to install dependencies and mount your Drive.
4. Set `ARCHIVE_DIR`, `DATA_ROOT`, and `OUTPUT_ROOT` to your paths. Environment overrides are `CITYSCAPES_ARCHIVE_DIR`, `CITYSCAPES_ROOT`, and `URBAN_SCENE_OUTPUT_ROOT`.
5. Set `PROVENANCE['source_commit']` to the current Git commit. Set `RUN_TRAINING=True` to train, then run the notebook sequentially.
6. To inspect existing runs, use `RUN_TRAINING=False` and supply compatible completed run directories under `OUTPUT_ROOT`.

The default configuration mounts Google Drive. For a local environment, set `MOUNT_DRIVE=False` and configure all three paths before execution; install dependencies from `requirements.txt` and use a compatible PyTorch/TorchVision pair. GPU training is recommended. Dependency versions are not fully locked, so a fresh environment has not been verified against the historical results.

**Data, trained checkpoints, and full evaluation exports are not tracked in Git. This local copy includes the separately downloaded S0 checkpoint.** Without these artifacts, existing training results cannot be regenerated merely by running in read-only mode. Static figures are provided for inspection. Notebook execution outputs are cleared for a clean checkout.

## Repository

- `app.py`, `segmentation.py`, `web/`: standalone local demo and inference.
- `启动街景.command`: macOS launcher.
- `requirements-demo.txt`: minimal demo dependencies.
- `project.ipynb`: data loading, model construction, training, evaluation, and error analysis.
- `assets/`: recorded visual results.
- `docs/experiments.md`: configuration, results, and limitations.
- `requirements.txt`: Python dependencies.

## Methods and data references

Models and pretrained weights come from TorchVision; label definitions and official instance scoring come from cityscapesScripts. Cityscapes data remains subject to its own terms and is not redistributed here.

- Cordts et al., *The Cityscapes Dataset for Semantic Urban Scene Understanding*, CVPR 2016.
- Long et al., *Fully Convolutional Networks for Semantic Segmentation*, CVPR 2015.
- Chen et al., *Rethinking Atrous Convolution for Semantic Image Segmentation*, 2017.
- He et al., *Mask R-CNN*, ICCV 2017.
