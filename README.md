# Urban Scene Understanding

Pixel-level urban scene analysis with PyTorch and ResNet50 backbones. This project compares semantic segmentation architectures and investigates which road users are hardest to separate as individual instances.

![Semantic segmentation comparison](assets/semantic-comparison.png)

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

**Data, trained checkpoints, and full evaluation exports are not included.** Without these artifacts, existing training results cannot be regenerated merely by running in read-only mode. Static figures are provided for inspection. Notebook execution outputs are cleared for a clean checkout.

## Repository

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
