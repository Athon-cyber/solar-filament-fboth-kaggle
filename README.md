# Solar Filament Segmentation Challenge 2026

This repository packages the OPT-030 F_BOTH epoch-80 Mask2Former Swin-S model and the OPT-031 three-view inference recipe used for Kaggle submission **57372264**. The archived CSV contains **1,112** instances generated from **180** competition test images. The displayed public score was **0.39**.

| Material | Path |
|---|---|
| Four-page technical report draft | `output/pdf/solar-filament-fboth-report.pdf` |
| LaTeX sources and figures | `report/` |
| Inference and CSV verification walkthrough | `notebooks/reproduce.ipynb` |
| Model-only checkpoint | `weights/fboth-epoch80-model.pth` (Git LFS) |
| Archived submitted CSV | `evidence/submission-opt031-c090-p040.csv` |
| Digests, configuration and submission ID | `evidence/manifest.json` |

## Environment

The original inference used Linux x86_64, Python 3.9, PyTorch 2.1.0 with CUDA 12.1, and a CUDA GPU. Install the Python requirements before building the CUDA extensions. A matching CUDA toolkit, a compatible C++ toolchain, and Git LFS are required. Keep the Mask2Former license included with its staged source.

```bash
git lfs pull
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation 'detectron2 @ git+https://github.com/facebookresearch/detectron2.git@1e3e13bbf607b54f62205c4c33922521822fb298'
(cd vendor/mask2former/mask2former/modeling/pixel_decoder/ops && python setup.py build install)
```

Detectron2 is pinned to the commit embedded in the historical source archive. The Mask2Former directory contains the historical staged architecture, config and extension sources, based on upstream archive commit `9b0651c6c1d5b3af2e6da0589b719c514ec0d69a`. This build procedure has yet to be verified end to end in a clean CUDA environment.

## Reproduce the submission

Accept the [competition rules and obtain the official input data](https://www.kaggle.com/competitions/filament-segmentation-2026/data). Keep the 180 unlabelled competition images in their original filenames. Use the CLI or open the Notebook from the repository root.

```bash
python reproduce.py --test-images /path/to/MAGFiLO_1.0_Kaggle_2026/test/test_images \
  --output predictions/submission.csv
```

The entry point loads the exact model state without training or initializing from a second checkpoint, runs identity/horizontal/vertical views, fuses predictions, and checks the resulting CSV. An exact byte mismatch against the historical submission produces a nonzero exit status.

To independently verify the archived CSV without model execution or image access:

```bash
python reproduce.py --validate-only evidence/submission-opt031-c090-p040.csv
```

## Evaluation scope

The final `0.90/0.40` configuration scored notebook PQ 0.459301 and strict micro-PQ 0.459102 on 148 development photos with 372 independent references. The 148-photo internal holdout PQ 0.446006 belongs to the earlier `0.90/0.50` configuration. The report records that distinction.

## Review before release

The report draft uses the Kaggle/GitHub handle as temporary author text. Confirm the real author list, affiliations and contact details. Review permission to redistribute the model, report illustrations and repository contents before switching visibility. The final review also needs a clean CUDA run that rebuilds the submitted CSV and access checks for the public repository and weights.
