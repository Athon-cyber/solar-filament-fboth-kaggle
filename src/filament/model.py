"""Construct the frozen one-class Mask2Former without COCO initialization."""
from pathlib import Path
from types import SimpleNamespace
import sys


def load_model(checkpoint: Path, vendor: Path):
    import torch
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    from detectron2.modeling import build_model
    from detectron2.projects.deeplab import add_deeplab_config

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA and the matching Detectron2/MSDeformAttn builds are required")
    sys.path.insert(0, str(vendor))
    from mask2former import add_maskformer2_config

    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(str(vendor / "configs/coco/instance-segmentation/swin/maskformer2_swin_small_bs16_50ep.yaml"))
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    dataset_name = "filament_fboth"
    MetadataCatalog.get(dataset_name).set(thing_classes=["filament"], thing_dataset_id_to_contiguous_id={1: 0})
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = (dataset_name,)
    cfg.INPUT.MIN_SIZE_TRAIN = (2048,)
    cfg.INPUT.MAX_SIZE_TRAIN = 2048
    cfg.INPUT.MIN_SIZE_TEST = 2048
    cfg.INPUT.MAX_SIZE_TEST = 2048
    cfg.MODEL.SWIN.USE_CHECKPOINT = True
    cfg.freeze()

    model = build_model(cfg)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    device = torch.device("cuda:0")
    return SimpleNamespace(model=model.to(device).eval(), device=device)
