"""OPT-031 candidate extraction and CSV checks adapted from the archived run."""
import csv
import hashlib
from pathlib import Path

def calibrate_photo(bridge, image, torch, configs, associate, vote, mask_nms, exclusive_soft_groups):
    """One model pass per view; threshold-specific candidate crops are reused across configs."""
    import cv2
    import torch.nn.functional as F
    from detectron2.structures import ImageList
    SIZE = 2048
    candidates = {key: [] for key in configs}
    confidences = sorted({key[0] for key in configs})
    pixels = sorted({key[1] for key in configs})
    n_raw = 0
    np = __import__("numpy")
    with torch.no_grad():
        for view in VIEWS:
            if view == "identity":
                view_image = image
            elif view == "hflip":
                view_image = image[:, ::-1].copy()
            else:
                view_image = image[::-1].copy()
            net = (view_image if view_image.shape[:2] == (SIZE, SIZE) else
                   cv2.resize(view_image, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR))
            tensor = torch.from_numpy(np.ascontiguousarray(net)).permute(2, 0, 1).float().to(bridge.device)
            tensor = (tensor - bridge.model.pixel_mean) / bridge.model.pixel_std
            inputs = ImageList.from_tensors([tensor], bridge.model.size_divisibility)
            output = bridge.model.sem_seg_head(bridge.model.backbone(inputs.tensor))
            scores = output["pred_logits"][0].softmax(-1)[:, 0]
            score_values = scores.detach().float().cpu().numpy()
            selected = np.flatnonzero(score_values >= min(confidences)).tolist()
            n_raw += len(selected)
            per_pixel = {p: [] for p in pixels}
            for index in selected:
                class_score = float(score_values[index])
                full_prob = F.interpolate(output["pred_masks"][0, index][None, None].float(),
                                          size=(2048, 2048), mode="bilinear", align_corners=False)[0, 0].sigmoid()
                prob = full_prob.float().cpu().numpy()
                for pixel in pixels:
                    foreground = prob >= pixel
                    if not foreground.any():
                        continue
                    yy, xx = np.nonzero(foreground)
                    a, b = int(xx.min()), int(yy.min())
                    c, d = int(xx.max()) + 1, int(yy.max()) + 1
                    mean_probability = float(full_prob[full_prob >= pixel].mean())
                    crop = prob[b:d, a:c].astype("float32", copy=True)
                    base_row = dict(box=[a, b, c, d], prob=crop, source=view, view=view)
                    if view == "hflip":
                        base_row["box"] = [2048 - c, b, 2048 - a, d]
                        base_row["prob"] = crop[:, ::-1].copy()
                    elif view == "vflip":
                        base_row["box"] = [a, 2048 - d, c, 2048 - b]
                        base_row["prob"] = crop[::-1].copy()
                    per_pixel[pixel].append((class_score * mean_probability, class_score, base_row))
            del output
            for pixel, entries in per_pixel.items():
                for composite_score, class_score, row in entries:
                    for confidence in confidences:
                        key = (confidence, pixel)
                        if key not in candidates:
                            continue
                        if class_score < confidence or composite_score < confidence:
                            continue
                        candidates[key].append(dict(row, score=composite_score))
    output_by_key = {}
    diagnostics = {}
    for key, rows in candidates.items():
        confidence, pixel = key
        groups, ambiguous = associate(rows, 0.5, 0.1, pixel)
        fused = vote(groups, total_views=3, min_support=2, threshold=pixel)
        fused = mask_nms(fused, 0.8, 100)
        outputs = exclusive_soft_groups(fused, pixel)
        output_by_key[key] = [dict(confidence=x.get("confidence", x.get("score", 0.0)), rle=x["rle"])
                              for x in outputs]
        diagnostics[f"{confidence:.2f}_{pixel:.2f}"] = dict(raw_candidates=len(rows),
                                                            groups=len(groups), ambiguous=ambiguous,
                                                            predictions=len(outputs))
    return output_by_key, diagnostics, n_raw


def validate_csv(path, official_names):
    import numpy as np
    import pycocotools.mask as cm
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    stems = {Path(n).stem for n in official_names}
    seen = {stem: [] for stem in stems}
    ids = set()
    for row in rows:
        fid = row["filament_id"]
        stem, sep, serial = fid.rpartition("_")
        if not sep or stem not in stems or fid in ids:
            raise ValueError(f"bad/duplicate id {fid}")
        ids.add(fid); seen[stem].append((int(serial), row["segmentation_rle"]))
    if set(seen) != stems:
        raise ValueError("official coverage mismatch")
    for stem, items in seen.items():
        items.sort()
        if [i for i, _ in items] != list(range(len(items))):
            raise ValueError(f"non-contiguous ids {stem}")
        occupied = np.zeros((2048, 2048), dtype=bool)
        for _, counts in items:
            rle = dict(size=[2048, 2048], counts=counts.encode("ascii"))
            mask = cm.decode(rle).astype(bool)
            if mask.shape != (2048, 2048) or not mask.any() or (mask & occupied).any():
                raise ValueError(f"empty/overlap RLE {stem}")
            occupied |= mask
            if cm.encode(np.asfortranarray(mask.astype("uint8")))["counts"] != rle["counts"]:
                raise ValueError(f"RLE roundtrip failed {stem}")
    return dict(photos=len(stems), rows=len(rows), empty_photos=sum(not x for x in seen.values()),
                unique_ids=True, contiguous_ids=True, zero_overlap=True, rle_roundtrip=True,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
