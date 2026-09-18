"""Affine-aware sagittal views. No placeholder image fallback is permitted."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

PREPROCESS_VERSION = "ras-sagittal-physical/1.0"


def resolve_volume(root, volume, index=None):
    root = Path(root).resolve()
    relative = Path(volume)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Volume must be a relative path inside images_root")
    direct = (root / relative).resolve()
    if not direct.is_relative_to(root):
        raise ValueError("Volume escapes images_root")
    if direct.is_file():
        return direct
    candidates = index.get(relative.name, []) if index is not None else list(root.rglob(relative.name))
    candidates = [Path(x).resolve() for x in candidates if Path(x).is_file()]
    if len(candidates) != 1 or not candidates[0].is_relative_to(root):
        raise FileNotFoundError(f"Expected exactly one volume {relative.name}; found {len(candidates)}")
    return candidates[0]


def volume_index(root):
    result = {}
    for path in Path(root).rglob("*"):
        if path.is_file() and (path.name.endswith(".nii.gz") or path.name.endswith(".nii")):
            result.setdefault(path.name, []).append(path)
    return result


def normalize_slice_to_pil(slice_2d, target_size=(384, 384), pixel_spacing=(1.0, 1.0)):
    array = np.asarray(slice_2d)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("MRI slice must be finite and two-dimensional")
    low, high = np.percentile(array, [1, 99])
    if high <= low:
        raise ValueError("Constant/empty MRI slice")
    array = (np.clip((array - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    width, height = array.shape[1] * pixel_spacing[0], array.shape[0] * pixel_spacing[1]
    scale = min(target_size[0] / width, target_size[1] / height)
    shape = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = Image.fromarray(array).resize(shape, Image.Resampling.BILINEAR).convert("RGB")
    return ImageOps.pad(image, target_size, color=(0, 0, 0))


def extract_sagittal_views(path, target_size=(384, 384), offsets=(0,), source_center=None):
    if not offsets or any(isinstance(x, bool) or not isinstance(x, int) for x in offsets):
        raise ValueError("At least one integer sagittal offset is required")
    if len(target_size) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in target_size):
        raise ValueError("Image size must contain two positive integers")
    import nibabel as nib
    from nibabel.processing import resample_to_output
    original = nib.load(str(path))
    if len(original.shape) != 3 or not np.isfinite(original.affine).all():
        raise ValueError("Expected finite-affine 3D NIfTI")
    canonical = nib.as_closest_canonical(original)
    # Permuting axes alone does not remove obliquity.
    if np.max(nib.affines.obliquity(canonical.affine)) > np.deg2rad(1):
        canonical = resample_to_output(canonical, voxel_sizes=canonical.header.get_zooms()[:3], order=1)
    volume = canonical.get_fdata(dtype=np.float32)
    if not np.isfinite(volume).all():
        raise ValueError("MRI contains non-finite intensities")
    if source_center is None:
        center = (volume.shape[0] - 1) / 2
    else:
        point = np.asarray(source_center, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all() or np.any(point < 0) or np.any(point > np.asarray(original.shape) - 1):
            raise ValueError("Source voxel center is outside original NIfTI")
        world = nib.affines.apply_affine(original.affine, point)
        center = nib.affines.apply_affine(np.linalg.inv(canonical.affine), world)[0]
    indices = [int(round(center)) + int(offset) for offset in offsets]
    if len(indices) != len(set(indices)) or any(i < 0 or i >= volume.shape[0] for i in indices):
        raise ValueError("Sagittal offsets duplicate or exceed volume bounds")
    spacing = nib.affines.voxel_sizes(canonical.affine)
    # RAS: superior at top; anterior on left, posterior on right.
    return [normalize_slice_to_pil(volume[i, :, :].T[::-1, ::-1], target_size,
                                  (float(spacing[1]), float(spacing[2]))) for i in indices]


def cached_views(path, cache_dir=None, target_size=(384, 384), offsets=(0,), source_center=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    key = hashlib.sha256(json.dumps([str(path.resolve()), stat.st_size, stat.st_mtime_ns,
        PREPROCESS_VERSION, list(target_size), list(offsets), source_center], sort_keys=True).encode()).hexdigest()
    cached = [Path(cache_dir) / f"{key}_{i}.png" for i in range(len(offsets))] if cache_dir else []
    if cached and all(p.exists() for p in cached):
        result = []
        for p in cached:
            with Image.open(p) as image:
                result.append(image.convert("RGB"))
        return result
    views = extract_sagittal_views(path, target_size, offsets, source_center)
    for image, target in zip(views, cached):
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target)
    return views
