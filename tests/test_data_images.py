import numpy as np
import nibabel as nib
import pytest
from src.data.v2_adapter import DatasetAdapter
from src.data.v1_dataset import SpineVLMDataset
from src.data.mri_images import extract_sagittal_views, resolve_volume, cached_views
from src.io_utils import file_hash


def test_legacy_zero_restored_without_source_write(master_rows, write_master):
    legacy = {"patient_id": "synthetic-case", "levels": [{"level": r["level"], "gradings": {
        k: r[k] for k in ("pfirrmann_grade", "modic", "disc_herniation", "disc_bulging", "disc_narrowing",
                          "spondylolisthesis", "up_endplate", "low_endplate")}} for r in master_rows]}
    master_rows[3]["disc_bulging"] = ""
    path = write_master(master_rows)
    before = file_hash(path)
    request, corrections = DatasetAdapter(path).convert(legacy)
    assert request.levels[3].gradings.disc_bulging.value is None and len(corrections) == 1
    assert file_hash(path) == before
    legacy["levels"][0]["gradings"]["modic"] = 2
    with pytest.raises(ValueError): DatasetAdapter(path).convert(legacy)


def test_dataset_fails_missing_images_and_inconsistent_split(master_rows, write_master, tmp_path):
    path = write_master(master_rows)
    assert len(SpineVLMDataset(path, tmp_path, require_images=False)) == 1
    with pytest.raises(FileNotFoundError): SpineVLMDataset(path, tmp_path)
    master_rows[2]["fold1_split"] = "test"
    with pytest.raises(ValueError): SpineVLMDataset(write_master(master_rows), tmp_path, require_images=False)


def test_missing_targets_and_conflicting_alias(master_rows, write_master, tmp_path):
    for row in master_rows: row["report_vi_findings"] = "nan"
    with pytest.raises(ValueError): SpineVLMDataset(write_master(master_rows), tmp_path, require_images=False)
    for row in master_rows:
        row["report_vi_findings"] = "canonical"
        row["report_vi_mota"] = "different"
    with pytest.raises(ValueError): SpineVLMDataset(write_master(master_rows), tmp_path, require_images=False)


def test_affine_reorientation_preserves_anatomical_view(tmp_path):
    data = np.arange(7 * 9 * 11, dtype=np.float32).reshape(7, 9, 11)
    image = nib.Nifti1Image(data, np.diag([2., 3., 4., 1.]))
    source = tmp_path / "ras.nii.gz"
    rotated = tmp_path / "permuted.nii.gz"
    nib.save(image, source)
    nib.save(image.as_reoriented(np.array([[2, -1], [0, 1], [1, -1]])), rotated)
    first = np.asarray(extract_sagittal_views(source)[0])
    assert np.array_equal(first, np.asarray(extract_sagittal_views(rotated)[0]))
    assert np.array_equal(first, np.asarray(extract_sagittal_views(source, source_center=[3, 4, 5])[0]))
    with pytest.raises(ValueError): extract_sagittal_views(source, source_center=[-1, 4, 5])
    with pytest.raises(ValueError): extract_sagittal_views(source, offsets=[100])
    assert np.array_equal(first, np.asarray(cached_views(source, tmp_path / "cache")[0]))
    assert np.array_equal(first, np.asarray(cached_views(source, tmp_path / "cache")[0]))


def test_nested_resolution_and_ambiguity(tmp_path):
    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "fake.nii.gz").touch()
    assert resolve_volume(tmp_path, "a/fake.nii.gz").parent.name == "a"
    with pytest.raises(FileNotFoundError): resolve_volume(tmp_path, "fake.nii.gz")
    with pytest.raises(ValueError): resolve_volume(tmp_path, "../fake.nii.gz")
