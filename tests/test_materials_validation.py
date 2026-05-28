"""Unit tests for the materials JSON validation function."""

import json
from pathlib import Path
import pytest

from coatopt.utils.utils import validate_materials


def test_validate_materials_valid(tmp_path):
    """Test validation with a perfectly valid materials JSON."""
    python_data = {
        "0": {
            "name": "air",
            "desc": "Air",
            "n": 1,
            "a": None,
            "alpha": None,
            "beta": None,
            "kappa": None,
            "C": None,
            "Y": None,
            "prat": None,
            "phiM": None,
            "k": 0
        },
        "1": {
            "name": "SiO2",
            "desc": "Silica",
            "n": 1.45,
            "a": 0,
            "alpha": 5.1e-7,
            "beta": 8e-6,
            "kappa": 1.38,
            "C": 1.6412e6,
            "Y": 70e9,
            "prat": 0.19,
            "phiM": 2.3e-5,
            "k": 0
        },
        "2": {
            "name": "TiGermania",
            "desc": "Titania doped Germania",
            "n": 1.866,
            "a": 1,
            "alpha": 12.82e-7,
            "beta": 2.4e-5,
            "kappa": 33,
            "C": 2.51e6,
            "Y": 92e9,
            "prat": 0.29,
            "phiM": 9.013672e-05,
            "k": 2e-7
        }
    }
    filepath = tmp_path / "valid_materials.json"
    with open(filepath, "w") as f:
        json.dump(python_data, f, indent=2)

    # Should not raise any error
    validate_materials(str(filepath))


def test_validate_materials_missing_file():
    """Test validation with a non-existent file."""
    with pytest.raises(FileNotFoundError):
        validate_materials("non_existent_file.json")


def test_validate_materials_invalid_json(tmp_path):
    """Test validation with a syntactically invalid JSON file."""
    filepath = tmp_path / "invalid.json"
    with open(filepath, "w") as f:
        f.write("{ invalid json }")

    with pytest.raises(ValueError, match="Invalid JSON format"):
        validate_materials(str(filepath))


def test_validate_materials_non_dict_json(tmp_path):
    """Test validation with a JSON that is not an object/dictionary."""
    filepath = tmp_path / "array.json"
    with open(filepath, "w") as f:
        json.dump([1, 2, 3], f)

    with pytest.raises(ValueError, match="must represent a JSON object"):
        validate_materials(str(filepath))


def test_validate_materials_non_sequential_keys(tmp_path):
    """Test validation when keys are out of order or have gaps."""
    # Gap in keys (0, 2)
    filepath = tmp_path / "gap.json"
    with open(filepath, "w") as f:
        json.dump({"0": {"name": "air"}, "2": {"name": "SiO2"}}, f)

    with pytest.raises(ValueError, match="Materials keys must be consecutive"):
        validate_materials(str(filepath))

    # Out of order keys (1, 0)
    filepath = tmp_path / "out_of_order.json"
    with open(filepath, "w") as f:
        json.dump({"1": {"name": "SiO2"}, "0": {"name": "air"}}, f)

    with pytest.raises(ValueError, match="Materials keys must be consecutive"):
        validate_materials(str(filepath))


def test_validate_materials_air_missing_or_invalid(tmp_path):
    """Test validation when material "0" is not air."""
    # Key "0" is completely missing
    filepath = tmp_path / "no_zero.json"
    with open(filepath, "w") as f:
        json.dump({"1": {"name": "SiO2"}}, f)

    with pytest.raises(ValueError, match="Materials keys must be consecutive"):
        validate_materials(str(filepath))

    # Key "0" is present but named "SiO2"
    filepath = tmp_path / "zero_not_air.json"
    with open(filepath, "w") as f:
        json.dump({"0": {"name": "SiO2"}}, f)

    with pytest.raises(ValueError, match="Material '0' must represent 'air'"):
        validate_materials(str(filepath))


def test_validate_materials_null_in_non_air(tmp_path):
    """Test validation when a non-air material has a null property."""
    data = {
        "0": {
            "name": "air",
            "n": 1,
            "alpha": None
        },
        "1": {
            "name": "SiO2",
            "n": 1.45,
            "alpha": None  # Null value here should trigger validation error
        }
    }
    filepath = tmp_path / "null_prop.json"
    with open(filepath, "w") as f:
        json.dump(data, f)

    with pytest.raises(ValueError, match="Null value found in non-air material '1'"):
        validate_materials(str(filepath))
