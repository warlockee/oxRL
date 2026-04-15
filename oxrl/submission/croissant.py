"""
Croissant metadata generation and validation for HuggingFace datasets.

Fetches auto-generated Croissant JSON from HuggingFace, injects the NeurIPS-
required RAI fields, and validates the result with mlcroissant.
"""
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import requests


# RAI fields required by NeurIPS Evaluations and Datasets Hosting Guidelines.
REQUIRED_RAI_FIELDS = [
    "dataBiases",
    "dataCollection",
    "personalSensitiveInformation",
    "isLiveDataset",
]

# Recommended fields that suppress validator warnings.
RECOMMENDED_FIELDS = ["datePublished", "version", "license", "citeAs"]


def fetch_croissant(dataset_id: str, hf_token: Optional[str] = None) -> Dict:
    """Fetch auto-generated Croissant JSON-LD from HuggingFace."""
    url = f"https://huggingface.co/api/datasets/{dataset_id}/croissant"
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def inject_rai_fields(
    croissant: Dict,
    data_biases: str,
    data_collection: str,
    personal_sensitive_info: str,
    is_live_dataset: bool = False,
    date_published: Optional[str] = None,
    version: Optional[str] = None,
    license_url: Optional[str] = None,
    cite_as: Optional[str] = None,
) -> Dict:
    """Add NeurIPS-required RAI fields and recommended metadata."""
    croissant["dataBiases"] = data_biases
    croissant["dataCollection"] = data_collection
    croissant["personalSensitiveInformation"] = personal_sensitive_info
    croissant["isLiveDataset"] = is_live_dataset

    if date_published:
        croissant["datePublished"] = date_published
    if version:
        croissant["version"] = version
    if license_url:
        croissant["license"] = license_url
    if cite_as:
        croissant["citeAs"] = cite_as

    # Ensure the RAI context key exists.
    if "@context" in croissant and isinstance(croissant["@context"], dict):
        croissant["@context"]["rai"] = "http://mlcommons.org/croissant/RAI/"

    return croissant


def generate_croissant(
    dataset_id: str,
    output_path: str,
    hf_token: Optional[str] = None,
    data_biases: str = "",
    data_collection: str = "",
    personal_sensitive_info: str = "No personal or sensitive information.",
    is_live_dataset: bool = False,
    date_published: Optional[str] = None,
    version: Optional[str] = None,
    license_url: Optional[str] = None,
    cite_as: Optional[str] = None,
) -> Path:
    """Fetch Croissant from HF, inject RAI fields, write to disk.

    Returns the output path.
    """
    croissant = fetch_croissant(dataset_id, hf_token)
    croissant = inject_rai_fields(
        croissant,
        data_biases=data_biases,
        data_collection=data_collection,
        personal_sensitive_info=personal_sensitive_info,
        is_live_dataset=is_live_dataset,
        date_published=date_published,
        version=version,
        license_url=license_url,
        cite_as=cite_as,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(croissant, indent=2))
    return out


def validate_croissant(path: str) -> Dict:
    """Validate a Croissant JSON-LD file.

    Returns a dict with keys: valid (bool), errors (list), warnings (list).
    Uses mlcroissant if importable, otherwise falls back to schema checks.
    """
    result = {"valid": False, "errors": [], "warnings": []}
    path = Path(path)

    if not path.exists():
        result["errors"].append(f"File not found: {path}")
        return result

    data = json.loads(path.read_text())

    # Check required RAI fields.
    for field in REQUIRED_RAI_FIELDS:
        if field not in data or data[field] == "":
            result["errors"].append(f"Missing RAI field: {field}")

    # Check recommended fields.
    for field in RECOMMENDED_FIELDS:
        if field not in data:
            result["warnings"].append(f"Missing recommended field: {field}")

    # Try mlcroissant validator.
    try:
        from mlcroissant import Dataset

        Dataset(jsonld=str(path))
    except ImportError:
        result["warnings"].append(
            "mlcroissant not installed — skipped deep validation. "
            "Install with: pip install mlcroissant"
        )
    except TypeError:
        # Python 3.9 is incompatible with mlcroissant >=1.0 (uses X | Y syntax).
        result["warnings"].append(
            "mlcroissant requires Python >=3.10. RAI field checks passed; "
            "run with Python 3.10+ for full schema validation."
        )
    except Exception as e:
        result["errors"].append(f"mlcroissant validation failed: {e}")

    result["valid"] = len(result["errors"]) == 0

    return result


def check_rai_fields(path: str) -> Dict:
    """Quick check: which RAI fields are present/missing."""
    data = json.loads(Path(path).read_text())
    status = {}
    for field in REQUIRED_RAI_FIELDS:
        val = data.get(field)
        if val is None or val == "":
            status[field] = "MISSING"
        else:
            status[field] = "OK"
    return status
