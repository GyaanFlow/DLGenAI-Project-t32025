# uploader.py
"""
Uploader module for exporting trained models to KaggleHub.
This script contains simple helper functions that copy model folders
and push them as KaggleHub model versions.
"""

import os
import shutil
import datetime
from pathlib import Path

def upload_folder_kagglehub(local_dir, handle, notes=""):
    """
    Upload an entire folder (e.g., saved HF model) to KaggleHub.
    Only works in Kaggle environment where kagglehub library is available.
    """

    try:
        import kagglehub
    except ImportError:
        print("KaggleHub is not available in this environment.")
        return

    local_dir = Path(local_dir)
    if not local_dir.exists():
        print("Folder does not exist:", local_dir)
        return

    variation = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    full_handle = f"{handle}/{variation}"
    print("Uploading model folder to:", full_handle)

    temp_dir = Path(f"temp_upload_{variation}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Copy model files into a temporary directory
    for item in local_dir.iterdir():
        try:
            if item.is_file():
                shutil.copy(item, temp_dir / item.name)
            else:
                shutil.copytree(item, temp_dir / item.name)
        except Exception:
            pass

    try:
        kagglehub.model_upload(
            handle=full_handle,
            local_model_dir=str(temp_dir),
            version_notes=notes,
        )
        print("Uploaded:", full_handle)
    except Exception as e:
        print("Upload failed:", e)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)



def upload_pt_file(pt_path, handle, notes=""):
    """
    Upload a single .pt file to KaggleHub.
    Useful for scratch model or state_dict.
    """

    try:
        import kagglehub
    except ImportError:
        print("KaggleHub is not available.")
        return

    pt_path = Path(pt_path)
    if not pt_path.exists():
        print("PT file not found:", pt_path)
        return

    variation = pt_path.stem
    full_handle = f"{handle}/{variation}"
    print("Uploading PT file to:", full_handle)

    temp_dir = Path(f"temp_upload_{variation}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(pt_path, temp_dir / pt_path.name)

    try:
        kagglehub.model_upload(
            handle=full_handle,
            local_model_dir=str(temp_dir),
            version_notes=notes,
        )
        print("Uploaded:", full_handle)
    except Exception as e:
        print("Upload failed:", e)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
