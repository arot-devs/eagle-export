import os
import json
import glob
import io
import pandas as pd
from typing import Optional, List, Dict, Union, Any
from datasets import Dataset, Features, Image as DSImage
from PIL import Image as PILImage
from tqdm import tqdm
import unibox as ub

def load_eagle_jsons(eagle_img_dir: str) -> List[tuple]:
    """
    Scans eagle_img_dir and its subdirectories for .json files
    and loads each into a Python dict, returning a list of tuples:
    
    [
      ( "/path/to/.../metadata.json", {json_data} ),
      ( "/path/to/.../metadata.json", {json_data} ),
      ...
    ]
    """
    # Collect all "metadata.json" paths recursively
    json_files = ub.traverses(eagle_img_dir, ["metadata.json"])
    json_contents = ub.concurrent_loads(json_files)
    assert len(json_files) == len(json_contents), "Mismatched JSON file count"
    
    # Pair each file path with its loaded JSON content
    return [(path, content) for path, content in zip(json_files, json_contents)]

def preprocess_dict(data: dict) -> dict:
    """
    Cleans a single Eagle metadata dict, extracting relevant fields.
    Also picks the top palette color by ratio (if present).
    """
    def rgb_to_hex(color):
        return "#{:02x}{:02x}{:02x}".format(*color)

    # Copy all except 'palettes'
    base_info = {k: v for k, v in data.items() if k != "palettes"}

    # If palettes exist, pick the one with highest ratio
    palettes = data.get("palettes", [])
    if palettes:
        top_palette = max(palettes, key=lambda p: p["ratio"])
        base_info["palette_color"] = rgb_to_hex(top_palette["color"])
        base_info["palette_ratio"] = top_palette["ratio"]
    else:
        base_info["palette_color"] = None
        base_info["palette_ratio"] = None

    return base_info

def eagle_jsons_to_df(eagle_jsons: List[tuple]) -> pd.DataFrame:
    """
    Processes the list of (metadata_json_path, metadata_dict) tuples into
    a cleaned pandas DataFrame. Adds `filename` = name + ext, then drops
    unwanted columns.
    """
    rows = []
    for json_path, content in eagle_jsons:
        row = preprocess_dict(content)
        # Keep track of where this metadata was loaded from
        row["metadata_json_path"] = json_path
        rows.append(row)

    df = pd.DataFrame(rows)

    # Add filename
    if "name" in df.columns and "ext" in df.columns:
        df["filename"] = df["name"] + "." + df["ext"]
    else:
        # Fallback, if missing 'name' or 'ext'
        df["filename"] = df.get("id", pd.Series(range(len(df)))).astype(str)

    # Drop some known unwanted columns
    unwanted_cols = [
        "id", "btime", "mtime", "modificationTime", "lastModified",
        "noThumbnail", "deletedTime", "name", "ext"
    ]
    for col in unwanted_cols:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Reorder columns for convenience
    new_cols = ["filename"] + [c for c in df.columns if c != "filename"]
    df = df[new_cols]

    return df

def parse_s5cmd_file(s5cmd_file: str) -> pd.DataFrame:
    """
    Parses an s5cmd file to extract lines like:
      cp local/path/filename s3://bucket/path/filename
    Returns a DataFrame with columns [filename, s3_uri].
    """
    lines = []
    with open(s5cmd_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "cp":
                local_path = parts[1]
                s3_path = parts[2]
                fname = os.path.basename(local_path)
                lines.append((fname, s3_path))

    df_s5 = pd.DataFrame(lines, columns=["filename", "s3_uri"])
    return df_s5

def add_s3_uri_col(df: pd.DataFrame, s5cmd_file: Optional[str]) -> pd.DataFrame:
    """
    If s5cmd_file is provided, merges the Eagle DataFrame
    with a DataFrame of (filename, s3_uri).
    """
    if not s5cmd_file or not os.path.exists(s5cmd_file):
        return df

    df_s5 = parse_s5cmd_file(s5cmd_file)
    merged_df = df.merge(df_s5, on="filename", how="left")
    return merged_df

def get_image_path_from_metadata_path(json_path: str, filename: str) -> Optional[str]:
    """
    Determines the image's full path by placing `filename` next to `metadata.json`.
    If exact match not found, tries case-insensitive search.
    """
    base_dir = os.path.dirname(json_path)
    candidate = os.path.join(base_dir, filename)

    if os.path.exists(candidate):
        return candidate

    # Otherwise, try case-insensitive search in the directory
    try:
        all_files = os.listdir(base_dir)
        for f in all_files:
            if f.lower() == filename.lower():
                return os.path.join(base_dir, f)
    except Exception as e:
        print(f"Could not list directory {base_dir}. Error: {e}")

    return None

def load_image(image_path: str) -> Optional[bytes]:
    """
    Loads an image file and returns its bytes, or None on failure.
    """
    try:
        with open(image_path, 'rb') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def add_images(df: pd.DataFrame, include_images: bool = False) -> pd.DataFrame:
    """
    Adds image paths and optionally loads the actual images as raw bytes.
    Uses the 'metadata_json_path' column to locate the image next to each metadata.json.
    """
    if not include_images:
        df['image_path'] = df.apply(
            lambda row: get_image_path_from_metadata_path(
                row["metadata_json_path"],
                row["filename"]
            ),
            axis=1
        )
        return df
    
    # Otherwise, also load the actual image data
    image_data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading images"):
        image_path = get_image_path_from_metadata_path(row["metadata_json_path"], row["filename"])
        if image_path:
            img_bytes = load_image(image_path)
            hf_img_bytes = {
                'bytes': img_bytes
            }
            image_data.append({
                'image_path': image_path,
                'image': hf_img_bytes
            })
        else:
            image_data.append({
                'image_path': None,
                'image': None
            })

    image_df = pd.DataFrame(image_data)
    return pd.concat([df.reset_index(drop=True), image_df.reset_index(drop=True)], axis=1)

def build_dataframe(eagle_dir: str,
                    s5cmd_file: Optional[str] = None,
                    include_images: bool = False
                    ) -> pd.DataFrame:
    """
    Main function to build the final metadata DataFrame from an Eagle library path.
    
    Args:
        eagle_dir: Path to Eagle library directory (folder that contains images/ subdir)
        s5cmd_file: Optional path to an s5cmd file for injecting S3 URIs
        include_images: If True, loads the actual images from disk as bytes.
    """
    eagle_img_dir = os.path.join(eagle_dir, "images")
    # Load all (path, metadata) pairs
    eagle_json_tuples = load_eagle_jsons(eagle_img_dir)
    
    # Convert to DataFrame
    df_cleaned = eagle_jsons_to_df(eagle_json_tuples)
    
    # Merge S3 URIs if provided
    df_with_s3 = add_s3_uri_col(df_cleaned, s5cmd_file)

    # Attach image paths (and optional image bytes)
    df_final = add_images(df_with_s3, include_images=include_images)

    # cleanup: remove metadata_json_path if not needed    
    for col in ["metadata_json_path", "image_path"]:
        if col in df_final.columns:
            df_final.drop(columns=col, inplace=True)

    return df_final

def export_parquet(df: pd.DataFrame, output_path: str):
    """
    Exports a DataFrame to a Parquet file.
    NOTE: If 'image' column exists, it will be dropped because binary data
    doesn't fit well into Parquet.
    """
    export_df = df.copy()
    if 'image' in export_df.columns:
        export_df.drop(columns=['image'], inplace=True)
        print("Note: Image binary data was removed for Parquet export.")
    export_df.to_parquet(output_path, index=False)
    print(f"Saved parquet to: {output_path}")


def export_huggingface(df: pd.DataFrame, repo_id: str, private: bool = False):
    """
    Exports a DataFrame to a Hugging Face dataset (push_to_hub).
    If 'image' column exists, convert it to the dictionary format
    that 'datasets.Image' expects, i.e. {"bytes": b"..."}.
    """    
    # Use whichever approach you prefer for pushing:
    import unibox as ub
    ub.saves(df, f"hf://{repo_id}", private=private)