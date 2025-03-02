import os
import json
import glob
import io
import pandas as pd
from typing import Optional, List, Dict, Union, Any
from datasets import Dataset, DatasetDict, Features, Image as DSImage
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
    Adds an 'image_path' column. If include_images=True, also loads the
    image bytes into a dictionary {'bytes': ...} under 'image'.
    
    NOTE: We now keep this "bytes" approach only for local usage
    (e.g. if you're eventually exporting to a Parquet or just storing the DF).
    For Hugging Face, we will ignore these bytes and do the HF-based approach 
    in export_huggingface().
    """
    # Always add an image_path column
    df['image_path'] = df.apply(
        lambda row: get_image_path_from_metadata_path(
            row["metadata_json_path"],
            row["filename"]
        ),
        axis=1
    )

    if not include_images:
        return df

    # Otherwise, also load the actual image data as bytes in 'image'
    image_data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading images"):
        if row["image_path"] is not None:
            img_bytes = load_image(row["image_path"])
            if img_bytes is not None:
                image_data.append({'image': {'bytes': img_bytes}})
            else:
                image_data.append({'image': None})
        else:
            image_data.append({'image': None})

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
        include_images: If True, loads the actual images from disk as bytes (for local usage).
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
    if "metadata_json_path" in df_final.columns:
        df_final.drop(columns=["metadata_json_path"], inplace=True)

    return df_final

def export_parquet(df: pd.DataFrame, output_path: str):
    """
    Exports a DataFrame to a Parquet file.
    NOTE: If 'image' column exists, it will be dropped because
    storing arbitrary binary data in Parquet can be problematic.
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
    
    If 'image_path' exists, we convert it to a Hugging Face 'image' column
    using {"path": ...}, then push the result as a DatasetDict with
    an 85/15 train/validation split.
    """
    from datasets import Dataset, DatasetDict, Features, Value, Image

    # We only need the relevant columns for HF.
    # In particular, we want to rename 'image_path' -> 'image', with {'path': x}.
    hf_df = df.copy()

    # If we previously added bytes in 'image', remove them because for HF
    # we want the recommended approach (using paths).
    if "image" in hf_df.columns:
        hf_df.drop(columns=["image"], inplace=True, errors="ignore")

    # If there's no 'image_path', we can just save a tabular dataset. 
    if "image_path" not in hf_df.columns:
        print("No 'image_path' found. Uploading DataFrame as-is to HF.")
        # Just push tabular data
        # (You can adjust this logic to do something else if needed.)
        ub.saves(hf_df, f"hf://{repo_id}", private=private)
        print(f"Exported dataset to Hugging Face: {repo_id}")
        return

    # Convert the 'image_path' column to the HF-compatible dict
    hf_df["image"] = hf_df["image_path"].apply(lambda x: {"path": x} if pd.notnull(x) else None)
    hf_df.drop(columns=["image_path"], inplace=True)

    # Build features. For columns besides 'image', treat them as strings or numeric as feasible.
    # Here we do a simple approach: if dtypes is object, use Value("string"); else use the numeric type.
    features_dict = {}
    for col in hf_df.columns:
        if col == "image":
            features_dict[col] = Image()
        else:
            # A simple guess: if numeric, we'll store as float or int
            if pd.api.types.is_integer_dtype(hf_df[col]):
                features_dict[col] = Value("int64")
            elif pd.api.types.is_float_dtype(hf_df[col]):
                features_dict[col] = Value("float64")
            else:
                features_dict[col] = Value("string")

    features = Features(features_dict)

    hf_dataset = Dataset.from_pandas(hf_df, features=features)
    total_len = len(hf_dataset)

    if total_len == 0:
        print("Warning: No rows in dataset. Nothing to push.")
        return

    # 85/15 split
    train_size = int(0.85 * total_len)
    hf_train = hf_dataset.select(range(train_size))
    hf_val = hf_dataset.select(range(train_size, total_len))

    dataset_dict = DatasetDict({
        "train": hf_train,
        "validation": hf_val
    })

    # Now push to HF using the standard HF push method
    dataset_dict.push_to_hub(repo_id, private=private)
    print(f"Exported dataset with images to Hugging Face: {repo_id}")