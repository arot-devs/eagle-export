import os
import json
import glob
import pandas as pd
from typing import Optional, List
from datasets import Dataset

def load_eagle_jsons(eagle_img_dir: str) -> List[dict]:
    """
    Scans eagle_img_dir for .json files and loads each into a Python dict.
    """
    pattern = os.path.join(eagle_img_dir, "*.json")
    json_files = glob.glob(pattern)
    results = []

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
            results.append(data)

    return results

def preprocess_dict(data: dict) -> dict:
    """
    Cleans a single dictionary, extracting relevant fields and picking the
    top palette color by ratio (if present).
    """
    def rgb_to_hex(color):
        # color is [R, G, B]
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

def eagle_jsons_to_df(eagle_jsons: List[dict]) -> pd.DataFrame:
    """
    Processes a list of Eagle JSON dictionaries into a cleaned pandas DataFrame.
    Adds `filename` as a new column from name + ext, then drops unwanted columns.
    """
    rows = [preprocess_dict(d) for d in eagle_jsons]
    df = pd.DataFrame(rows)

    # Add filename
    if "name" in df.columns and "ext" in df.columns:
        df["filename"] = df["name"] + "." + df["ext"]
    else:
        # Fallback, not typical if Eagle data is missing 'name' or 'ext'
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
            # Attempt naive parse: 'cp local s3://....'
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "cp":
                # parts[1] => local path, parts[2] => s3 path
                local_path = parts[1]
                s3_path = parts[2]
                fname = os.path.basename(local_path)
                lines.append((fname, s3_path))

    df_s5 = pd.DataFrame(lines, columns=["filename", "s3_uri"])
    return df_s5

def add_s3_uri_col(df: pd.DataFrame, s5cmd_file: Optional[str]) -> pd.DataFrame:
    """
    If s5cmd_file is provided, merges the eagle DataFrame
    with a second DataFrame that has (filename, s3_uri).
    """
    if not s5cmd_file or not os.path.exists(s5cmd_file):
        return df

    df_s5 = parse_s5cmd_file(s5cmd_file)
    merged_df = df.merge(df_s5, on="filename", how="left")
    return merged_df

def build_dataframe(eagle_dir: str, s5cmd_file: Optional[str] = None) -> pd.DataFrame:
    """
    Main function to build the final metadata DataFrame from an Eagle library path.
    """
    # Eagle library images path
    eagle_img_dir = os.path.join(eagle_dir, "images")
    eagle_jsons = load_eagle_jsons(eagle_img_dir)
    df_cleaned = eagle_jsons_to_df(eagle_jsons)
    df_merged = add_s3_uri_col(df_cleaned, s5cmd_file)
    return df_merged

def export_parquet(df: pd.DataFrame, output_path: str):
    """
    Exports a DataFrame to a Parquet file.
    """
    df.to_parquet(output_path, index=False)
    print(f"Saved parquet to: {output_path}")

def export_huggingface(df: pd.DataFrame, repo_id: str, private: bool = False):
    """
    Exports a DataFrame to a Hugging Face dataset (push_to_hub).
    """
    from datasets import Dataset
    dataset = Dataset.from_pandas(df)
    result = dataset.push_to_hub(repo_id, private=private)
    print(f"Pushed to Hugging Face: {result}")
