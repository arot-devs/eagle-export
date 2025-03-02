import os
import json
import glob
import pandas as pd
from typing import Optional, List, Dict, Union, Any
from datasets import Dataset, Features, Image
from PIL import Image as PILImage
import io
from tqdm import tqdm
import unibox as ub

def load_eagle_jsons(eagle_img_dir: str) -> List[dict]:
    """
    Scans eagle_img_dir and its subdirectories for .json files and loads each into a Python dict.
    """
    # Use **/*.json for recursive globbing
    json_files = ub.traverses(eagle_img_dir, ["metadata.json"])
    return ub.concurrent_loads(json_files)

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

def find_image_path(eagle_dir: str, folder_id: str, filename: str) -> Optional[str]:
    """
    Finds the actual image path within an Eagle library directory.
    
    Args:
        eagle_dir: Path to Eagle library directory
        folder_id: The folder ID from the metadata
        filename: The filename including extension
        
    Returns:
        Full path to the image file or None if not found
    """
    # The actual image is typically stored in a subfolder with the folder ID
    image_dir = os.path.join(eagle_dir, "images", folder_id)
    
    # Check if the directory exists
    if not os.path.exists(image_dir):
        return None
    
    # Try to find the exact file
    image_path = os.path.join(image_dir, filename)
    if os.path.exists(image_path):
        return image_path
    
    # If not found, try a case-insensitive match (common issue with extensions)
    for file in os.listdir(image_dir):
        if file.lower() == filename.lower():
            return os.path.join(image_dir, file)
    
    return None

def load_image(image_path: str) -> Optional[bytes]:
    """
    Load an image file and return its bytes.
    
    Args:
        image_path: Full path to the image file
        
    Returns:
        Image bytes or None if loading fails
    """
    try:
        with open(image_path, 'rb') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def add_images(df: pd.DataFrame, eagle_dir: str, include_images: bool = False) -> pd.DataFrame:
    """
    Adds image paths and optionally loads the actual images.
    
    Args:
        df: DataFrame with metadata
        eagle_dir: Path to Eagle library directory
        include_images: If True, loads the actual images as bytes
        
    Returns:
        DataFrame with added image information
    """
    if not include_images:
        # Just add image paths without loading images
        df['image_path'] = df.apply(
            lambda row: find_image_path(eagle_dir, row.get('folders', [''])[0], row['filename']), 
            axis=1
        )
        return df
    
    # Add both paths and image data
    image_data = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading images"):
        folder_id = row.get('folders', [''])[0]
        filename = row['filename']
        image_path = find_image_path(eagle_dir, folder_id, filename)
        
        if image_path:
            image_bytes = load_image(image_path)
            image_data.append({
                'image_path': image_path,
                'image': image_bytes if image_bytes else None
            })
        else:
            image_data.append({
                'image_path': None,
                'image': None
            })
    
    # Add image info to DataFrame
    image_df = pd.DataFrame(image_data)
    return pd.concat([df.reset_index(drop=True), image_df.reset_index(drop=True)], axis=1)

def build_dataframe(eagle_dir: str, s5cmd_file: Optional[str] = None, include_images: bool = False) -> pd.DataFrame:
    """
    Main function to build the final metadata DataFrame from an Eagle library path.
    
    Args:
        eagle_dir: Path to Eagle library directory
        s5cmd_file: Optional path to s5cmd file for S3 URIs
        include_images: If True, loads the actual images
        
    Returns:
        DataFrame with metadata and optionally images
    """
    # Eagle library images path
    eagle_img_dir = os.path.join(eagle_dir, "images")
    eagle_jsons = load_eagle_jsons(eagle_img_dir)
    df_cleaned = eagle_jsons_to_df(eagle_jsons)
    df_with_s3 = add_s3_uri_col(df_cleaned, s5cmd_file)
    df_with_images = add_images(df_with_s3, eagle_dir, include_images)
    return df_with_images

def export_parquet(df: pd.DataFrame, output_path: str):
    """
    Exports a DataFrame to a Parquet file.
    Note: If 'image' column exists, it will be dropped as binary data
    is not well-suited for Parquet format.
    """
    # Make a copy to avoid modifying the original DataFrame
    export_df = df.copy()
    
    # Drop binary image data if present
    if 'image' in export_df.columns:
        export_df = export_df.drop(columns=['image'])
        print("Note: Image binary data was removed for Parquet export.")
    
    export_df.to_parquet(output_path, index=False)
    print(f"Saved parquet to: {output_path}")

def export_huggingface(df: pd.DataFrame, repo_id: str, private: bool = False):
    """
    Exports a DataFrame to a Hugging Face dataset (push_to_hub).
    If 'image' column exists, it will be properly formatted as a Dataset
    with image features.
    
    Args:
        df: DataFrame with metadata and optionally images
        repo_id: Hugging Face repository ID
        private: If True, the dataset will be private
    """
    from datasets import Dataset, Features, Value, Image as DSImage
    
    # Check if we have image data
    has_images = 'image' in df.columns and any(df['image'].notna())
    
    if has_images:
        # Convert binary image data to PIL images
        images = []
        for img_bytes in tqdm(df['image'], desc="Processing images for HF"):
            if img_bytes is not None:
                try:
                    # Convert bytes to PIL Image
                    img = PILImage.open(io.BytesIO(img_bytes))
                    images.append(img)
                except Exception:
                    images.append(None)
            else:
                images.append(None)
        
        # Create a new DataFrame without the original binary data
        hf_df = df.drop(columns=['image']).copy()
        hf_df['image'] = images
        
        # Create dataset with proper image features
        features = Features({
            'image': DSImage() if has_images else None
        })
        
        dataset = Dataset.from_pandas(hf_df)
    else:
        dataset = Dataset.from_pandas(df)
    
    result = dataset.push_to_hub(repo_id, private=private)
    print(f"Pushed to Hugging Face: {result}")