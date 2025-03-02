# eagle_exporter.cli

import os
import click
from .core import build_dataframe, export_parquet, export_huggingface


def export_metadata(eagle_dir, s5cmd, dest, hf_public, include_images):
    """
    Core function for exporting Eagle metadata.
    """
    # Ensure eagle_dir exists
    if not os.path.isdir(eagle_dir):
        raise FileNotFoundError(f"Directory '{eagle_dir}' does not exist.")

    # Build the main DataFrame
    df = build_dataframe(eagle_dir, s5cmd, include_images)

    # Export based on destination type
    if dest.lower().endswith(".parquet"):
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        export_parquet(df, dest)
    else:
        export_huggingface(df, dest, private=not hf_public)


@click.command()
@click.argument("eagle_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--s5cmd", type=click.Path(exists=True), default=None,
              help="Path to a s5cmd file to add s3 URIs.")
@click.option("--to", "dest", default="eagle_metadata.parquet",
              help="Destination: either a .parquet filename or a Hugging Face repo id (e.g. user/dataset).")
@click.option("--hf-public", is_flag=True,
              help="If exporting to Hugging Face, make the dataset public.")
@click.option("--include-images", is_flag=True,
              help="Include the actual image files in the export (only useful for Hugging Face exports).")
def main(eagle_dir, s5cmd, dest, hf_public, include_images):
    """
    CLI entry point for exporting metadata.
    
    EAGLE_DIR should point to an Eagle library directory (the folder containing 'images/').
    
    Examples:
        # Export to Parquet file
        eagle-exporter path/to/eagle.library --to output.parquet
        
        # Export to Hugging Face with images included
        eagle-exporter path/to/eagle.library --to username/dataset --hf-public --include-images
    """
    export_metadata(eagle_dir, s5cmd, dest, hf_public, include_images)