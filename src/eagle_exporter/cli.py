import click
import os
from .core import build_dataframe, export_parquet, export_huggingface

@click.command()
@click.argument("eagle_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--s5cmd", type=click.Path(exists=True), default=None,
              help="Path to a s5cmd file to add s3 URIs.")
@click.option("--to", "dest", default="eagle_metadata.parquet",
              help="Destination: either a .parquet filename or a Hugging Face repo id (e.g. user/dataset).")
@click.option("--hf-public", is_flag=True,
              help="If exporting to Hugging Face, make the dataset public.")
def main(eagle_dir, s5cmd, dest, hf_public):
    """
    Exports Eagle library metadata from EAGLE_DIR to the specified destination.
    The EAGLE_DIR should contain 'images/' with Eagle JSON files.

    Examples:
      eagle-export /path/to/library --to out.parquet
      eagle-export /path/to/library --to myuser/mydataset --hf-public
    """
    # 1) Build the main DataFrame from Eagle
    df = build_dataframe(eagle_dir, s5cmd)

    # 2) Decide if 'dest' is a parquet path or a Hugging Face dataset
    if dest.lower().endswith(".parquet"):
        # Export to Parquet
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        export_parquet(df, dest)
    else:
        # Export to Hugging Face
        export_huggingface(df, dest, private=not(hf_public))
