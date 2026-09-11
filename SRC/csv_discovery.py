from pathlib import Path
import argparse
import json
import sys

import pandas as pd


# =========================================================
# COMMAND-LINE ARGUMENTS
# =========================================================

def get_arguments():

    parser = argparse.ArgumentParser(
        description="Discover and profile CSV files."
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help="Folder containing the CSV files."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path where csv_metadata.json will be saved."
    )

    return parser.parse_args()


# =========================================================
# DISCOVER CSV FILES
# =========================================================

def find_csv_files(data_dir: Path):

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Selected data folder does not exist:\n{data_dir}"
        )

    if not data_dir.is_dir():
        raise NotADirectoryError(
            f"Selected path is not a directory:\n{data_dir}"
        )

    csv_files = sorted(
        data_dir.glob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files were found in:\n{data_dir}"
        )

    return csv_files


# =========================================================
# JSON-SAFE SAMPLE VALUES
# =========================================================

def make_json_safe(value):

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if isinstance(
        value,
        (str, int, float, bool)
    ):
        return value

    return str(value)


# =========================================================
# PROFILE ONE CSV
# =========================================================

def profile_csv(file_path: Path):

    print(
        f"Profiling: {file_path.name}",
        flush=True
    )

    df = pd.read_csv(
        file_path,
        low_memory=False
    )

    profile = {
        "file_name": file_path.name,
        "file_path": str(file_path.resolve()),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_details": {}
    }

    for column in df.columns:

        raw_samples = (
            df[column]
            .dropna()
            .head(5)
            .tolist()
        )

        sample_values = [
            make_json_safe(value)
            for value in raw_samples
        ]

        profile["column_details"][str(column)] = {
            "dtype": str(df[column].dtype),

            "null_count": int(
                df[column].isna().sum()
            ),

            "unique_count": int(
                df[column].nunique(
                    dropna=True
                )
            ),

            "sample_values": sample_values
        }

    return profile


# =========================================================
# PROFILE ALL CSV FILES
# =========================================================

def profile_all_csvs(data_dir: Path):

    csv_files = find_csv_files(
        data_dir
    )

    profiles = []

    for file_path in csv_files:

        profile = profile_csv(
            file_path
        )

        profiles.append(
            profile
        )

    return profiles


# =========================================================
# SAVE METADATA
# =========================================================

def save_metadata(
    profiles,
    output_file: Path
):

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            profiles,
            file,
            indent=2,
            ensure_ascii=False
        )

    if not output_file.exists():
        raise RuntimeError(
            f"Metadata file was not created:\n{output_file}"
        )

    if output_file.stat().st_size == 0:
        raise RuntimeError(
            f"Metadata file was created but is empty:\n{output_file}"
        )


# =========================================================
# MAIN
# =========================================================

def main():

    args = get_arguments()

    data_dir = Path(
        args.data_dir
    ).expanduser()

    output_file = Path(
        args.output
    ).expanduser()

    print("\n" + "=" * 80)
    print("CSV DISCOVERY")
    print("=" * 80)

    print(
        f"\nSelected data source:\n{data_dir}"
    )

    print(
        f"\nMetadata output:\n{output_file}\n"
    )

    profiles = profile_all_csvs(
        data_dir
    )

    save_metadata(
        profiles,
        output_file
    )

    print(
        f"\nFound {len(profiles)} CSV files."
    )

    print("\n" + "=" * 80)
    print("CSV DISCOVERY COMPLETE")
    print("=" * 80)

    print(
        f"\nMetadata saved to:\n{output_file}"
    )

    print(
        f"\nFile exists: {output_file.exists()}"
    )

    print(
        f"File size: {output_file.stat().st_size:,} bytes\n"
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print(
            "\nCSV DISCOVERY ERROR",
            file=sys.stderr
        )

        print(
            str(error),
            file=sys.stderr
        )

        sys.exit(1)