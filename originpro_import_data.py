# -*- coding: utf-8 -*-
"""
Optional OriginPro automation starter script.

This script is intentionally conservative:
it imports the generated CSV files into Origin workbooks.
After import, you can use Origin's GUI to draw grouped bars, ROC curves,
scatter plots, and histograms.

Run requirement:
    pip install originpro
or run inside Origin's embedded Python if originpro is already available.

Usage:
    python originpro_import_data.py --data-dir origin_data --opju origin_project/coal_rock_figures.opju

Note:
    Different Origin versions may have slightly different template names.
    This starter script focuses on importing and organizing data reliably.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="origin_data")
    parser.add_argument("--opju", default="origin_project/coal_rock_figures.opju")
    args = parser.parse_args()

    try:
        import originpro as op
    except Exception as e:
        raise SystemExit(
            "Cannot import originpro. Please install OriginPro Python package "
            "or run this script inside Origin's embedded Python.\n"
            f"Original error: {e}"
        )

    data_dir = Path(args.data_dir).resolve()
    opju_path = Path(args.opju).resolve()
    opju_path.parent.mkdir(parents=True, exist_ok=True)

    op.set_show(True)
    op.new()

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    for csv in csv_files:
        short_name = csv.stem[:24].replace("-", "_").replace(" ", "_")
        book = op.new_book("w", lname=short_name)
        sheet = book[0]
        try:
            sheet.from_file(str(csv))
            sheet.name = short_name[:31]
            print("Imported:", csv)
        except Exception as e:
            print("Failed to import:", csv, "Error:", e)

    op.save(str(opju_path))
    print("\nSaved Origin project:", opju_path)
    print("Next step: use Origin GUI to create grouped bars, ROC curves, scatter plots, and histograms.")


if __name__ == "__main__":
    main()
