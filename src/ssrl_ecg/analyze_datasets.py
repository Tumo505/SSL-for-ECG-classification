from __future__ import annotations

import argparse
from pathlib import Path

from ssrl_ecg.data.ptbxl import DEFAULT_DATA_ROOT, DIAGNOSTIC_CLASSES, load_ptbxl_metadata, make_default_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print dataset readiness and class stats for PTB-XL.")
    parser.add_argument("--ptbxl-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--mitbih-root", type=Path, default=Path("data/MIT-BIH/files/mitdb/1.0.0"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    db_df, labels = load_ptbxl_metadata(args.ptbxl_root)
    splits = make_default_splits(db_df)

    print("PTB-XL rows:", len(db_df))
    print("PTB-XL unique patients:", db_df["patient_id"].nunique())
    print("PTB-XL fold sizes:", {
        "train(1-8)": len(splits.train_idx),
        "val(9)": len(splits.val_idx),
        "test(10)": len(splits.test_idx),
    })

    class_counts = labels.sum(axis=0)
    print("PTB-XL superclass counts:")
    for cls, cnt in zip(DIAGNOSTIC_CLASSES, class_counts):
        print(f"  {cls}: {int(cnt)}")

    lr_dat = sum((args.ptbxl_root / (p + ".dat")).exists() for p in db_df["filename_lr"].values)
    hr_dat = sum((args.ptbxl_root / (p + ".dat")).exists() for p in db_df["filename_hr"].values)
    print("PTB-XL availability:", {"lr_dat": f"{lr_dat}/{len(db_df)}", "hr_dat": f"{hr_dat}/{len(db_df)}"})

    mit_root = args.mitbih_root
    hea = len(list(mit_root.glob("*.hea")))
    dat = len(list(mit_root.glob("*.dat")))
    atr = len(list(mit_root.glob("*.atr")))
    print("MIT-BIH availability:", {"hea": hea, "dat": dat, "atr": atr})


if __name__ == "__main__":
    main()
