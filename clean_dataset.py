# clean_dataset.py
import os
import glob

GARBAGE_FONTS = [
    "symbol", "webdings", "wingding", "seguisym", "seguiemj",
    "marlett", "segmdl2", "SegoeIcons", "wingdings"
]

DATASET_DIR = "./dataset"

removed = []

for digit in range(10):
    for font in GARBAGE_FONTS:
        # case-insensitive match
        pattern = os.path.join(DATASET_DIR, str(digit), f"{font}.png")
        matches = glob.glob(pattern, recursive=False)
        
        # also try case-insensitive by scanning the folder
        folder = os.path.join(DATASET_DIR, str(digit))
        for filename in os.listdir(folder):
            name_no_ext = os.path.splitext(filename)[0].lower()
            if name_no_ext in [g.lower() for g in GARBAGE_FONTS]:
                full_path = os.path.join(folder, filename)
                if full_path not in removed:
                    os.remove(full_path)
                    removed.append(full_path)
                    print(f"Removed: {full_path}")

print(f"\nDone. {len(removed)} files removed.")