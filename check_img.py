from PIL import Image
import os

train_folder = './training'

bad_files = []

for fname in sorted(os.listdir(train_folder)):
    if not fname.endswith('.png'):
        continue
    path = os.path.join(train_folder, fname)
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as e:
        print(f"❌ Corrupt file: {fname} — {e}")
        bad_files.append(fname)

print("\nTotal bad files:", len(bad_files))
