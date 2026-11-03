import csv
import json
import sys
from collections import Counter
from pathlib import Path

coco_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/processed/coco.json")
manifest_path = Path(sys.argv[2] if len(sys.argv) > 2 else "splits/v1.csv")

coco = json.loads(coco_path.read_text())
names = {c["id"]: c["name"] for c in coco["categories"]}
images = {i["id"]: i for i in coco["images"]}
with manifest_path.open() as f:
    split_of = {row["image_id"]: row["split"] for row in csv.DictReader(f)}


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q / 100 * len(values)))] if values else 0


print("== Images")
print("count:", len(images))
print("sizes (w x h):", dict(Counter(f"{i['width']}x{i['height']}" for i in images.values())))

per_image = Counter(a["image_id"] for a in coco["annotations"])
counts = [per_image.get(i, 0) for i in images]
print("\n== Vehicles per image")
print(f"mean {sum(counts) / len(counts):.1f}  median {pct(counts, 50)}  max {max(counts)}")
print("images with 0 vehicles:", sum(c == 0 for c in counts))
print("images with >100 vehicles (mAP ignores detections past 100):", sum(c > 100 for c in counts))

sides = [(a["bbox"][2] * a["bbox"][3]) ** 0.5 for a in coco["annotations"]]
print("\n== Box size (sqrt of area, pixels)")
print("  ".join(f"p{q} {pct(sides, q):.0f}" for q in (1, 5, 25, 50, 75, 95, 99)))
small = sum(s < 32 for s in sides)
medium = sum(32 <= s < 96 for s in sides)
large = len(sides) - small - medium
total = len(sides)
print(
    f"COCO buckets: small {small} ({small / total:.1%})  "
    f"medium {medium} ({medium / total:.1%})  large {large} ({large / total:.1%})"
)

edge = 0
for a in coco["annotations"]:
    x, y, w, h = a["bbox"]
    img = images[a["image_id"]]
    if x <= 0.5 or y <= 0.5 or x + w >= img["width"] - 0.5 or y + h >= img["height"] - 0.5:
        edge += 1
print(f"boxes touching the image edge: {edge} ({edge / total:.1%})")

print("\n== Boxes per class and split (splits/v1.csv)")
splits = ["train", "val", "test"]
table = {cid: Counter() for cid in names}
split_images = Counter()
for i in images.values():
    split_images[split_of.get(Path(i["file_name"]).stem, "missing")] += 1
for a in coco["annotations"]:
    split = split_of.get(Path(images[a["image_id"]]["file_name"]).stem, "missing")
    table[a["category_id"]][split] += 1
print(f"{'class':<14}" + "".join(f"{s:>8}" for s in splits) + f"{'total':>8}")
for cid, name in sorted(names.items()):
    row = table[cid]
    print(f"{name:<14}" + "".join(f"{row[s]:>8}" for s in splits) + f"{sum(row.values()):>8}")
print(f"{'images':<14}" + "".join(f"{split_images[s]:>8}" for s in splits) + f"{len(images):>8}")
if split_images["missing"]:
    print("images not in the manifest:", split_images["missing"])
