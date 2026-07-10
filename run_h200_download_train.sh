#!/usr/bin/env bash
set -euo pipefail

echo "===== H200 OpenPCDet TransFusion-L with nuScenes Part 1 ====="

cd "$(dirname "$0")"

echo "===== System check ====="
python --version
nvidia-smi || true

echo "===== Environment settings ====="
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo "===== Install dependencies ====="

pip install -U pip
pip install "setuptools<82" wheel

pip uninstall -y opencv-python opencv-contrib-python opencv-contrib-python-headless || true

pip install numpy==1.26.4 opencv-python-headless==4.11.0.86
pip install nuscenes-devkit pyyaml easydict tqdm tensorboardX scikit-learn SharedArray gpustat gdown

pip install spconv-cu120 || pip install spconv-cu121 || pip install spconv-cu118

echo "===== Patch OpenPCDet compatibility ====="

python - <<'PY'
from pathlib import Path

# Patch old NumPy aliases.
repls = {
    "np.int": "int",
    "np.float": "float",
    "np.bool": "bool",
}

for folder in ["pcdet", "tools"]:
    if not Path(folder).exists():
        continue

    for p in Path(folder).rglob("*.py"):
        text = p.read_text()
        new = text

        for old, new_val in repls.items():
            new = new.replace(old, new_val)

        if new != text:
            p.write_text(new)
            print("patched numpy alias:", p)

# Disable Argo2 import because this run is nuScenes-only.
p = Path("pcdet/datasets/__init__.py")
if p.exists():
    text = p.read_text()
    new_lines = []

    for line in text.splitlines():
        if "Argo2Dataset" in line and not line.lstrip().startswith("#"):
            indent = line[:len(line) - len(line.lstrip())]
            new_lines.append(indent + "# " + line.lstrip() + "  # disabled for nuScenes-only run")
        else:
            new_lines.append(line)

    p.write_text("\n".join(new_lines) + "\n")
    print("Argo2 disabled if present.")

# Patch torch.load for PyTorch >= 2.6 checkpoint loading.
p = Path("pcdet/models/detectors/detector3d_template.py")
if p.exists():
    text = p.read_text()
    old = "torch.load(filename, map_location=loc_type)"
    new = "torch.load(filename, map_location=loc_type, weights_only=False)"

    if old in text:
        p.write_text(text.replace(old, new))
        print("patched torch.load weights_only=False")
    elif "weights_only=False" in text:
        print("torch.load patch already present")
    else:
        print("WARNING: torch.load line not found. Check detector3d_template.py manually.")
PY

echo "===== Patch nuScenes evaluator for subset evaluation ====="

python - <<'PY'
from pathlib import Path
import inspect
import nuscenes.eval.detection.evaluate as ev

p = Path(inspect.getfile(ev))
text = p.read_text()

print("nuScenes evaluator:", p)

if "[Subset Eval]" in text:
    print("Subset eval patch already exists.")
    raise SystemExit

backup = p.with_suffix(".py.bak_subset")
if not backup.exists():
    backup.write_text(text)
    print("Backup saved:", backup)

lines = text.splitlines()
out = []
patched = False
i = 0

while i < len(lines):
    line = lines[i]

    if (
        "assert" in line
        and "pred_boxes.sample_tokens" in line
        and "gt_boxes.sample_tokens" in line
    ):
        indent = line[:len(line) - len(line.lstrip())]

        out.extend([
            indent + "# Patched for partial nuScenes subset evaluation.",
            indent + "# Official nuScenes evaluation requires prediction and GT sample tokens to match exactly.",
            indent + "# For part-one-only experiments, evaluate only common prediction/GT tokens.",
            indent + "pred_tokens = set(self.pred_boxes.sample_tokens)",
            indent + "gt_tokens = set(self.gt_boxes.sample_tokens)",
            indent + "if pred_tokens != gt_tokens:",
            indent + "    common_tokens = pred_tokens & gt_tokens",
            indent + "    print(f\"[Subset Eval] Prediction samples: {len(pred_tokens)}\")",
            indent + "    print(f\"[Subset Eval] Official GT samples: {len(gt_tokens)}\")",
            indent + "    print(f\"[Subset Eval] Evaluating common subset samples: {len(common_tokens)}\")",
            indent + "    self.pred_boxes.boxes = {",
            indent + "        token: self.pred_boxes.boxes[token]",
            indent + "        for token in common_tokens",
            indent + "        if token in self.pred_boxes.boxes",
            indent + "    }",
            indent + "    self.gt_boxes.boxes = {",
            indent + "        token: self.gt_boxes.boxes[token]",
            indent + "        for token in common_tokens",
            indent + "        if token in self.gt_boxes.boxes",
            indent + "    }",
        ])

        patched = True
        i += 1

        if i < len(lines) and "Samples in split" in lines[i]:
            i += 1

        continue

    out.append(line)
    i += 1

if not patched:
    print("WARNING: Could not find official sample-token assertion. Continuing without evaluator patch.")
else:
    p.write_text("\n".join(out) + "\n")
    print("Subset evaluation patch applied.")
PY

echo "===== Build OpenPCDet ====="
python setup.py develop

echo "===== Import test ====="

python - <<'PY'
import numpy
import cv2
import torch
import spconv
import pcdet
import nuscenes

print("NumPy:", numpy.__version__)
print("OpenCV:", cv2.__version__)
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("spconv OK")
print("OpenPCDet OK")
print("nuScenes devkit OK")
PY

echo "===== Download nuScenes part one ====="

mkdir -p downloads
mkdir -p data/nuscenes/v1.0-trainval

# Google Drive IDs.
# First link: metadata.
# Second link: trainval part-one blob.
NUSC_META_GDRIVE_ID="${NUSC_META_GDRIVE_ID:-1RhJHJPC_euONoxHPDth2IXgH8G-Tr4Ku}"
NUSC_BLOB01_GDRIVE_ID="${NUSC_BLOB01_GDRIVE_ID:-15Gyeo7X7qelTxXPKW3z4lhEtoTkydBum}"

# Maps public nuScenes link.
NUSC_MAPS_URL="${NUSC_MAPS_URL:-https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/maps.tgz}"

echo "Metadata Google Drive ID: $NUSC_META_GDRIVE_ID"
echo "Blob01 Google Drive ID: $NUSC_BLOB01_GDRIVE_ID"
echo "Maps URL: $NUSC_MAPS_URL"

echo "Downloading metadata from Google Drive..."
gdown --id "$NUSC_META_GDRIVE_ID" -O downloads/v1.0-trainval_meta.tgz

echo "Downloading trainval part-one blob from Google Drive..."
gdown --id "$NUSC_BLOB01_GDRIVE_ID" -O downloads/v1.0-trainval01_blobs.tgz

echo "Downloading maps from nuScenes public link..."
curl -L -C - "$NUSC_MAPS_URL" -o downloads/maps.tgz

echo "===== Check downloaded archives ====="

ls -lh downloads/

file downloads/v1.0-trainval_meta.tgz
file downloads/v1.0-trainval01_blobs.tgz
file downloads/maps.tgz

echo "Testing metadata archive..."
tar -tzf downloads/v1.0-trainval_meta.tgz | head

echo "Testing maps archive..."
tar -tzf downloads/maps.tgz | head

echo "Testing blob01 archive..."
tar -tzf downloads/v1.0-trainval01_blobs.tgz | head

echo "===== Extract nuScenes ====="

tar -xzf downloads/v1.0-trainval_meta.tgz -C data/nuscenes/v1.0-trainval
tar -xzf downloads/maps.tgz -C data/nuscenes/v1.0-trainval
tar -xzf downloads/v1.0-trainval01_blobs.tgz -C data/nuscenes/v1.0-trainval

echo "===== Dataset check ====="

find data/nuscenes/v1.0-trainval -maxdepth 2 -type d | sort | head -50

echo "Metadata files:"
ls data/nuscenes/v1.0-trainval/v1.0-trainval | head

echo "LiDAR samples:"
ls data/nuscenes/v1.0-trainval/samples/LIDAR_TOP | head

echo "LiDAR sweeps:"
ls data/nuscenes/v1.0-trainval/sweeps/LIDAR_TOP | head

echo "===== Create nuScenes info files ====="

python -m pcdet.datasets.nuscenes.nuscenes_dataset \
    --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-trainval

echo "===== Filter info files to existing part-one files only ====="

python - <<'PY'
import pickle
from pathlib import Path

root = Path("data/nuscenes/v1.0-trainval")

def exists_path(path):
    if path is None:
        return False

    p = Path(path)

    if p.is_absolute():
        return p.exists()

    return (root / p).exists()

def filter_info_file(pkl_name):
    pkl = root / pkl_name

    if not pkl.exists():
        print("Missing:", pkl)
        return

    backup = root / (pkl_name + ".full_backup")

    data = pickle.load(open(pkl, "rb"))
    infos = data["infos"] if isinstance(data, dict) and "infos" in data else data

    kept = []
    removed = 0

    for info in infos:
        ok = exists_path(info.get("lidar_path"))

        for sweep in info.get("sweeps", []):
            spath = sweep.get("lidar_path", sweep.get("data_path", None))
            if spath is not None and not exists_path(spath):
                ok = False
                break

        if ok:
            kept.append(info)
        else:
            removed += 1

    if not backup.exists():
        backup.write_bytes(pkl.read_bytes())

    if isinstance(data, dict) and "infos" in data:
        data["infos"] = kept
        pickle.dump(data, open(pkl, "wb"))
    else:
        pickle.dump(kept, open(pkl, "wb"))

    print(pkl_name)
    print("  original:", len(infos))
    print("  kept:", len(kept))
    print("  removed:", removed)

filter_info_file("nuscenes_infos_10sweeps_train.pkl")
filter_info_file("nuscenes_infos_10sweeps_val.pkl")
PY

echo "===== Train TransFusion-L LiDAR-only ====="

cd tools

CUDA_VISIBLE_DEVICES=0 python train.py \
    --cfg_file cfgs/nuscenes_models/transfusion_lidar.yaml \
    --batch_size 1 \
    --workers 2 \
    --epochs 2 \
    --wo_gpu_stat

cd ..

echo "===== Save results ====="

mkdir -p h200_results

cp -r output h200_results/ || true
cp -r data/nuscenes/v1.0-trainval/*.pkl h200_results/ || true

if [ -f output/nuscenes_models/transfusion_lidar/default/eval/epoch_2/val/default/final_result/data/results_nusc.json ]; then
    cp output/nuscenes_models/transfusion_lidar/default/eval/epoch_2/val/default/final_result/data/results_nusc.json h200_results/ || true
fi

tar -czf h200_transfusion_part1_results.tar.gz h200_results

echo "DONE: h200_transfusion_part1_results.tar.gz"
