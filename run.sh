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


python -m pip install scipy numba pyyaml easydict tqdm tensorboardX scikit-learn SharedArray gpustat gdown
python -m pip install scikit-image==0.21.0
python -m pip install nuscenes-devkit==1.0.5
python -m pip install --force-reinstall --no-cache-dir numpy==1.26.4 opencv-python-headless==4.11.0.86
pip install spconv-cu120 || pip install spconv-cu121 || pip install spconv-cu118

echo "===== Patch nuScenes devkit NumPy aliases ====="

python - <<'PY'
from pathlib import Path
import inspect
import re
import nuscenes

root = Path(inspect.getfile(nuscenes)).parent
print("nuScenes package root:", root)

repls = {
    r"\bnp\.float\b": "float",
    r"\bnp\.int\b": "int",
    r"\bnp\.bool\b": "bool",
}

patched = []

for p in root.rglob("*.py"):
    text = p.read_text()
    new = text

    for pattern, replacement in repls.items():
        new = re.sub(pattern, replacement, new)

    if new != text:
        p.write_text(new)
        patched.append(str(p))

print("Patched nuScenes files:")
for p in patched:
    print("  ", p)

if not patched:
    print("No deprecated NumPy aliases found in nuScenes package.")
PY








echo "===== Patch OpenPCDet compatibility ====="

python - <<'PY'
from pathlib import Path

# Patch only the exact deprecated NumPy aliases.
# Do not use plain string replacement here: replacing "np.int" would also
# corrupt valid names such as np.int32 and np.int64.
import re

repls = {
    r"\bnp\.int\b": "int",
    r"\bnp\.float\b": "float",
    r"\bnp\.bool\b": "bool",
}

for folder in ["pcdet", "tools"]:
    if not Path(folder).exists():
        continue

    for p in Path(folder).rglob("*.py"):
        text = p.read_text()
        new = text

        for pattern, replacement in repls.items():
            new = re.sub(pattern, replacement, new)

        if new != text:
            p.write_text(new)
            print("patched exact NumPy alias:", p)

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
# python setup.py develop
rm -rf build *.egg-info pcdet.egg-info

python setup.py develop --no-deps

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

echo "===== Use existing nuScenes dataset on H200 server ====="

# Use the actual server path.
# You wrote /app/data/nuscense/maps, so the dataset root is probably /app/data/nuscense.
REMOTE_NUSC_ROOT="${REMOTE_NUSC_ROOT:-/app/data/nuscense}"

# OpenPCDet expects:
# data/nuscenes/v1.0-trainval/
#   ├── maps
#   ├── samples
#   ├── sweeps
#   └── v1.0-trainval
LOCAL_NUSC_PARENT="data/nuscenes"
LOCAL_NUSC_ROOT="${LOCAL_NUSC_PARENT}/v1.0-trainval"

mkdir -p "$LOCAL_NUSC_PARENT"

echo "REMOTE_NUSC_ROOT=$REMOTE_NUSC_ROOT"
echo "LOCAL_NUSC_ROOT=$LOCAL_NUSC_ROOT"

# Case 1: remote path is already the nuScenes version root:
# /app/data/nuscense/maps
# /app/data/nuscense/samples
# /app/data/nuscense/sweeps
# /app/data/nuscense/v1.0-trainval
if [[ -d "$REMOTE_NUSC_ROOT/maps" && \
      -d "$REMOTE_NUSC_ROOT/samples" && \
      -d "$REMOTE_NUSC_ROOT/sweeps" && \
      -d "$REMOTE_NUSC_ROOT/v1.0-trainval" ]]; then

    rm -rf "$LOCAL_NUSC_ROOT"
    ln -s "$REMOTE_NUSC_ROOT" "$LOCAL_NUSC_ROOT"

# Case 2: remote path contains v1.0-trainval as a subfolder:
# /app/data/nuscense/v1.0-trainval/maps
# /app/data/nuscense/v1.0-trainval/samples
# /app/data/nuscense/v1.0-trainval/sweeps
elif [[ -d "$REMOTE_NUSC_ROOT/v1.0-trainval/maps" && \
        -d "$REMOTE_NUSC_ROOT/v1.0-trainval/samples" && \
        -d "$REMOTE_NUSC_ROOT/v1.0-trainval/sweeps" ]]; then

    rm -rf "$LOCAL_NUSC_ROOT"
    ln -s "$REMOTE_NUSC_ROOT/v1.0-trainval" "$LOCAL_NUSC_ROOT"

else
    echo "ERROR: Could not find valid nuScenes structure under $REMOTE_NUSC_ROOT"
    echo "Expected one of:"
    echo "  $REMOTE_NUSC_ROOT/maps"
    echo "  $REMOTE_NUSC_ROOT/samples"
    echo "  $REMOTE_NUSC_ROOT/sweeps"
    echo "  $REMOTE_NUSC_ROOT/v1.0-trainval"
    echo ""
    echo "or:"
    echo "  $REMOTE_NUSC_ROOT/v1.0-trainval/maps"
    echo "  $REMOTE_NUSC_ROOT/v1.0-trainval/samples"
    echo "  $REMOTE_NUSC_ROOT/v1.0-trainval/sweeps"
    exit 1
fi

echo "Testing metadata archive..."
sed -n '1,20p' "$META_CONTENTS"

echo "Testing blob01 archive..."
sed -n '1,20p' "$BLOB01_CONTENTS"

if ! grep -Eq '(^|/)v1\.0-trainval/' "$META_CONTENTS"; then
    echo "ERROR: metadata archive does not contain v1.0-trainval/."
    exit 1
fi

if ! grep -Eq '(^|/)maps/' "$META_CONTENTS"; then
    echo "ERROR: metadata archive does not contain maps/."
    echo "The script will not attempt a separate maps download."
    exit 1
fi

echo "Confirmed: maps/ is present inside the metadata archive."

echo "===== Extract nuScenes ====="

tar -xzf "$META_ARCHIVE" -C "$NUSC_ROOT"
tar -xzf "$BLOB01_ARCHIVE" -C "$NUSC_ROOT"

echo "===== Dataset check ====="

required_dirs=(
    "$NUSC_ROOT/v1.0-trainval"
    "$NUSC_ROOT/maps"
    "$NUSC_ROOT/samples/LIDAR_TOP"
    "$NUSC_ROOT/sweeps/LIDAR_TOP"
)

for required_dir in "${required_dirs[@]}"; do
    if [[ ! -d "$required_dir" ]]; then
        echo "ERROR: expected directory was not created: $required_dir"
        echo "Inspect $META_CONTENTS and $BLOB01_CONTENTS for unexpected nesting."
        exit 1
    fi
done

find "$NUSC_ROOT" -maxdepth 2 -type d | sort | sed -n '1,50p'

echo "Metadata files:"
find "$NUSC_ROOT/v1.0-trainval" -maxdepth 1 -type f -printf '%f\n' | sort | sed -n '1,20p'

echo "Map files:"
find "$NUSC_ROOT/maps" -maxdepth 2 -type f -printf '%P\n' | sort | sed -n '1,20p'

echo "LiDAR samples:"
find "$NUSC_ROOT/samples/LIDAR_TOP" -maxdepth 1 -type f -printf '%f\n' | sort | sed -n '1,10p'

echo "LiDAR sweeps:"
find "$NUSC_ROOT/sweeps/LIDAR_TOP" -maxdepth 1 -type f -printf '%f\n' | sort | sed -n '1,10p'

echo "Metadata JSON count: $(find "$NUSC_ROOT/v1.0-trainval" -maxdepth 1 -type f -name '*.json' | wc -l)"
echo "Map file count:       $(find "$NUSC_ROOT/maps" -type f | wc -l)"
echo "LiDAR sample count:   $(find "$NUSC_ROOT/samples/LIDAR_TOP" -maxdepth 1 -type f | wc -l)"
echo "LiDAR sweep count:    $(find "$NUSC_ROOT/sweeps/LIDAR_TOP" -maxdepth 1 -type f | wc -l)"

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
    --workers 4 \
    --epochs 30 \
    --wo_gpu_stat

cd ..

echo "===== Save results ====="

mkdir -p h200_results

cp -r output h200_results/ || true
cp -r data/nuscenes/v1.0-trainval/*.pkl h200_results/ || true

if [ -f output/nuscenes_models/transfusion_lidar/default/eval/epoch_2/val/default/final_result/data/results_nusc.json ]; then
    cp output/nuscenes_models/transfusion_lidar/default/eval/epoch_30/val/default/final_result/data/results_nusc.json h200_results/ || true
fi

tar -czf h200_transfusion_part1_results.tar.gz h200_results

echo "DONE: h200_transfusion_part1_results.tar.gz"