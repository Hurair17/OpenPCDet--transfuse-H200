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

echo "===== Prepare nuScenes using mounted raw data + precomputed GT database ====="

DOWNLOAD_DIR="downloads"
LOCAL_NUSC_PARENT="data/nuscenes"
LOCAL_NUSC_ROOT="${LOCAL_NUSC_PARENT}/v1.0-trainval"

mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$LOCAL_NUSC_PARENT"

echo "===== Debug /app/data ====="
ls -lah /app || true
ls -lah /app/data || true

echo "Searching for nuScenes-like folders:"
find /app -maxdepth 5 -type d \( \
    -name "nuscenes" -o \
    -name "nuscense" -o \
    -name "nuScenes" -o \
    -name "v1.0-trainval" -o \
    -name "samples" -o \
    -name "sweeps" -o \
    -name "maps" \
\) 2>/dev/null | sort | sed -n '1,150p'

CANDIDATE_ROOTS=(
    "/app/data/nuscenes"
    "/app/data/nuscense"
    "/app/data/nuScenes"
    "/app/data/nuscenes/v1.0-trainval"
    "/app/data/nuscense/v1.0-trainval"
    "/app/data/nuScenes/v1.0-trainval"
)

FOUND_NUSC_ROOT=""

for candidate in "${CANDIDATE_ROOTS[@]}"; do
    echo "Checking candidate: $candidate"

    if [[ -d "$candidate/maps" && \
          -d "$candidate/samples" && \
          -d "$candidate/sweeps" && \
          -d "$candidate/v1.0-trainval" ]]; then
        FOUND_NUSC_ROOT="$candidate"
        break
    fi
done

if [[ -z "$FOUND_NUSC_ROOT" ]]; then
    echo "ERROR: Could not find mounted nuScenes raw dataset."
    echo "Expected:"
    echo "  ROOT/maps"
    echo "  ROOT/samples"
    echo "  ROOT/sweeps"
    echo "  ROOT/v1.0-trainval"
    find /app/data -maxdepth 5 -type d 2>/dev/null | sort | sed -n '1,200p'
    exit 1
fi

echo "Found raw nuScenes root: $FOUND_NUSC_ROOT"

# Create writable local root. Do NOT symlink the whole root,
# because OpenPCDet writes .pkl files into this folder.
rm -rf "$LOCAL_NUSC_ROOT"
mkdir -p "$LOCAL_NUSC_ROOT"

ln -s "$FOUND_NUSC_ROOT/maps" "$LOCAL_NUSC_ROOT/maps"
ln -s "$FOUND_NUSC_ROOT/samples" "$LOCAL_NUSC_ROOT/samples"
ln -s "$FOUND_NUSC_ROOT/sweeps" "$LOCAL_NUSC_ROOT/sweeps"
ln -s "$FOUND_NUSC_ROOT/v1.0-trainval" "$LOCAL_NUSC_ROOT/v1.0-trainval"

echo "Local writable nuScenes root:"
ls -lah "$LOCAL_NUSC_ROOT"

echo "===== Download precomputed nuScenes info/GT database from Google Drive ====="

PRECOMP_GDRIVE_ID="1cl5bMmNG-12qAqXXnXrYe8csisC_NexI"
PRECOMP_ARCHIVE="$DOWNLOAD_DIR/nuscenes_precomputed"
PRECOMP_EXTRACT="$DOWNLOAD_DIR/nuscenes_precomputed_extract"

rm -rf "$PRECOMP_EXTRACT"
mkdir -p "$PRECOMP_EXTRACT"

gdown --continue "https://drive.google.com/uc?id=${PRECOMP_GDRIVE_ID}" -O "$PRECOMP_ARCHIVE"

echo "Downloaded precomputed archive:"
ls -lh "$PRECOMP_ARCHIVE"
file "$PRECOMP_ARCHIVE" || true

echo "===== Extract precomputed files ====="

if tar -tzf "$PRECOMP_ARCHIVE" >/dev/null 2>&1; then
    tar -xzf "$PRECOMP_ARCHIVE" -C "$PRECOMP_EXTRACT"
elif unzip -t "$PRECOMP_ARCHIVE" >/dev/null 2>&1; then
    unzip -q "$PRECOMP_ARCHIVE" -d "$PRECOMP_EXTRACT"
else
    echo "ERROR: precomputed file is neither .tar.gz nor .zip"
    file "$PRECOMP_ARCHIVE" || true
    exit 1
fi

echo "Precomputed extracted files:"
find "$PRECOMP_EXTRACT" -maxdepth 4 -type f | sort | sed -n '1,100p'
find "$PRECOMP_EXTRACT" -maxdepth 4 -type d | sort | sed -n '1,100p'

echo "===== Copy precomputed .pkl/.npy and GT database into local nuScenes root ====="

python - <<'PY'
from pathlib import Path
import shutil
import os

src = Path("downloads/nuscenes_precomputed_extract")
dst = Path("data/nuscenes/v1.0-trainval")

dst.mkdir(parents=True, exist_ok=True)

# Copy standard precomputed files.
for p in src.rglob("*"):
    if p.is_file() and (
        p.name.endswith(".pkl")
        or p.name.endswith(".npy")
        or p.name.endswith(".json")
    ):
        target = dst / p.name
        shutil.copy2(p, target)
        print("copied file:", p, "->", target)

# Copy GT database directories if present.
for d in src.rglob("*"):
    if d.is_dir() and ("gt_database" in d.name or "database" in d.name):
        target = dst / d.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(d, target)
        print("copied dir:", d, "->", target)

required = [
    "nuscenes_infos_10sweeps_train.pkl",
    "nuscenes_infos_10sweeps_val.pkl",
    "nuscenes_dbinfos_10sweeps_withvelo.pkl",
]

missing = [name for name in required if not (dst / name).exists()]
if missing:
    print("ERROR: Missing required precomputed files:")
    for name in missing:
        print("  ", name)
    print("Available files in local nuScenes root:")
    for p in sorted(dst.iterdir()):
        print("  ", p.name)
    raise SystemExit(1)

print("Required precomputed files found.")
PY

echo "===== Dataset check ====="

required_dirs=(
    "$LOCAL_NUSC_ROOT/v1.0-trainval"
    "$LOCAL_NUSC_ROOT/maps"
    "$LOCAL_NUSC_ROOT/samples/LIDAR_TOP"
    "$LOCAL_NUSC_ROOT/sweeps/LIDAR_TOP"
)

for required_dir in "${required_dirs[@]}"; do
    if [[ ! -d "$required_dir" ]]; then
        echo "ERROR: expected directory missing: $required_dir"
        exit 1
    fi
done

echo "Precomputed files:"
ls -lh "$LOCAL_NUSC_ROOT"/*.pkl "$LOCAL_NUSC_ROOT"/*.npy 2>/dev/null || true

echo "GT database dirs:"
find "$LOCAL_NUSC_ROOT" -maxdepth 1 -type d \( -name "*gt_database*" -o -name "*database*" \) -print || true

echo "Dataset preparation finished."

echo "===== Train TransFusion-L LiDAR-only ====="

cd tools


CUDA_VISIBLE_DEVICES=0 python train.py     --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml     --batch_size 1     --epochs 30     --wo_gpu_stat

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