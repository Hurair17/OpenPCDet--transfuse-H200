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

echo "===== Download nuScenes part one from Hugging Face ====="

DOWNLOAD_DIR="downloads"
NUSC_ROOT="data/nuscenes/v1.0-trainval"
META_ARCHIVE="$DOWNLOAD_DIR/v1.0-trainval_meta.tgz"
BLOB01_ARCHIVE="$DOWNLOAD_DIR/v1.0-trainval01_blobs.tgz"
META_CONTENTS="$DOWNLOAD_DIR/v1.0-trainval_meta.contents.txt"
BLOB01_CONTENTS="$DOWNLOAD_DIR/v1.0-trainval01_blobs.contents.txt"

mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$NUSC_ROOT"

HF_DATASET_REPO="Hurair123/globe-part1"

HF_META_URL="https://huggingface.co/datasets/${HF_DATASET_REPO}/resolve/main/v1.0-trainval_meta.tgz"
HF_BLOB01_URL="https://huggingface.co/datasets/${HF_DATASET_REPO}/resolve/main/v1.0-trainval01_blobs.tgz"

echo "HF dataset repo: $HF_DATASET_REPO"
echo "Metadata URL: $HF_META_URL"
echo "Blob01 URL:   $HF_BLOB01_URL"

download_url() {
    local url="$1"
    local output_path="$2"

    echo "Downloading:"
    echo "$url"
    echo "to:"
    echo "$output_path"

    if command -v wget >/dev/null 2>&1; then
        wget -c "$url" -O "$output_path"
    else
        curl -L --retry 5 -C - "$url" -o "$output_path"
    fi
}

download_url "$HF_META_URL" "$META_ARCHIVE"
download_url "$HF_BLOB01_URL" "$BLOB01_ARCHIVE"



echo "===== Check downloaded archives ====="

ls -lh "$DOWNLOAD_DIR"

validate_tgz() {
    local archive="$1"

    if [[ ! -s "$archive" ]]; then
        echo "ERROR: archive is missing or empty: $archive"
        exit 1
    fi

    if ! tar -tzf "$archive" >/dev/null 2>&1; then
        echo "ERROR: invalid or incomplete .tgz archive: $archive"
        echo "Google Drive may have returned an error page or the transfer may be incomplete."
        file "$archive" || true
        exit 1
    fi

    echo "Valid archive: $archive"
    file "$archive" || true
    du -h "$archive"
}

validate_tgz "$META_ARCHIVE"
validate_tgz "$BLOB01_ARCHIVE"

# Save complete archive listings. This avoids `tar | head` failures caused by
# SIGPIPE when the script is running with `set -o pipefail`.
tar -tzf "$META_ARCHIVE" > "$META_CONTENTS"
tar -tzf "$BLOB01_ARCHIVE" > "$BLOB01_CONTENTS"

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
    --batch_size 2 \
    --workers 1 \
    --epochs 20 \
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