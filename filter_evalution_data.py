cd ~/OpenPCDet

python - <<'PY'
import pickle
from pathlib import Path

root = Path("data/nuscenes/v1.0-trainval")
val_pkl = root / "nuscenes_infos_10sweeps_val.pkl"
backup = root / "nuscenes_infos_10sweeps_val_full_backup.pkl"

data = pickle.load(open(val_pkl, "rb"))
infos = data["infos"] if isinstance(data, dict) and "infos" in data else data

def exists_path(path):
    if path is None:
        return False
    p = Path(path)
    if p.is_absolute():
        return p.exists()
    return (root / p).exists()

kept = []
missing = 0

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
        missing += 1

print("Original val samples:", len(infos))
print("Kept val samples:", len(kept))
print("Removed missing samples:", missing)

if not backup.exists():
    backup.write_bytes(val_pkl.read_bytes())
    print("Backup saved:", backup)

if isinstance(data, dict) and "infos" in data:
    data["infos"] = kept
    pickle.dump(data, open(val_pkl, "wb"))
else:
    pickle.dump(kept, open(val_pkl, "wb"))

print("Filtered val pkl written:", val_pkl)
PY