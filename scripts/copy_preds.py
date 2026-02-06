import subprocess
import random
import pathlib
import shlex
import os

REMOTE = "ekatasonov@cluster.hpc.hse.ru"
PORT = "2222"
KEY = "~/keys/auth_key.pem"

REMOTE_BASE = "/home/ekatasonov/vesuvius-challenge-surface-detection/data"
LOCAL_BASE = pathlib.Path("data")


def run_ssh(cmd):
    full_cmd = [
        "ssh", "-p", PORT, "-i", KEY, REMOTE, cmd
    ]
    out = subprocess.check_output(full_cmd, text=True)
    return out.strip().splitlines()


def scp_from(remote_path, local_dir):
    local_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "scp",
        "-P", PORT,
        "-i", KEY,
        f"{REMOTE}:{remote_path}",
        str(local_dir),
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        # ignore missing files
        pass


# 1️⃣ get predicted model directories
print("Listing predicted dirs...")
models = run_ssh(f"ls -d {REMOTE_BASE}/predicted/*")

chosen_model = random.choice(models)
model_name = pathlib.Path(chosen_model).name

print("Chosen model:", model_name)

# 2️⃣ pick 5 random tif files
# files = run_ssh(
#     f"ls {chosen_model}/val/*.tif | xargs -n1 basename"
# )

# files = random.sample(files, min(5, len(files)))
files = os.listdir("data/predicted/elastic/val")
print("Chosen files:", files)

# 3️⃣ get ALL predicted model dirs
all_models = models

for f in files:
    print("Processing:", f)

    # predicted folders
    for m in all_models:
        mname = pathlib.Path(m).name
        remote_file = f"{m}/val/{f}"
        local_dir = LOCAL_BASE / "predicted" / mname / "val"
        scp_from(remote_file, local_dir)

    # train_images
    scp_from(
        f"{REMOTE_BASE}/train_images/{f}",
        LOCAL_BASE / "train_images"
    )

    # train_labels
    scp_from(
        f"{REMOTE_BASE}/train_labels/{f}",
        LOCAL_BASE / "train_labels"
    )

    scp_from(
        f"{REMOTE_BASE}/train_skels/{f}",
        LOCAL_BASE / "train_labels"
    )

print("Done.")
