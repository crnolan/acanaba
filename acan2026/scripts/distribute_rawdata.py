# %%
from asyncio import subprocess
import os
import shutil
import pandas as pd
import subprocess

# ---------------------------------------------------------
# USER‑DEFINED PATHS
# ---------------------------------------------------------

# MAP_PATH = r"C:\Users\tburton\OneDrive - UNSW\ABA\staging\ACAN2026\exp-map.xlsx"
# STAGING_ROOT = r"C:\Users\tburton\OneDrive - UNSW\ABA\staging\ACAN2026"
# OUTPUT_ROOT = r"C:\Users\tburton\OneDrive - UNSW\ACAN-ACAN2026 - Documents\Modules\theme3_conditioning\rawdata"
MAP_PATH = r"/mnt/c/Users/cnolan/UNSW/ACAN-ABA - Documents/ABA/staging/ACAN2026/exp-map.xlsx"
STAGING_ROOT = r"/mnt/c/Users/cnolan/UNSW/ACAN-ABA - Documents/ABA/staging/ACAN2026"
OUTPUT_ROOT = r"/mnt/c/Users/cnolan/UNSW/ACAN-ACAN2026 - Documents/Modules/theme3_conditioning/rawdata"
#for testing:
#OUTPUT_ROOT = r"C:\Users\tburton\Projects\acan-aba-2026\organised"

# ---------------------------------------------------------
# LOAD EXCEL SHEETS
# ---------------------------------------------------------

counterbal = pd.read_excel(MAP_PATH, sheet_name="counterbal")
sched = pd.read_excel(MAP_PATH, sheet_name="sched")
trimtimes = pd.read_excel(MAP_PATH, sheet_name="trimtimes")

#sched["ses"] = sched["ses"].astype(int).astype(str).str.zfill(2)


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def camera_for_acq(pre_grn, acq, ses_name):
    # Determine whether this is a prerev or rev session
    ses_lower = str(ses_name).lower()
    is_prerev = "prerev" in ses_lower
    is_rev = ("rev" in ses_lower) and not is_prerev

    # Base mapping (prerev or normal)
    if pre_grn.upper() == "L":
        cam_A = "overhead_left"
        cam_B = "overhead_right"
    else:
        cam_A = "overhead_right"
        cam_B = "overhead_left"

    # Flip mapping for rev sessions
    if is_rev:
        cam_A, cam_B = cam_B, cam_A

    return cam_A if acq == "A" else cam_B


def acq_folder(acq):
    return acq.upper()


def find_video_file(search_dir, box, camera_side):
    if not os.path.isdir(search_dir):
        return None
    for f in os.listdir(search_dir):
        f_low = f.lower()
        if f"box{box}".lower() in f_low and camera_side.lower() in f_low:
            return os.path.join(search_dir, f)
    return None

# %%

# ---------------------------------------------------------
# MAIN PROCESSING LOOP FOR CREATING FOLDER STRUCTURE AND VIDEO ORGANISATION
# ---------------------------------------------------------

VIDEO_ROOT = os.path.join(STAGING_ROOT, "video")

for _, row in counterbal.iterrows():

    sub = row["sub"]
    squad = row["squad"]
    pre_grn = row["pre-grn"]
    box = str(row["box"])

    for _, srow in sched.iterrows():

        ses = srow["ses"]
        task = srow["task"]

        # Build output folder structure
        sub_folder = os.path.join(OUTPUT_ROOT, f"sub-{sub}")
        ses_folder = os.path.join(sub_folder, f"ses-{ses}")
        os.makedirs(ses_folder, exist_ok=True)

        # Determine squad folder
        # You may need to adjust this depending on your exact folder naming
        session_folder_name = f"{ses}"  # adjust if needed
        squad_folder = os.path.join(VIDEO_ROOT, session_folder_name, squad)

        for acq in ["A", "B"]:

            # Build output filename
            new_name = f"sub-{sub}_ses-{ses}_task-{task}_acq-{acq}.mp4"
            new_path = os.path.join(ses_folder, new_name)

            # SKIP if already processed
            if os.path.exists(new_path):
                print(f"Skipping existing file: {new_name}")
                continue

            # Determine acquisition folder
            acq_dir_name = acq_folder(acq)
            acq_dir = os.path.join(squad_folder, acq_dir_name)
            if not os.path.isdir(acq_dir):
                acq_dir = squad_folder  # fallback to squad folder if A/B not found

            # Determine camera side
            camera_side = camera_for_acq(pre_grn, acq, ses)

            # Get trim times for this session
            trim_row = trimtimes.query(f"session == '{ses}' and squad == '{squad}' and box == {box} and camera == '{camera_side}'")

            if trim_row.empty:
                print(f"No trim times found for {ses} {squad} box {box} camera {camera_side}, skipping video.")
                continue

            # Find matching video
            video_path = find_video_file(acq_dir, box, camera_side)

            if video_path is None:
                print(f"Missing: {sub}, {task}.{ses}, acq {acq} (waiting for data)")
                continue

            cmd = (f'ffmpeg -ss {trim_row.iloc[0].trimstart} '
                   f'-to {trim_row.iloc[0].trimend} '
                   f'-i "{video_path}" -c copy '
                   f'-reset_timestamps 1 '
                   f'"{new_path}"')
            # # Copy and rename
            # shutil.copy2(video_path, new_path)
            print(f"Running: {cmd}")
            output = subprocess.run(cmd, shell=True, capture_output=True)
            print(output.stderr.decode())
            print(f"Trimmed source video {video_path} → {new_path}")
            # print(f"Copied: {video_path} → {new_path}")

print("Incremental processing complete.")

# %%

# ---------------------------------------------------------
# PROCESS MED TEXT FILES (incremental + only scheduled sessions)
# ---------------------------------------------------------

MED_ROOT = os.path.join(STAGING_ROOT, "med")

# Only process sessions listed in sched["ses"]
valid_sessions = set(sched["ses"].astype(str).tolist())

for ses_name in os.listdir(MED_ROOT):
    ses_dir = os.path.join(MED_ROOT, ses_name)
    if not os.path.isdir(ses_dir):
        continue

    # Skip sessions not in the sched worksheet
    if ses_name not in valid_sessions:
        print(f"Skipping MED session not in sched: {ses_name}")
        continue

    # squads: S1, S2
    for squad in os.listdir(ses_dir):
        squad_dir = os.path.join(ses_dir, squad)
        if not os.path.isdir(squad_dir):
            continue

        # acquisitions: A, B
        for acq in ["A", "B"]:
            acq_dir = os.path.join(squad_dir, acq)
            if not os.path.isdir(acq_dir):
                continue

            # loop through all text files
            for fname in os.listdir(acq_dir):
                if not fname.lower().endswith(".txt"):
                    continue

                # Extract subject name from filename
                lower = fname.lower()
                if "subject" not in lower:
                    continue

                subj = lower.split("subject")[-1].replace(".txt", "").strip()
                subj = subj.lower()

                # Build output folder
                sub_folder = os.path.join(OUTPUT_ROOT, f"sub-{subj}")
                ses_folder = os.path.join(sub_folder, f"ses-{ses_name}")
                os.makedirs(ses_folder, exist_ok=True)

                # Build new filename
                task = ses_name.split(".")[0]  # e.g. RR10prerev
                new_name = f"sub-{subj}_ses-{ses_name}_task-{task}_acq-{acq}_rawmed.txt"
                new_path = os.path.join(ses_folder, new_name)

                # SKIP if already processed (incremental safety)
                if os.path.exists(new_path):
                    print(f"Skipping existing MED file: {new_name}")
                    continue

                # Copy
                src_path = os.path.join(acq_dir, fname)
                shutil.copy2(src_path, new_path)
                print(f"Copied MED: {src_path} → {new_path}")



# %%
