# %%
import pandas as pd
import subprocess
from pathlib import Path
import os
import re

# %%
project_path = Path(r'C:\Users\HP Demo\Documents\steymornejad')
trim_times = pd.read_csv(project_path / 'etc/trimtimes_box3.csv',
                         index_col=['filename', 'sub', 'ses', 'task', 'acq'])
crop_coords = pd.read_csv(project_path / 'etc/boxcrops_wallacewurth.csv')
crop_coords['w'] = crop_coords['x1'] - crop_coords['x0']
crop_coords['h'] = crop_coords['y1'] - crop_coords['y0']
in_path = Path(r'C:\Users\HP Demo\Documents\steymornejad\sourcedata\videos')
out_path = Path(r'C:\Users\HP Demo\Documents\steymornejad\sourcedata\videos')
filenames = list(in_path.glob('*.mp4'))
fn_pattern = r'group-([^_]+)_camera-overhead_(right|left)-(.*)\.mp4'

# %%
#
# First trim the videos
trimmed_fns = {}

for index, row in trim_times.iterrows():
    fn, sub, ses, task, acq = index[0:5]
    fn_path = Path(fn)
    print(fn, sub, ses, task, acq)
    fn_postfix = f"sub-{sub}_ses-{ses}_task-{task}_acq-{acq}_vidcropped"
    new_fn = in_path / (fn_path.stem + fn_postfix)
    trimmed_fns[index] = str(new_fn) + '.mp4'
    cmd = (f'ffmpeg -i "{in_path / fn}" -c copy -map 0 -f segment '
           f'-segment_times {row.trimstart},{row.trimend} '
           f'-reset_timestamps 1 '
           f'"{new_fn}%03d.mp4"')
    print('Running:', cmd)
    output = subprocess.run(cmd, shell=True, capture_output=True)
    os.rename(f"{new_fn}001.mp4",
              f"{new_fn}.mp4")
    os.remove(f"{new_fn}000.mp4")
    os.remove(f"{new_fn}002.mp4")

trim_times['trimmed_fn'] = pd.Series(trimmed_fns)
trim_times['trimmed_fn'].iloc[0]

# %%
#
# Now crop
new_fn_pattern = r'group-([^_]+)_camera-overhead_(right|left)_trimmed_(.*)'
for index, row in trim_times.iterrows():
    fn, sub, ses, task, acq = index[0:5]
    match = re.search(fn_pattern, index[0])
    # print(row["trimmed_fn"])
    # print(index[0])
    if match:
        box, camera, out_fn = match.groups()
        out_fn = f"sub-{sub}_ses-{ses}_task-{task}_acq-{acq}_vidcropped.mp4"
        box_crop = crop_coords.query(f'box == "{box}" and camera == "{camera}"').iloc[0]
        cmd = (f'ffmpeg -i "{row["trimmed_fn"]}" '
               f'-vf "crop={box_crop.w}:{box_crop.h}:{box_crop.x0}:{box_crop.x1}" '
               f'"{out_path / out_fn}"')
        print('Running:', cmd)
        output = subprocess.run(cmd, shell=True, capture_output=True)
        print(output.stderr.decode())

# %%
