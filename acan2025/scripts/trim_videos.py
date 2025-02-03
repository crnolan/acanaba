# %%
import pandas as pd
import subprocess
from pathlib import Path
import os
import re

# %%
project_path = Path(r'C:\Users\cnolan\Projects\acanaba\acan2025')
trim_times = pd.read_csv(project_path / 'etc/trimtimes_RR20rev.01.csv',
                         index_col=['filename'])
crop_coords = pd.read_csv(project_path / 'etc/boxcrops_wallacewurth.csv')
crop_coords['w'] = crop_coords['x1'] - crop_coords['x0']
crop_coords['h'] = crop_coords['y1'] - crop_coords['y0']
in_path = Path(r'C:\Users\cnolan\UNSW\ACAN - ABA - ABA\staging\ACAN2025\RR20rev.01')
out_path = Path(r'C:\Users\cnolan\tmp\trim\RR20rev.01_trimmed')
filenames = list(in_path.glob('*.mp4'))
fn_pattern = r'group-([^_]+)_camera-overhead_(right|left)-.*'

# %%
#
# First trim the videos
for fn in filenames:
    tt = trim_times.loc[fn.name]
    cmd = (f'ffmpeg -i "{fn}" -c copy -map 0 -f segment '
           f'-segment_times {tt.trimstart},{tt.trimend} '
           f'-reset_timestamps 1 '
           f'"{out_path / fn.stem}_trimmed%03d.mp4"')
    print('Running:', cmd)
    output = subprocess.run(cmd, shell=True, capture_output=True)
    os.rename(f"{out_path / fn.stem}_trimmed001.mp4",
              f"{out_path / fn.stem}_trimmed.mp4")
    os.remove(f"{out_path / fn.stem}_trimmed000.mp4")
    os.remove(f"{out_path / fn.stem}_trimmed002.mp4")

# %%
#
# Now crop
for fn in filenames:
    match = re.search(fn_pattern, str(fn.name))
    if match:
        box, camera = match.groups()
        box_crop = crop_coords.query(f'box == "{box}" and camera == "{camera}"').iloc[0]
        cmd = (f'ffmpeg -i "{out_path / fn.stem}_trimmed.mp4" '
               f'-vf "crop={box_crop.w}:{box_crop.h}:{box_crop.x0}:{box_crop.x1}" '
               f'"{out_path / fn.stem}_cropped_trimmed.mp4"')
        print('Running:', cmd)
        output = subprocess.run(cmd, shell=True, capture_output=True)
        print(output.stderr.decode())

# %%
