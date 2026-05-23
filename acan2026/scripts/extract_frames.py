# %%
import os
from pathlib import Path
import deeplabcut

# %%
dlc_path = Path('/home/cnolan/Projects/dlc/medass_topviewmouse-cn-2025-08-07')
dlc_config = dlc_path / 'config.yaml'
superanimal_name = 'superanimal_topviewmouse'
model_name = 'hrnet_w32'
detector_name = 'fasterrcnn_resnet50_fpn_v2'
# pseudo_checkpoint_path = 'derivatives/dlc/pseudo_checkpoints/pseudo_sub-itchy_ses-RR20.03_task-RR20_acq-A_vid/checkpoints'
video_root = Path(r'/mnt/c/Users/cnolan/UNSW/ACAN-ACAN2026 - Documents/Modules/theme3_conditioning/rawdata')
video_paths = [str(fn)
               for fn in video_root.glob('**/*.mp4')]
video_paths

# %%
deeplabcut.add_new_videos(dlc_config, video_paths, copy_videos=False)

# %%
deeplabcut.extract_frames(
    dlc_config,
    algo='kmeans',
    userfeedback=False,
    crop=False,
    videos_list=video_paths,
)

# %%
