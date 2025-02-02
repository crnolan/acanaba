# %%
import os
from pathlib import Path
import deeplabcut

# %%
project_path = Path(r'C:\Users\cnolan\Projects\acan\acanaba\acan2025')
# dlc_path = Path('/scratch/acan/acan2025/derivatives/dlc/acan-2025-tjb-2025-01-25')
dlc_path = project_path / 'derivatives/dlc/acan-cn-2025-01-28'
dlc_config = dlc_path / 'config.yaml'
superanimal_name = 'superanimal_topviewmouse'
model_name = 'hrnet_w32'
detector_name = 'fasterrcnn_resnet50_fpn_v2'
pseudo_checkpoint_path = 'derivatives/dlc/pseudo_checkpoints/pseudo_sub-itchy_ses-RR20.03_task-RR20_acq-A_vid/checkpoints'
video_root = Path(r'C:\Users\cnolan\Projects\acan\acanaba\acan2025\sourcedata\videos-cropped')
video_paths = [str(fn)
               for fn in video_root.glob('*.mp4')]

# %%
deeplabcut.extract_frames(
    dlc_config,
    algo='uniform',
    userfeedback=False,
    crop=False,
    videos_list=video_paths
)

# %%
