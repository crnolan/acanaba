# %%
from pathlib import Path
import deeplabcut

# %%
project_path = Path(r'C:\Users\cnolan\Development\acan2025')
dlc_path = Path(r'C:\Users\cnolan\Development\acan2025\derivatives\dlc')
superanimal_name = 'superanimal_topviewmouse'
model_name = 'hrnet_w32'
detector_name = 'fasterrcnn_resnet50_fpn_v2'
videos = [str(p) for p in Path('/scratch/acan/acan2025/sourcedata/videos').glob('**/*.mp4')]

# %%
deeplabcut.create_pretrained_project(
    project="acan-2025",
    experimenter="cn",
    model=superanimal_name,
    working_directory=dlc_path,
    analyzevideo=False,
    videos=videos
)
# %%
