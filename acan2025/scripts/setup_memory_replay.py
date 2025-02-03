# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import deeplabcut
import deeplabcut.utils.auxiliaryfunctions as auxiliaryfunctions
from deeplabcut.pose_estimation_pytorch.apis import (
    superanimal_analyze_images,
)
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.core.engine import Engine
from deeplabcut.modelzoo import build_weight_init
from deeplabcut.modelzoo.utils import (
    create_conversion_table,
    read_conversion_table_from_csv,
)
from deeplabcut.modelzoo.video_inference import video_inference_superanimal
from deeplabcut.utils.pseudo_label import keypoint_matching

# %%
project_path = Path('/scratch/acan/acanaba/acan2025')
# dlc_path = Path('/scratch/acan/acan2025/derivatives/dlc/acan-2025-tjb-2025-01-25')
dlc_path = project_path / 'derivatives/dlc/acan-cn-2025-01-28'
dlc_config = dlc_path / 'config.yaml'
superanimal_name = 'superanimal_topviewmouse'
model_name = 'hrnet_w32'
detector_name = 'fasterrcnn_resnet50_fpn_v2'
pseudo_checkpoint_path = 'derivatives/dlc/pseudo_checkpoints/pseudo_sub-itchy_ses-RR20.03_task-RR20_acq-A_vid/checkpoints'

# %%
keypoint_matching(
    dlc_config,
    superanimal_name=superanimal_name,
    model_name=model_name,
    detector_name=detector_name,
    copy_images=True,
)

conversion_table_path = dlc_path / "memory_replay" / "conversion_table.csv"
confusion_matrix_path = dlc_path / "memory_replay" / "confusion_matrix.png"

# You can visualize the pseudo predictions, or do pose embedding clustering etc.
pseudo_prediction_path = dlc_path / "memory_replay" / "pseudo_predictions.json"

# %%
confusion_matrix_image = Image.open(confusion_matrix_path)

plt.imshow(confusion_matrix_image)
plt.axis('off')  # Hide the axes for better view
plt.show()

# %%
df = pd.read_csv(conversion_table_path)
df = df.dropna()

df

# %%
table = create_conversion_table(
    config=dlc_config,
    super_animal=superanimal_name,
    project_to_super_animal=read_conversion_table_from_csv(conversion_table_path),
)

# %%
# weight_init = WeightInitialization(
#     snapshot_path=dlc_path / 'dlc-models-pytorch/train',
#     detector_snapshot_path=dlc_path / 'dlc-models-pytorch/detector',
#     dataset=superanimal_name,
#     conversion_array=table.to_array(),
#     with_decoder=True,
#     memory_replay=True,
# )
weight_init = build_weight_init(
    cfg=auxiliaryfunctions.read_config(dlc_config),
    super_animal=superanimal_name,
    model_name=model_name,
    detector_name=detector_name,
    with_decoder=True,
    memory_replay=True
)
# weight_init = build_weight_init(
#     cfg=auxiliaryfunctions.read_config(dlc_config),
#     super_animal=superanimal_name,
#     model_name=model_name,
#     detector_name=detector_name,
#     with_decoder=True,
#     memory_replay=True,
#     customized_detector_checkpoint=project_path / pseudo_checkpoint_path / 'snapshot-fasterrcnn_resnet50_fpn_v2-004.pt',
#     customized_pose_checkpoint=project_path / pseudo_checkpoint_path / 'snapshot-hrnet_w32-004.pt'
# )

deeplabcut.create_training_dataset(
    dlc_config,
    Shuffles=[6],
    engine=Engine.PYTORCH,
    net_type=f"top_down_{model_name}",
    detector_type=detector_name,
    weight_init=weight_init,
    userfeedback=False,
)


# %%
