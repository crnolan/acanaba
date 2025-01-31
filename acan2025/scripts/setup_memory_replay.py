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
project_path = Path('/scratch/acan/acan2025')
dlc_path = Path('/scratch/acan/acan2025/derivatives/dlc/acan-2025-tjb-2025-01-25')
dlc_config = dlc_path / 'config.yaml'
superanimal_name = 'superanimal_topviewmouse'
model_name = 'hrnet_w32'
detector_name = 'fasterrcnn_resnet50_fpn_v2'

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
    memory_replay=True,
)

deeplabcut.create_training_dataset(
    dlc_config,
    Shuffles=[1],
    engine=Engine.PYTORCH,
    net_type=f"top_down_{model_name}",
    detector_type=detector_name,
    weight_init=weight_init,
    userfeedback=False,
)

# %%
deeplabcut.train_network(
    dlc_config,
    detector_epochs=200,
    epochs=400,
    save_epochs=10,
    batch_size=64,  # if you get a CUDA OOM error when training on a GPU, reduce to 32, 16, ...!
    detector_batch_size=16,
    display_iters=10,
    shuffle=1
)

# %%
deeplabcut.evaluate_network(dlc_config,
                            Shuffles=[1],
                            plotting=True)

# %%
