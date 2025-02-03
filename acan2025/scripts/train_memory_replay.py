# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import deeplabcut

# %%
project_path = Path('/scratch/acan/acanaba/acan2025')
dlc_path = project_path / 'derivatives/dlc/acan-cn-2025-01-28'
dlc_config = dlc_path / 'config.yaml'
superanimal_name = 'superanimal_topviewmouse'
model_name = 'hrnet_w32'
detector_name = 'fasterrcnn_resnet50_fpn_v2'

# %%
deeplabcut.train_network(
    dlc_config,
    detector_epochs=400,
    epochs=400,
    save_epochs=10,
    batch_size=64,  # if you get a CUDA OOM error when training on a GPU, reduce to 32, 16, ...!
    detector_batch_size=8,
    display_iters=10,
    shuffle=6
)

# %%
deeplabcut.evaluate_network(dlc_config,
                            Shuffles=[6],
                            plotting=True)


# %%
