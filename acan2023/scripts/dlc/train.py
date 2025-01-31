import deeplabcut
from pathlib import Path

# %%
project_path = Path('/scratch/tburton/acanaba')
config_path = project_path / 'derivatives/dlc/dlc_acan_master-tjb-2022-10-23/config.yaml'
analysis_path = project_path / 'derivatives/dlc/analysed'

# %%
# deeplabcut.create_training_dataset(config_path, windows2linux=True, augmenter_type='imgaug')
# deeplabcut.create_training_dataset(config_path, net_type='top_down_resnet_50')

# %%
deeplabcut.train_network(config_path, shuffle=2)

# %%
deeplabcut.evaluate_network(config_path, Shuffles=[2], plotting=True, show_errors=True)

# %%
