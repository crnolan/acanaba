# %%
from pathlib import Path
import deeplabcut
import deeplabcut.utils.auxiliaryfunctions as auxiliaryfunctions
from deeplabcut.modelzoo import build_weight_init
from deeplabcut.core.engine import Engine

# %%
dlc_path = Path('/home/cnolan/Projects/medass_topviewmouse-cn-2025-08-07')
dlc_config = dlc_path / 'config.yaml'
superanimal_name = 'superanimal_topviewmouse'
model_name = 'hrnet_w32'
detector_name = 'fasterrcnn_resnet50_fpn_v2'

# %%
weight_init = build_weight_init(
    cfg=auxiliaryfunctions.read_config(dlc_config),
    super_animal=superanimal_name,
    model_name=model_name,
    detector_name=detector_name,
    with_decoder=True,
    memory_replay=True
)

# %%
deeplabcut.create_training_dataset(
    dlc_config,
    Shuffles=[3],
    engine=Engine.PYTORCH,
    net_type=f"top_down_{model_name}",
    detector_type=detector_name,
    weight_init=weight_init,
    userfeedback=False,
)

# %%
deeplabcut.train_network(
    dlc_config,
    # detector_epochs=400,
    # epochs=400,
    # save_epochs=10,
    batch_size=64,  # if you get a CUDA OOM error when training on a GPU, reduce to 32, 16, ...!
    detector_batch_size=8,
    # display_iters=10,
    shuffle=3
)

# %%
deeplabcut.evaluate_network(dlc_config,
                            Shuffles=[6],
                            plotting=True)


# %%
