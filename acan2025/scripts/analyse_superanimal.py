# %%
import deeplabcut
from pathlib import Path

# %%
project_path = Path('/scratch/acan/acan2025')
analysis_path = project_path / 'derivatives/dlc/analysed_superanimal'
video_root = project_path / 'sourcedata/videos'
video_paths = [str(fn)
               for fn in video_root.glob('**/*.mp4')
               if not (analysis_path / (fn.stem + 'DLC_Resnet50_umbrellaOct11shuffle2_snapshot_1000.h5')).exists()]
superanimal_name = 'superanimal_topviewmouse'
model_name = 'hrnet_w32'
detector_name = 'fasterrcnn_resnet50_fpn_v2'
# scale_list = range(80, 780, 50)

# %%
# deeplabcut.video_inference_superanimal(video_paths,
#                                        superanimal_name,
#                                        model_name='hrnet_w32',
#                                        detector_name='fasterrcnn_resnet50_fpn_v2',
#                                        video_adapt=True,
#                                        dest_folder=analysis_path,
#                                        batch_size=8,
#                                        detector_batch_size=8)


# %%
_ = deeplabcut.video_inference_superanimal(
    videos=video_paths[4:5],
    superanimal_name=superanimal_name,
    model_name=model_name,
    detector_name=detector_name,
    video_adapt=True,
    max_individuals=1,
    pseudo_threshold=0.1,
    bbox_threshold=0.9,
    detector_epochs=4,
    pose_epochs=4,
    dest_folder=analysis_path,
    batch_size=64,
    detector_batch_size=8
)

# %%
