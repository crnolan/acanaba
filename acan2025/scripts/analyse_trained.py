# %%
import deeplabcut
from pathlib import Path

# %%
project_path = Path('/scratch/acan/acanaba/acan2025')
analysis_path = project_path / 'derivatives/dlc/analysed_mr_shuffle6'
dlc_path = project_path / 'derivatives/dlc/acan-cn-2025-01-28'
dlc_config = dlc_path / 'config.yaml'
video_root = project_path / 'sourcedata/videos-cropped-trimmed'
video_glob = '**/*trimmed.mp4'
video_paths = [str(fn)
               for fn in video_root.glob(video_glob)
               if not (analysis_path / (fn.stem + 'DLC_HrnetW32_acanJan28shuffle5_detector_360_snapshot_110.h5')).exists()]
superanimal_name = 'superanimal_topviewmouse'
model_name = 'hrnet_w32'
detector_name = 'fasterrcnn_resnet50_fpn_v2'

# %%
deeplabcut.analyze_videos(
    dlc_config,
    videos=video_paths,
    shuffle=6,
    destfolder=analysis_path,
    device='cuda:0')

# %%
video_paths = [str(fn)
               for fn in video_root.glob(video_glob)
               if not (analysis_path / (fn.stem + 'DLC_HrnetW32_acanJan28shuffle5_detector_360_snapshot_110_filtered.h5')).exists()]
deeplabcut.filterpredictions(dlc_config,
                             video_paths,
                             shuffle=6,
                             destfolder=analysis_path)
# deeplabcut.plot_trajectories(config_path, video_paths)

video_paths = [str(fn)
               for fn in video_root.glob(video_glob)
               if not (analysis_path / (fn.stem + 'DLC_HrnetW32_acanJan28shuffle5_detector_360_snapshot_110_filtered_p60_labeled.mp4')).exists()]
deeplabcut.create_labeled_video(dlc_config,
                                video_paths,
                                shuffle=6,
                                filtered=True,
                                save_frames=False,
                                draw_skeleton=False,
                                destfolder=analysis_path)

# %%
