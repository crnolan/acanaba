# %%
import deeplabcut
from pathlib import Path

# %%
project_path = Path('/scratch/tburton/acanaba')
config_path = project_path / 'derivatives/dlc/dlc_acan_master-tjb-2022-10-23/config.yaml'
analysis_path = project_path / 'derivatives/dlc/analysed_epoch1000'
video_root = project_path / 'sourcedata/videos'
video_paths = [str(fn)
               for fn in video_root.glob('*.mp4')
               if not (analysis_path / (fn.stem + 'DLC_Resnet50_dlc_acan_masterOct23shuffle2_snapshot_060.h5')).exists()]

# %%
deeplabcut.analyze_videos(config_path,
                          video_paths,
                          shuffle=2,
                          destfolder=analysis_path)

# %%
# video_paths = ['/scratch/cnolan/CSG010/derivatives/dlc/analysed/CSG010_032419_Hab3.mp4']
video_paths = [str(fn)
               for fn in video_root.glob('*.mp4')
               if not (analysis_path / (fn.stem + 'DLC_Resnet50_dlc_acan_masterOct23shuffle2_snapshot_060_filtered.h5')).exists()]
deeplabcut.filterpredictions(config_path, video_paths, shuffle=2,
                             destfolder=analysis_path)
# deeplabcut.plot_trajectories(config_path, video_paths)

video_paths = [str(fn)
               for fn in video_root.glob('*.mp4')
               if not (analysis_path / (fn.stem + 'DLC_Resnet50_dlc_acan_masterOct23shuffle2_snapshot_060_filtered_p10_labeled.mp4')).exists()]
deeplabcut.create_labeled_video(config_path, video_paths,
                                shuffle=2,
                                filtered=True, save_frames=False,
                                draw_skeleton=False,
                                destfolder=analysis_path)

# %%
