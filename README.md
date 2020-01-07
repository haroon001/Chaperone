# Chaperone
An AI based system for bowlers to detect injury prone action in Cricket
Instructions:

1. Install Openpose.
2. Install ildoonet's tf-pose-estimation.
3. Replace "estimator.py" of tf-pose-estimation with estimator.py from this repo.
4. Use Chaperone_Video.py for angle detection in a video at every instant.
5. User Chaperone_realsense.py with Intel D-series cameras for more accurate 3d analysis. This is currently under-development and code is not complete.
6. Use python3 Chaperone.py --video_path '(PATH)'.
7. Only covers side angles.
