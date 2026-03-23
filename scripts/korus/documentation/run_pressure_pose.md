# Steps to Generate a Pressure-Pose Dataset

```bash
# running the script
preload ./isaaclab.sh -p ./scripts/korus/run_korus_bed_interactive_foam.py --num_envs 5 --env_spacing 10 --save_data --record_dir scripts/korus/korus_pressure_pose_dataset --pressure_every 550 

python3 scripts/korus/make_pressure_pose_dataset.py   --in_dir  scripts/korus/korus_pressure_pose_dataset/   --out_dir scripts/korus/korus_pressure_pose_dataset_final   --ts_decimals 2 

python scripts/korus/make_pressure_images.py --rows 8 --cols 4 --norm power --gamma 0.5 --in_dir scripts/korus/korus_pressure_pose_dataset_final --out_dir scripts/korus/korus_pressure_pose_dataset_images

# for reset
cd ~/IsaacLab/scripts/korus
rm -rf korus_pressure_pose_dataset/ korus_pressure_pose_dataset_final/ korus_pressure_pose_dataset_images/
```