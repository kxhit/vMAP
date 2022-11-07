## Installation

```bash
make install
```

## Dataset
* live demo download ckpts from [Model Zoo](https://github.com/facebookresearch/Detic/blob/main/docs/MODEL_ZOO.md) and put into ./detector/Detic folder
* [Replica renders.](https://drive.google.com/drive/folders/1yK3ZTZtWzY5wA1g17lbblpOd3dAKBcJ4?usp=sharing)
* [Kinect recordings.](https://drive.google.com/drive/folders/1QgrPXTn_NAKrDRudbRlr4i7ixwzOfffn?usp=sharing)
* ScanNet
```bash
conda activate py2
python2 reader.py --filename ~/data/ScanNet/scannet/scans/scene0024_00/scene0024_00.sens --output_path ~/data/ScanNet/objnerf/ --export_depth_images --export_color_images --export_poses --export_intrinsics
```


## Usage

```bash
source .anaconda/bin/activate
cd ScenePriors/train/examples
export PYTHONPATH=$PYTHONPATH:~/code/HierarchicalPriors
```

Change dataset paths in *config_azure.json* for camera recording:
```json
"dataset": {
        "ims_file": "path/to/ims/folder/",
    }
```

Change dataset paths in *config_replica.json* for replica scene:
```json
"dataset": {
        "ims_file": "path/to/ims/folder/",
        "scene_file": "path/to/mesh.ply",
        "traj_file": "path/to/traj.txt",
    }
```

#### Offline demo
```bash
./train.py --config "config_file.json"
```
Flags:
* `--no_incremental`: Run with ground truth poses from selected images.

Keys:
* `-s`: Start and stop execution.

#### Online demo
```bash
./parallel_train.py --config "config_file.json"
```

It is preferable to run in two GPU setup, but can be run with single GPU (`--single_gpu`) with slower rendering framerate.

Flags:
* `--single_gpu`: Run on single GPU all porcesses.
* `--live`: Run from live kinect camera.
* `--do_sem`: Enable semantic labeling.

Keys:
* `-s`: Start execution (rgb_vis window).
* `-p`: Pause execution (rgb_vis window).
* `-m`: Generate/update mesh (iMAP window).
* `-f`: Enable/disable camera following in mesh visualisation (iMAP window).

#### Obj iMap
Online replica with 2D gt instances
```bash
./parallel_train.py --config config_replica_sem.json --do_obj --exp_name debug
```
Vis obj ckpt models mesh
```bash
python vis_imap_obj.py --config config_replica_sem.json -e 1899 -i 0
```


Offline replica with 2D gt instances with metrics and meshes
```bash
python replica_train.py --room_id room2
```

iMAP class train
```bash
python runiMap.py --room_id room0_iMAP_obj3_nk --config_file config_replica_sem.json --do_obj True
```

```bash
python run_iMap.py --config config_replica_sem.json --room_id room0_ilabel --do_obj True --do_sem True --ilabel True
```

Keys:
* `--sparse_sem`: use sparse sems from gt mask.
* `--do_track`: track cam in background model.
* `--do_track_obj`: track obj poses T_WO.
* `--live`: use kinect azure for live demo.


# basline
```bash
python replica_train_iMAP.py --config config_replica_room0_iMAP.json --room_id room0_None
```

```bash
python replica_train_iMAP.py --config config_replica_room0_iMAP.json --room_id room0_forloop32 --do_obj True --do_sem True --do_hull True
```

# obj
```bash
python paralell_run_iMap_process.py --config config_azure_ros.json --room_id orb_obj  --live True --do_obj True --do_sem True --do_hull True --active 0
```

# vmap live
```bash
source ~/catkin_ws/devel/setup.bash
roslaunch orb_slam3_ros_wrapper run_kinect.launch
python replica_train_bMAP_parallel.py --config config_azure_ros.json --room_id debug_live --do_sem True --do_obj True --live True
```


# for calib
```angular2html
go to
~/code/lib/Azure-Kinect-Sensor-SDK/examples/calibration/build
./calibration_1280
```

# ipad arkit
```bash
cd dockerfiles/rabbitmq
sudo docker build -t rabbitmq .
sudo docker run --rm -it --hostname my-rabbit -p 15672:15672 -p 5672:5672 --hostname my-rabbit rabbitmq
```

# ros catkin
```bash
catkin make -DPYTHON_EXECUTABLE=/usr/bin/python3
```

# scannet
```bash
python replica_train_bMAP_parallel.py --config config_scannet_iMAP.json --room_id scannet_b20_g256_h512 --do_obj True
```

# ROS bag
```bash
rosbag record -o chair_bottle /rgb/image_raw /depth_to_rgb/image_raw /imu
rosbag play chair_bottle_2022-07-24-14-17-36.bag -r 0.1
# for compressed img
rosrun image_transport republish compressed in:=/rgb/image_raw raw out:=/rgb/image_raw
```

# convert saved imgs to ros topic
```bash
python ros_pub.py --config configs/Dynamic/config_midfusion_chair_dyn_bMAP.json
```

# Run on server
```bash
ssh xk221@bigboy.doc.ic.ac.uk
```
## too many files opened
```bash
ulimit -n 1000000
```

## Send files
from Xin's PC to bigboy
```angular2html
scp -r /home/xin/data/ScanNet/NiceSLAM/ xk221@bigboy.doc.ic.ac.uk:~/data/ScanNet/NiceSLAM
```
from bigboy to Xin's PC
```angular2html
scp -r ./ScanNet/ xin@129.31.142.177:~/data/Results/
```