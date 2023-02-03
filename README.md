[comment]: <> (# vMAP: Vectorised Object Mapping for Neural Field SLAM)

<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">vMAP: Vectorised Object Mapping for Neural Field SLAM</h1>
  <p align="center">
    <a href="https://kxhit.github.io"><strong>Xin Kong</strong></a>
    ·
    <a href="https://shikun.io"><strong>Shikun Liu</strong></a>
    ·
    <a href="https://marwan99.github.io/"><strong>Marwan Taher</strong></a>
    ·
    <a href="https://www.doc.ic.ac.uk/~ajd/"><strong>Andrew Davison</strong></a>
  </p>

[comment]: <> (  <h2 align="center">arXiv</h2>)
  <h3 align="center"><a href="https://arxiv.org/abs/TODO">arXiv</a> | Video | <a href="https://kxhit.github.io/vMAP">Project Page</a></h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="./media/teaser.png" alt="Logo" width="80%">
  </a>
</p>
<p align="center">
vMAP builds an object-level map from a real-time RGB-D input stream. Each object is represented by a separate MLP neural field model, all optimised in parallel via vectorised training. 
</p>
<br>



<br>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#Install">Installation</a>
    </li>
    <li>
      <a href="#Run">Run</a>
    </li>
    <li>
      <a href="#Evaluation">Evaluation</a>
    </li>
    <li>
      <a href="#Acknowledgement">Acknowledgement</a>
    </li>
    <li>
      <a href="#Citation">Citation</a>
    </li>
  </ol>
</details>

This repo contains implementation of vMAP (official) and a simplified / improved iMAP* (non-official).
## Install
```bash
conda env create -f environment.yml
```

## Dataset
* [Replica Demo](https://huggingface.co/datasets/kxic/vMAP/resolve/main/demo_replica_room_0.zip)
* [Replica](https://huggingface.co/datasets/kxic/vMAP/resolve/main/vmap.zip)
* [ScanNet](https://github.com/ScanNet/ScanNet)
```bash
conda activate py2
python2 reader.py --filename ~/data/ScanNet/scannet/scans/scene0024_00/scene0024_00.sens --output_path ~/data/ScanNet/objnerf/ --export_depth_images --export_color_images --export_poses --export_intrinsics
```

## Config
Change dataset paths in *configs/.json*:
```json
"dataset": {
        "path": "path/to/ims/folder/",
    }
```

## Run

[comment]: <> (### Single thread demo)
#### vMAP
```bash
./train.py --config ./configs/Replica/config_replica_room0_vMAP.json --logdir ./logs/vMAP/room0 --save_ckpt True
```
#### iMAP*
```bash
./train.py --config ./configs/Replica/config_replica_room0_iMAP.json --logdir ./logs/iMAP/room0 --save_ckpt True
```

[comment]: <> (#### Multi thread demo)

[comment]: <> (```bash)

[comment]: <> (./parallel_train.py --config "config_file.json" --logdir ./logs)

[comment]: <> (```)

## Evaluation
### Reconstruction Error
#### 3D Scene-level Eval
Same metrics follow iMAP, compare against GT scene mesh
```bash
./metric/eval_3D_scene.py
```
#### 3D Object-level Eval
We find scene-level metrics are biased by background mesh, so we propose object-level metrics by evaluating every object in each scene and then average.
```bash
./metric/eval_3D_obj.py
```

[comment]: <> (### Novel View Synthesis)

[comment]: <> (##### 2D Novel View Eval)

[comment]: <> (We rendered a new trajectory in each scene and randomly choose novel view pose from it, evaluating 2D rendering performance)

[comment]: <> (```bash)

[comment]: <> (./metric/eval_2D_view.py)

[comment]: <> (```)

## Acknowledgement
We thank great open-sourced repos: [NICE-SLAM](https://github.com/cvg/nice-slam), and [Functorch](https://github.com/pytorch/functorch).

## Citation
If you find our code or paper useful, please consider cite
```bibtex
@inproceedings{sucar2021imap,
  title={iMAP: Implicit mapping and positioning in real-time},
  author={Sucar, Edgar and Liu, Shikun and Ortiz, Joseph and Davison, Andrew J},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6229--6238},
  year={2021}
}

@article{kong2023vmap,
  title={vMAP: Vectorised Object Mapping for Neural Field SLAM},
  author={Kong, Xin and Liu, Shikun and Taher, Marwan and Davison, Andrew J},
  journal={arXiv preprint arXiv:TODO},
  year={2023}
}
```