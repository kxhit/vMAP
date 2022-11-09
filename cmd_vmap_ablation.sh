#! /bin/bash

#exp_list=("h16" "h32" "h64" "h128" "h256" "h512")
#exp_list=("h4" "h8" "h64")
#exp_list=("h256")

exp_list=("h4" "h8" "h16" "h32" "h64" "h128" "h256")

for exp in "${exp_list[@]}"
do
  python train_live.py --config ./configs/Replica/ablation/config_replica_room0_bMAP_"$exp".json --logdir ./logs/ablation/bmap_"$exp" > log_"$exp".txt
done
