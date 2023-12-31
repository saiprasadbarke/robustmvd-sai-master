#!/bin/bash

set -e

usage()
{
   echo "Usage: $0 -o [OPTIONAL] out_base -w [OPTIONAL] -g [OPTIONAL] gpu_list"
   echo -e "\t-o Path to output base directory. Optional. Default: /tmp/rmvd_train"
   echo -e "\t-w Enable wandb logging. Default: Do not enable wandb logging"
   echo -e "\t-g List of space-separated gpu numbers to launch train on (e.g. 0 2 4 5). Optional. Default: 0"
   echo -e "\tNote: the order of the arguments is important"
   exit 1 # Exit program after printing help
}

while getopts "o:wg" opt; do
    case "${opt}" in
        o )
          OUT_BASE=${OPTARG}
          ;;
        w )
          enable_wandb="--wandb"
          ;;
        g )
          gpu_list=${OPTARG}
          ;;
        ? ) usage ;;
    esac
done

if [ -z ${OUT_BASE} ]; then
    OUT_BASE=/tmp/rmvd_train
fi
echo Output base directory: ${OUT_BASE}

shift $((OPTIND-1))
GPU_IDX=("$@")
if [ -z ${GPU_IDX} ]; then
    GPU_IDX=(0)
fi
GPU_IDX_STR=$(printf ",%s" "${GPU_IDX[@]}")
GPU_IDX_STR=${GPU_IDX_STR:1}
NUM_GPUS=${#GPU_IDX[@]}
echo Using ${NUM_GPUS} GPUs with indices: ${GPU_IDX[@]}
export CUDA_VISIBLE_DEVICES=${GPU_IDX_STR}

echo

# robust_mvd model:
export MAX_ITER=600000
python train.py --training_type mvd --output ${OUT_BASE}/robust_mvd --num_gpus ${NUM_GPUS} --batch_size 4 --max_iterations $MAX_ITER --model robust_mvd --inputs poses intrinsics --optimizer adam --lr 1e-4 --grad_clip_max_norm 5 --scheduler flownet_scheduler --loss robust_mvd_loss --dataset staticthings3d.robust_mvd.mvd --dataset blendedmvs.robust_mvd.mvd --augmentations_per_dataset robust_mvd_augmentations_staticthings3d --augmentations_per_dataset robust_mvd_augmentations_blendedmvs --batch_augmentations robust_mvd_batch_augmentations --seed 42 ${enable_wandb}
python eval.py --eval_type robustmvd --model robust_mvd --weights ${OUT_BASE}/robust_mvd/weights_only_checkpoints_dir/snapshot-iter-$(printf %09d "$MAX_ITER").pt --inputs poses intrinsics --output ${OUT_BASE}/robust_mvd/eval/snapshot-iter-$(printf %09d "$MAX_ITER").pt --finished_iterations $MAX_ITER --log_dir ${OUT_BASE}/robust_mvd --num_gpus ${NUM_GPUS} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280 --eval_uncertainty --seed 42 ${enable_wandb}

# robust_mvd_5M model:
export MAX_ITER=5000000
python train.py --training_type mvd --output ${OUT_BASE}/robust_mvd_5M --num_gpus ${NUM_GPUS} --batch_size 4 --max_iterations $MAX_ITER --model robust_mvd_5M --inputs poses intrinsics --optimizer adam --lr 1e-4 --grad_clip_max_norm 5 --scheduler flownet_scheduler --loss robust_mvd_loss --dataset staticthings3d.robust_mvd.mvd --dataset blendedmvs.robust_mvd.mvd --augmentations_per_dataset robust_mvd_augmentations_staticthings3d --augmentations_per_dataset robust_mvd_augmentations_blendedmvs --batch_augmentations robust_mvd_batch_augmentations --seed 42 ${enable_wandb}
python eval.py --eval_type robustmvd --model robust_mvd_5M --weights ${OUT_BASE}/robust_mvd_5M/weights_only_checkpoints_dir/snapshot-iter-$(printf %09d "$MAX_ITER").pt --inputs poses intrinsics --output ${OUT_BASE}/robust_mvd_5M/eval/snapshot-iter-$(printf %09d "$MAX_ITER").pt --finished_iterations $MAX_ITER --log_dir ${OUT_BASE}/robust_mvd_5M --num_gpus ${NUM_GPUS} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280 --eval_uncertainty --seed 42 ${enable_wandb}

# mvsnet_blendedmvs model:
export MAX_ITER=160000
python train.py --training_type mvd --output ${OUT_BASE}/mvsnet_blendedmvs --num_gpus ${NUM_GPUS} --batch_size 1 --max_iterations $MAX_ITER --model mvsnet_blendedmvs --inputs poses intrinsics depth_range --optimizer rmsprop --lr 1e-3 --scheduler mvsnet_scheduler --loss mvsnet_loss --dataset blendedmvs.train_mvsnet.mvd --augmentations mvsnet_augmentations --seed 42 ${enable_wandb}
python eval.py --eval_type robustmvd --model mvsnet_blendedmvs --weights ${OUT_BASE}/mvsnet_blendedmvs/weights_only_checkpoints_dir/snapshot-iter-$(printf %09d "$MAX_ITER").pt --inputs poses intrinsics depth_range --output ${OUT_BASE}/mvsnet_blendedmvs/eval/snapshot-iter-$(printf %09d "$MAX_ITER").pt/known_depth_range --finished_iterations $MAX_ITER --eval_name known_depth_range --log_dir ${OUT_BASE}/mvsnet_blendedmvs --num_gpus ${NUM_GPUS} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280 --eval_uncertainty --seed 42 ${enable_wandb}

# supervised_monodepth2 model:
export MAX_ITER=64000
python train.py --training_type mvd --output ${OUT_BASE}/supervised_monodepth2 --num_gpus ${NUM_GPUS} --batch_size 12 --max_iterations $MAX_ITER --model supervised_monodepth2 --optimizer adam --lr 1e-4 --scheduler monodepth2_scheduler --loss supervised_monodepth2_loss --dataset kitti.eigen_dense_depth_train.mvd --augmentations supervised_monodepth2_augmentations --seed 42 ${enable_wandb}
python eval.py --eval_type robustmvd --model supervised_monodepth2 --weights ${OUT_BASE}/supervised_monodepth2/weights_only_checkpoints_dir/snapshot-iter-$(printf %09d "$MAX_ITER").pt --output ${OUT_BASE}/supervised_monodepth2/eval/snapshot-iter-$(printf %09d "$MAX_ITER").pt --finished_iterations $MAX_ITER --log_dir ${OUT_BASE}/supervised_monodepth2 --num_gpus ${NUM_GPUS} --eth3d_size 768 1152 --kitti_size 192 640 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280 --max_source_views 0 --seed 42 ${enable_wandb}
python eval.py --eval_type mvd --dataset kitti.eigen_dense_depth_test.mvd --model supervised_monodepth2 --weights ${OUT_BASE}/supervised_monodepth2/weights_only_checkpoints_dir/snapshot-iter-$(printf %09d "$MAX_ITER").pt --output ${OUT_BASE}/supervised_monodepth2/eval/snapshot-iter-$(printf %09d "$MAX_ITER").pt/kitti.eigen_dense_depth_test.mvd --finished_iterations $MAX_ITER --log_dir ${OUT_BASE}/supervised_monodepth2 --num_gpus ${NUM_GPUS} --input_size 192 640 --clipping 1e-3 80 --max_source_views 0 --seed 42 ${enable_wandb}
