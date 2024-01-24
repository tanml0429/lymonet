
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd ../../ && pwd )"  # 上两级目录
echo $DIR
MODEL_SIZE='s'

python train.py \
    --model "${DIR}/configs/yolov8s_1MHSA_CA_RB.yaml" \
    --data "${DIR}/configs/lymo_mixed.yaml" \
    --epochs 300 \
    --device 1 \
    --batch 16 \
    --imgsz 640 \
    --patience 0 \
    --freeze "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22" \
    --box 0.0 \
    --cls 0.0 \
    --dfl 0.0 \
    --content_loss_gain 1.0 \
    --texture_loss_gain 1.0 \
    --augment_in_training False \
    $@
