
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd ../../ && pwd )"  # 上两级目录
echo $DIR
MODEL_SIZE='s'

python train.py \
    --model "${DIR}/configs/yolov8s_1MHSA_CA_RB.yaml" \
    --data "${DIR}/configs/lymo_mixed.yaml" \
    --epochs 300 \
    --device 2 \
    --batch 16 \
    --imgsz 640 \
    --patience 0 \
    --content_loss_gain 0.0 \
    --texture_loss_gain 0.0 \
    --augment_in_training True \
    $@
