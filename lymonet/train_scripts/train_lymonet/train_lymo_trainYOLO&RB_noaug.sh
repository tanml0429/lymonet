
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd ../../ && pwd )"  # 上两级目录
echo $DIR
MODEL_SIZE='s'

python train.py \
    --model "${DIR}/configs/yolov8s_1MHSA_CA_RB.yaml" \
    --data "${DIR}/configs/lymo_mixed.yaml" \
    --weights '' \
    --epochs 300 \
    --device 1 \
    --batch 16 \
    --imgsz 640 \
    --patience 0 \
    --freeze "" \
    --box 7.5 \
    --cls 0.5 \
    --dfl 1.5 \
    --content_loss_gain 0.1 \
    --texture_loss_gain 0.1 \
    --augment_in_training False \
    --load_correspondence True \
    $@
