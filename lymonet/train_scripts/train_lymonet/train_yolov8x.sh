
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd ../../ && pwd )"  # 上两级目录
echo $DIR
MODEL_SIZE='s'

python train.py \
    --model "${DIR}/configs/yolov8s_1MHSA_CA_RB.yaml" \
    --data "${DIR}/configs/lymo_mixed.yaml" \
    --epochs 300 \
    --device 3 \
    --batch 16 \
    --imgsz 640 \
    --patience 0 \
    --name "lymonet_with_RB" \
    $@
