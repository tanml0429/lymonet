
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd ../../ && pwd )"  # 上两级目录
echo $DIR
MODEL_SIZE='x'

python train.py \
    --model "yolov8${MODEL_SIZE}.yaml" \
    --weights "yolov8${MODEL_SIZE}.pt" \
    --data "${DIR}/configs/lymo_mixed_clipped.yaml" \
    --epochs 300 \
    --device 4 \
    --batch 32 \
    --patience 0 \
    --name "yolov8${MODEL_SIZE}_clipped" \
    $@
