

python train.py \
    --model 'yolov8l.yaml' \
    --weights 'yolov8l.pt' \
    --epochs 300 \
    --device 5 \
    --batch 32 \
    --patience 0 \
    --name yolov8l \
    $@
