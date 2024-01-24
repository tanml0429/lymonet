

python train.py \
    --model 'yolov8x.yaml' \
    --weights 'yolov8x.pt' \
    --epochs 300 \
    --device 7 \
    --batch 32 \
    --patience 0 \
    --name yolov8x \
    $@
