

python train.py \
    --model 'yolov8m.yaml' \
    --weights 'yolov8m.pt' \
    --epochs 300 \
    --device 6 \
    --batch 32 \
    --patience 0 \
    --name yolov8m \
    $@

