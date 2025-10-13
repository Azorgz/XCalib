wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_outdoor.pt -P backbone/Depth_anything/metric_depth/checkpoints
wget https://huggingface.co/Azorgz/XCalib2/resolve/main/checkpoint-10000.ckpt -P flow/CrossModalFlow/cross_raft_ckpt/model
wget https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -P backbone/ml_depth_pro/checkpoints
cd backbone/ml_depth_pro
pip install -e .
cd ../..
echo "Download complete."