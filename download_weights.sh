cd model
wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_outdoor.pt -P backbone/ZoeDepth/metric_depth/checkpoints
wget https://huggingface.co/Azorgz/XCalib/resolve/main/checkpoint-10000.ckpt -P flow/CrossModalFlow/cross_raft_ckpt/model
#wget https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -P backbone/ml_depth_pro/checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth?download=true -P /backbone/Depth_Anything_V2/metric_depth/checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/depth_anything_v2_metric_vkitti_vits.pth?download=true -P /backbone/Depth_Anything_V2/metric_depth/checkpoints
#cd backbone/ml_depth_pro
#pip install -e .
cd ..
echo "Download complete."