export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
python3 distributedModelTraining.py --epochs 2 --lr 1e-3  --name BASELINE --recordCheckpoints 0 --checkpoint_path ./checkpoints/BASELINE_0.pt

# export LD_LIBRARY_PATH=/src/main/sweeping/random/build/lib/
# python3 distributedModelTraining.py --epochs 501 --lr 1e-3  --name RANDOM25

# export LD_LIBRARY_PATH=/src/main/sweeping/smallest/build/lib/
# python3 distributedModelTraining.py --epochs 501 --lr 1e-3  --name SMALLEST25


# --name = is the name of the trace file, without path since path is ./trace/ by default
# --recordCheckpoints = is boolean to tell the code to record checkpoints or not, 0 and 1
# --checkpoint_path = is the path of the checkpoint you want to load, don't add it if you don't want to load anything
#

