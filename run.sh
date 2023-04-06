export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
python3 distributedModelTraining.py --epochs 51 --lr 1e-2 --datatype F32 --name baseline

export LD_LIBRARY_PATH=/src/main/KimSum/build/lib/
python3 distributedModelTraining.py --epochs 51 --lr 1e-2 --datatype F32 --name 1DeviceDropped
