export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
python3 distributedModelTraining.py --epochs 300 --lr 1e-2 --datatype F32 --name baseline

# export LD_LIBRARY_PATH=/src/main/KimSum/build/lib/
# python3 distributedModelTraining.py --epochs 51 --lr 1e-2 --datatype F32 --name 1DeviceDropped


# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
# nvprof --devices 1 --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./timing/baseline/VGG16_timingTrace%p.csv python3 timingProfiling.py --epochs 51 --lr 1e-2 --name baseline


# export LD_LIBRARY_PATH=/src/main/KimSum/build/lib/
# nvprof --devices 1 --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./timing/modified/VGG16_timingTrace%p.csv python3 timingProfiling.py --epochs 51 --lr 1e-2 --name modified

# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
# python3 timingProfiling.py --epochs 51 --lr 1e-2 --name baseline


# export LD_LIBRARY_PATH=/src/main/KimSum/build/lib/
# python3 timingProfiling.py --epochs 51 --lr 1e-2 --name modified