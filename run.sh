#######################
#      Baselines      #
#######################

## RESNET50
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
# python3 distributedModelTraining_resnet50.py --gpus 4 --epochs 201 --lr 1e-2 --experiment baseline --name RESNET50 

## VGG16
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
# python3 distributedModelTraining_vgg16.py --gpus 4 --epochs 201 --lr 1e-2  --experiment baseline --name VGG16 


########################
#  Resnet50 Random     #
#                      #
########################


# export LD_LIBRARY_PATH=/src/main/modified_nccl/random/25/build/lib/
# python3 distributedModelTraining_resnet50.py --gpus 4 --epochs 201 --lr 1e-2 --experiment RandomPruning --name RESNET50_25

# export LD_LIBRARY_PATH=/src/main/modified_nccl/random/50/build/lib/
# python3 distributedModelTraining_resnet50.py --gpus 4 --epochs 201 --lr 1e-2 --experiment RandomPruning --name RESNET50_50

# export LD_LIBRARY_PATH=/src/main/modified_nccl/random/75/build/lib/
# python3 distributedModelTraining_resnet50.py --gpus 4 --epochs 201 --lr 1e-2 --experiment RandomPruning --name RESNET50_75

# export LD_LIBRARY_PATH=/src/main/modified_nccl/random/90/build/lib/
# python3 distributedModelTraining_resnet50.py --gpus 4 --epochs 201 --lr 1e-2 --experiment RandomPruning --name RESNET50_90


########################
#     VGG16 Random     #
#                      #
########################


# export LD_LIBRARY_PATH=/src/main/modified_nccl/random/25/build/lib/
# python3 distributedModelTraining_vgg16.py --gpus 4 --epochs 201 --lr 1e-2 --experiment RandomPruning --name VGG16_25

# export LD_LIBRARY_PATH=/src/main/modified_nccl/random/50/build/lib/
# python3 distributedModelTraining_vgg16.py --gpus 4 --epochs 201 --lr 1e-2 --experiment RandomPruning --name VGG16_50

# export LD_LIBRARY_PATH=/src/main/modified_nccl/random/75/build/lib/
# python3 distributedModelTraining_vgg16.py --gpus 4 --epochs 201 --lr 1e-2 --experiment RandomPruning --name VGG16_75

# export LD_LIBRARY_PATH=/src/main/modified_nccl/random/90/build/lib/
# python3 distributedModelTraining_vgg16.py --gpus 4 --epochs 201 --lr 1e-2 --experiment RandomPruning --name VGG16_90




########################
#  Resnet50 SkipReduce #
#                      #
########################

# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X1/build/lib/
# python3 distributedModelTraining_resnet50.py --gpus 4 --epochs 201 --lr 1e-2 --experiment SkipReduce --name RESNET50_X1

# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X2/build/lib/
# python3 distributedModelTraining_resnet50.py --gpus 4 --epochs 201 --lr 1e-2 --experiment SkipReduce --name RESNET50_X2

# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X1Y1/build/lib/
# python3 distributedModelTraining_resnet50.py --gpus 4 --epochs 201 --lr 1e-2 --experiment SkipReduce --name RESNET50_X1Y1



#####################
#  VGG16 SkipReduce #
#                   #
#####################

# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X1/build/lib/
# python3 distributedModelTraining_vgg16.py --gpus 4 --epochs 201 --lr 1e-2 --experiment SkipReduce --name VGG16_X1

# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X2/build/lib/
# python3 distributedModelTraining_vgg16.py --gpus 4 --epochs 201 --lr 1e-2 --experiment SkipReduce --name VGG16_X2

# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X1Y1/build/lib/
# python3 distributedModelTraining_vgg16.py --gpus 4 --epochs 201 --lr 1e-2 --experiment SkipReduce --name VGG16_X1Y1




####################
# Kernel Profiling #
#                  #
####################

# #RESNET50

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
nvprof --devices 0 --concurrent-kernels on --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./csv/RESNET50_%p.csv python3 resnet50_timing.py --gpus 4 --epochs 1 --lr 0.01  --name resnet50

export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X1/build/lib/
nvprof --devices 0 --concurrent-kernels on --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./csv/RESNET50_X1_%p.csv python3 resnet50_timing.py --gpus 4 --epochs 1 --lr 0.01  --name resnet50_x1

export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X2/build/lib/
nvprof --devices 0 --concurrent-kernels on --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./csv/RESNET50_X2_%p.csv python3 resnet50_timing.py --gpus 4 --epochs 1 --lr 0.01  --name resnet50_x2

export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X1Y1/build/lib/
nvprof --devices 0 --concurrent-kernels on --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./csv/RESNET50_X1Y1_%p.csv python3 resnet50_timing.py --gpus 4 --epochs 1 --lr 0.01  --name resnet50_x1y2

# #VGG16

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
nvprof --devices 0 --concurrent-kernels on --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./csv/VGG16_%p.csv python3 vgg16_timing.py --gpus 4 --epochs 1 --lr 0.01  --name vgg16

export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X1/build/lib/
nvprof --devices 0 --concurrent-kernels on --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./csv/VGG16_X1_%p.csv python3 vgg16_timing.py --gpus 4 --epochs 1 --lr 0.01  --name vgg16_x1

export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X2/build/lib/
nvprof --devices 0 --concurrent-kernels on --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./csv/VGG16_X2_%p.csv python3 vgg16_timing.py --gpus 4 --epochs 1 --lr 0.01  --name vgg16_x2

export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X1Y1/build/lib/
nvprof --devices 0 --concurrent-kernels on --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./csv/VGG16_X1Y1%p.csv python3 vgg16_timing.py --gpus 4 --epochs 1 --lr 0.01  --name vgg16_x1y1


#####################
#    CUDA TIMING    #
#####################

# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
# python3 resnet50_timing.py --gpus 4 --epochs 201 --lr 1e-2 --experiment null --name resnet50 

# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X1/build/lib/
# python3 resnet50_timing.py --gpus 4 --epochs 201 --lr 1e-2 --experiment null --name resnet50_X1 

# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X2/build/lib/
# python3 resnet50_timing.py --gpus 4 --epochs 201 --lr 1e-2 --experiment null --name resnet50_X2

# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X1Y1/build/lib/
# python3 resnet50_timing.py --gpus 4 --epochs 201 --lr 1e-2 --experiment null --name resnet50_X1Y1

# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
# python3 vgg16_timing.py --gpus 4 --epochs 201 --lr 1e-2 --experiment null --name vgg16 

# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X1/build/lib/
# python3 vgg16_timing.py --gpus 4 --epochs 201 --lr 1e-2 --experiment null --name vgg16_X1 

# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X2/build/lib/
# python3 vgg16_timing.py --gpus 4 --epochs 201 --lr 1e-2 --experiment null --name vgg16_X2

# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X1Y1/build/lib/
# python3 vgg16_timing.py --gpus 4 --epochs 201 --lr 1e-2 --experiment null --name vgg16_X1Y1