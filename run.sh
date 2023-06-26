#######################
#      Baselines      #
#######################

# RESNET50
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
# python3 distributedModelTraining_resnet50.py --gpus 4 --epochs 210 --lr 1e-2 --experiment baseline --name RESNET50 --recordCheckpoints=1

# VGG16
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
# python3 distributedModelTraining_vgg16.py --gpus 4 --epochs 201 --lr 1e-2  --experiment baseline --name VGG16_new 


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



# python3 resnet50_adaptive.py --gpus 4 --epochs 201 --lr 1e-2 --experiment Adaptive --name RESNET50_ADAPTIVE

#############################
#        VGG16 ADAPTIVE     #
#                           #
#############################


#in order: B.50, RS1.50, RS2.100

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
python3 vgg16_adaptive_chkpt.py --gpus 4 --epochs 51 --lr 1e-2 --experiment adapt_chkpt/inorder1 --name VGG16_STAGE1 --chkpt_dump ./checkpoints1/

export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X1/build/lib/
python3 vgg16_adaptive_chkpt.py --gpus 4 --epochs 51 --lr 1e-2 --experiment adapt_chkpt/inorder1 --name VGG16_STAGE2 --chkpt_dump ./checkpoints1/ --checkpoint ./checkpoints1/VGG16_STAGE1.pt

export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X2/build/lib/
python3 vgg16_adaptive_chkpt.py --gpus 4 --epochs 101 --lr 1e-2 --experiment adapt_chkpt/inorder1 --name VGG16_STAGE3 --chkpt_dump ./checkpoints1/ --checkpoint ./checkpoints1/VGG16_STAGE2.pt


#revorder: RS2.100, RS1.50, B.50

export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X2/build/lib/
python3 vgg16_adaptive_chkpt.py --gpus 4 --epochs 101 --lr 1e-2 --experiment adapt_chkpt/revorder1 --name VGG16_STAGE1 --chkpt_dump ./checkpoints2/

export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X1/build/lib/
python3 vgg16_adaptive_chkpt.py --gpus 4 --epochs 51 --lr 1e-2 --experiment adapt_chkpt/revorder1 --name VGG16_STAGE2 --chkpt_dump ./checkpoints2/ --checkpoint ./checkpoints2/VGG16_STAGE1.pt

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
python3 vgg16_adaptive_chkpt.py --gpus 4 --epochs 51 --lr 1e-2 --experiment adapt_chkpt/revorder1 --name VGG16_STAGE3 --chkpt_dump ./checkpoints2/ --checkpoint ./checkpoints2/VGG16_STAGE2.pt


#############################
#     RESNET50 ADAPTIVE     #
#                           #
#############################

#inorder: B.50, RS1.50, RS2.100
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
python3 resnet50_adaptive_chkpt.py --gpus 4 --epochs 51 --lr 1e-2 --experiment adapt_chkpt/inorder1 --name RESNET50_STAGE1 --chkpt_dump ./checkpoints1/

export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X1/build/lib/
python3 resnet50_adaptive_chkpt.py --gpus 4 --epochs 51 --lr 1e-2 --experiment adapt_chkpt/inorder1 --name RESNET50_STAGE2 --chkpt_dump ./checkpoints1/ --checkpoint ./checkpoints1/RESNET50_STAGE1.pt

export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X2/build/lib/
python3 resnet50_adaptive_chkpt.py --gpus 4 --epochs 101 --lr 1e-2 --experiment adapt_chkpt/inorder1 --name RESNET50_STAGE3 --chkpt_dump ./checkpoints1/ --checkpoint ./checkpoints1/RESNET50_STAGE2.pt


#revorder: RS2.100, RS1.50, B.50
export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X2/build/lib/
python3 resnet50_adaptive_chkpt.py --gpus 4 --epochs 101 --lr 1e-2 --experiment adapt_chkpt/revorder1 --name RESNET50_STAGE1 --chkpt_dump ./checkpoints2/

export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X1/build/lib/
python3 resnet50_adaptive_chkpt.py --gpus 4 --epochs 51 --lr 1e-2 --experiment adapt_chkpt/revorder1 --name RESNET50_STAGE2 --chkpt_dump ./checkpoints2/ --checkpoint ./checkpoints2/RESNET50_STAGE1.pt

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/ 
python3 resnet50_adaptive_chkpt.py --gpus 4 --epochs 51 --lr 1e-2 --experiment adapt_chkpt/revorder1 --name RESNET50_STAGE3 --chkpt_dump ./checkpoints2/ --checkpoint ./checkpoints2/RESNET50_STAGE2.pt

