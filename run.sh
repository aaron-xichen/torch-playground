EXP_NAME=try
PROJECT_NAME=huawei
FULL_FOLDER_PATH=project/$PROJECT_NAME/exps/$EXP_NAME
mkdir -p $FULL_FOLDER_PATH
LOG_PATH=$FULL_FOLDER_PATH/log
CUDA_VISIBLE_DEVICES=6 th main.lua -project huawei -dataset huawei -testOnly false -modelRoot /home/chenxi/modelzoo/siat_scene_vgg_13/  -externelMean /home/chenxi/modelzoo/places205_mean.t7  -batchSize 32 -netType baseline -LR 0.01 -lrRatio 0.01 -nEpochs 2 -experimentsName $EXP_NAME 2>&1 |tee $LOG_PATH 
