EXP_NAME=try
PROJECT_NAME=huawei
FULL_FOLDER_PATH=project/$PROJECT_NAME/exps/$EXP_NAME
mkdir -p $FULL_FOLDER_PATH
LOG_PATH=$FULL_FOLDER_PATH/log
CUDA_VISIBLE_DEVICES=7 th main.lua -project huawei -dataset huawei -data /home/chenxi/dataset/huawei_scene_labeling -testOnly false -modelRoot /home/chenxi/modelzoo/vgg16_places205  -last 1 -weightDecay 0.01 -batchSize 80 -netType baseline -LR 0.01 -lrRatio 0.1 -nEpochs 100 -experimentsName $EXP_NAME 2>&1 |tee $LOG_PATH 
