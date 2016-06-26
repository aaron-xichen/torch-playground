EXP_NAME=try
PROJECT_NAME=dssm
FULL_FOLDER_PATH=project/$PROJECT_NAME/exps/$EXP_NAME
mkdir -p $FULL_FOLDER_PATH
LOG_PATH=$FULL_FOLDER_PATH/log
CUDA_VISIBLE_DEVICES=6 th main.lua -testOnly false -project dssm -nBits 16 -momentum 0 -nesterov false  -activation ReLU -LR 0.1 -nEpochs 20 -batchSize 1024 -experimentsName $EXP_NAME 2>&1 |tee $LOG_PATH 
