EXP_NAME=try
PROJECT_NAME=template
FULL_FOLDER_PATH=project/$PROJECT_NAME/exps/$EXP_NAME
mkdir -p $FULL_FOLDER_PATH
LOG_PATH=$FULL_FOLDER_PATH/log
CUDA_VISIBLE_DEVICES=6 th main.lua -LR 0.001 -project $PROJECT_NAME -batchSize 100 -dataset imagenet -nUnit 1 -nEpochs 2 -experimentsName $EXP_NAME 2>&1 |tee $LOG_PATH 
