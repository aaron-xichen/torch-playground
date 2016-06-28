EXP_NAME=try
PROJECT_NAME=quantization
FULL_FOLDER_PATH=project/$PROJECT_NAME/exps/$EXP_NAME
mkdir -p $FULL_FOLDER_PATH
LOG_PATH=$FULL_FOLDER_PATH/log
CUDA_VISIBLE_DEVICES=6 th main.lua -nThreads 4 -testOnly true -project $PROJECT_NAME -batchSize 48 -dataset imagenet -modelRoot /home/chenxi/modelzoo/bvlc_alexnet/ -weightNBits -1 -activationNFrac -1 -overFlowRate 0.01 -experimentsName $EXP_NAME 2>&1 |tee $LOG_PATH 
