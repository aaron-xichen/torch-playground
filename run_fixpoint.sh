EXP_NAME=tryint32
PROJECT_NAME=fixpoint
FULL_FOLDER_PATH=project/$PROJECT_NAME/exps/$EXP_NAME
mkdir -p $FULL_FOLDER_PATH
LOG_PATH=$FULL_FOLDER_PATH/log
CUDA_VISIBLE_DEVICES=5 th main.lua -device cpu -nThreads 1 -testOnly true -project $PROJECT_NAME -batchSize 50 -dataset imagenet -modelRoot /home/chenxi/modelzoo/vgg16/ -shiftInfoPath /home/chenxi/experiments/platform/project/fixpoint/models/vgg16ShiftTable3.t7 -experimentsName $EXP_NAME 2>&1 |tee $LOG_PATH 