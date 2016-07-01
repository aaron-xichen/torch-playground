EXP_NAME=try
PROJECT_NAME=quantization
FULL_FOLDER_PATH=project/$PROJECT_NAME/exps/$EXP_NAME
mkdir -p $FULL_FOLDER_PATH
LOG_PATH=$FULL_FOLDER_PATH/log
CUDA_VISIBLE_DEVICES=6 th main.lua  -actScale 1 -nThreads 5 -testOnly true -project $PROJECT_NAME -batchSize 50 -dataset imagenet -modelRoot /home/chenxi/modelzoo/vgg19/ -convNBits 8 -fcNBits 8 -actNBits 8 -experimentsName $EXP_NAME 2>&1 |tee $LOG_PATH 
