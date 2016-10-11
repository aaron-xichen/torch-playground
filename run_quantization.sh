EXP_NAME=try
PROJECT_NAME=quantization
FULL_FOLDER_PATH=project/$PROJECT_NAME/exps/$EXP_NAME
mkdir -p $FULL_FOLDER_PATH
LOG_PATH=$FULL_FOLDER_PATH/log
CUDA_VISIBLE_DEVICES=7 th main.lua -stopNSamples 10 -bitWidthConfigPath bitsSetting.config -metaTablePath meta.config -nThreads 4 -device cpu -testOnly true -project $PROJECT_NAME -batchSize 1 -dataset imagenet -data /work/shadow/ -modelRoot /home/chenxi/modelzoo/vgg16/ -experimentsName $EXP_NAME 2>&1 |tee $LOG_PATH 
