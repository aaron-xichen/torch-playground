EXP_NAME=try
PROJECT_NAME=quantization
FULL_FOLDER_PATH=project/$PROJECT_NAME/exps/$EXP_NAME
mkdir -p $FULL_FOLDER_PATH
LOG_PATH=$FULL_FOLDER_PATH/log
CUDA_VISIBLE_DEVICES=0 th main.lua -nThreads 3 -device gpu -testOnly true -project $PROJECT_NAME -batchSize 50 -dataset imagenet -data /home/chenxi/dataset/shadow/ -modelRoot /home/chenxi/modelzoo/resnet152/ -tensorType float -convNBits 8 -fcNBits 8 -actNBits 8 -isQuantizeBN false -experimentsName $EXP_NAME 2>&1 |tee $LOG_PATH 
