EXP_NAME=try
PROJECT_NAME=quantization
FULL_FOLDER_PATH=project/$PROJECT_NAME/exps/$EXP_NAME
mkdir -p $FULL_FOLDER_PATH
LOG_PATH=$FULL_FOLDER_PATH/log
CUDA_VISIBLE_DEVICES=4 th main.lua -adderMaxBitWidth 20 -shiftInfoSavePath shiftInfo4.t7 -nThreads 1 -device gpu -testOnly true -project $PROJECT_NAME -batchSize 50 -dataset imagenet -data /work/shadow/ -modelRoot /home/chenxi/modelzoo/vgg16/ -tensorType float -convNBits 8 -fcNBits 8 -actNBits 8 -isQuantizeBN false -experimentsName $EXP_NAME 2>&1 |tee $LOG_PATH 
