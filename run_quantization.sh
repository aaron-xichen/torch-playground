EXP_NAME=trytrace
PROJECT_NAME=quantization
FULL_FOLDER_PATH=project/$PROJECT_NAME/exps/$EXP_NAME
mkdir -p $FULL_FOLDER_PATH
LOG_PATH=$FULL_FOLDER_PATH/log
CUDA_VISIBLE_DEVICES=7 th main.lua -nThreads 1 -device gpu -testOnly true -project $PROJECT_NAME -batchSize 20 -dataset imagenet -modelRoot /home/chenxi/modelzoo/vgg16/ -tensorType float -isQuantizeBN true -convNBits 8 -fcNBits 8 -actNBits 8 -experimentsName $EXP_NAME 3>&1 |tee $LOG_PATH 
