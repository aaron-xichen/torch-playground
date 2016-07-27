EXP_NAME=trytrace
PROJECT_NAME=quantization
FULL_FOLDER_PATH=project/$PROJECT_NAME/exps/$EXP_NAME
mkdir -p $FULL_FOLDER_PATH
LOG_PATH=$FULL_FOLDER_PATH/log
CUDA_VISIBLE_DEVICES=3 th main.lua -nThreads 1 -device gpu -testOnly true -project $PROJECT_NAME -batchSize 200 -dataset mnist -modelRoot /home/chenxi/modelzoo/lenet/ -tensorType float -convNBits 2 -fcNBits 2 -actNBits 2 -experimentsName $EXP_NAME 2>&1 |tee $LOG_PATH 
