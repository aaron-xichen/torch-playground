EXP_NAME=try
PROJECT_NAME=quantization
FULL_FOLDER_PATH=project/$PROJECT_NAME/exps/$EXP_NAME
mkdir -p $FULL_FOLDER_PATH
LOG_PATH=$FULL_FOLDER_PATH/log
CUDA_VISIBLE_DEVICES=6 th main.lua -collectNSamples 10 -stopNSamples 1 -bitWidthConfigPath bitsSetting2.config -metaTablePath meta2.config -nThreads 1 -device gpu -testOnly true -project $PROJECT_NAME -batchSize 50 -dataset lfw -data /home/chenxi/dataset/LFW/lfw-deepfunneled -valListPath /home/chenxi/dataset/LFW/pairs.txt -modelRoot /home/chenxi/modelzoo/vgg_face/ -experimentsName $EXP_NAME 2>&1 |tee $LOG_PATH 
