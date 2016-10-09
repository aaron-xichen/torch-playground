EXP_NAME=try
PROJECT_NAME=face-pair
FULL_FOLDER_PATH=project/$PROJECT_NAME/exps/$EXP_NAME
mkdir -p $FULL_FOLDER_PATH
LOG_PATH=$FULL_FOLDER_PATH/log
CUDA_VISIBLE_DEVICES=7 th main.lua -collectNSamples 200 -convNBits 4 -fcNBits 4 -actNBits 4 -crop 10 -batchSize 4 -device gpu -project face-pair -dataset lfw -data /home/chenxi/dataset/LFW/lfw-deepfunneled -valListPath /home/chenxi/dataset/LFW/pairs.txt -testOnly true -modelRoot /home/chenxi/modelzoo/vgg_face_caffe/ -experimentsName $EXP_NAME 2>&1 |tee $LOG_PATH 
