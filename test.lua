require 'loadcaffe';
require 'nn';
require 'cudnn';
require 'cutorch';
torch.setdefaulttensortype("torch.FloatTensor")
cudnn.fastest = false
cudnn.benchmark = false
torch.manualSeed(71)
cutorch.manualSeedAll(71)
model1 = loadcaffe.load('/home/chenxi/modelzoo/vgg16/deploy.prototxt', '/home/chenxi/modelzoo/vgg16/weights.caffemodel', 'cudnn')
model2 = loadcaffe.load('/home/chenxi/modelzoo/vgg16/deploy.prototxt', '/home/chenxi/modelzoo/vgg16/weights.caffemodel', 'nn')
mean = torch.Tensor(torch.load('/home/chenxi/modelzoo/vgg19/meanfile.t7').mean)
mean = - mean:reshape(3, 1, 1):expand(3, 224, 224)
model1:apply(
    function(m) 
        if m.setMode then m:setMode(1, 1, 1) end 
    end)
--model1:get(1):setMode('CUDNN_CONVOLUTION_FWD_ALGO_GEMM','CUDNN_CONVOLUTION_BWD_DATA_ALGO_1','CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1')
model1:get(1):setMode(1, 1, 1)
y1 = model1:get(1):forward(mean:cuda())
y2 = model2:get(1):forward(mean)
print(torch.sum(torch.abs(y1:float() - y2:float())))
print(torch.typename(model1:get(1).weight))
print(torch.typename(model2:get(1).weight))