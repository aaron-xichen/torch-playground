local utee = require 'utee'
local M = { }

function M.option(cmd)
    cmd:text('Torch-7 FacePair Arguments Options:')
    cmd:option('-modelRoot',       'none',    'Externel model folder')
    cmd:option('-convNBits',    -1,        'Number of bits for convolution parameters (including sign)')
    cmd:option('-fcNBits',      -1,        'Number of bits for fc parameters (including sign)')
    cmd:option('-actNBits',        -1,       'Number of bits for activation (including sign)')
    cmd:option('-tensorType',     'float',   'Tensor type of layers')
    cmd:option('-collectNSamples',  10,      'Number of samples to collect')
    cmd:option('-isQuantizeBN',  'true',      'Whether to quantize BN')
    cmd:option('-shiftInfoSavePath', 'shiftInfo.t7', 'Save path of shift bits, including weights, bias and activation')
    cmd:option('-valListPath', '/home/chenxi/dataset/LFW/pairs.txt', 'Val file list path')
    cmd:option('-crop',   1,         'Whether 10 samples')
    cmd:option('-savePath', 'log.t7',   'Save path')
    cmd:text()
    return cmd
end

function M.parse(cmd, opt)
    -- model path
    if opt.modelRoot == 'none' then
        cmd:error('model root required')
    end
    opt.netPath = opt.modelRoot .. '/deploy.prototxt'
    opt.modelPath = opt.modelRoot .. '/weights.caffemodel'
    opt.meanfilePath = opt.modelRoot .. '/meanfile.t7'
    if opt.device == 'gpu' then
        opt.torchModelPath = opt.modelRoot .. '/model.t7'
    else 
        opt.torchModelPath = opt.modelRoot .. '/modelCPU.t7'
    end
    
    if paths.filep(opt.shiftInfoSavePath) then
        print("Loading shift info table from " .. opt.shiftInfoSavePath)
        opt.shiftInfoTable = torch.load(opt.shiftInfoSavePath)
    end
    
    if opt.tensorType ~= 'float' and opt.tensorType ~= 'double' then
        cmd:error(('Unknown tensorType: %s'):format(opt.tensorType))
    end
    
    opt.isQuantizeBN = opt.isQuantizeBN ~= 'false'
    assert(opt.crop == 1 or opt.crop == 10, 'Neither 10 nor 1')
    return opt
end

return M
