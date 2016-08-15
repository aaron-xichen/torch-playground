local utee = require 'utee'
local M = { }

function M.option(cmd)
    cmd:text('Torch-7 Quantization Arguments Options:')
    cmd:option('-modelRoot',       'none',    'Externel model folder')
    cmd:option('-convNBits',    -1,        'Number of bits for convolution parameters (including sign)')
    cmd:option('-fcNBits',      -1,        'Number of bits for fc parameters (including sign)')
    cmd:option('-actNBits',        -1,       'Number of bits for activation (including sign)')
    cmd:option('-tensorType',     'float',   'Tensor type of layers')
    cmd:option('-collectNSamples',  10,      'Number of samples to collect')
    cmd:option('-isQuantizeBN',  'true',      'Whether to quantize BN')
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
    opt.torchModelPath = opt.modelRoot .. '/weights.t7'
    
    
    if opt.tensorType ~= 'float' and opt.tensorType ~= 'double' then
        cmd:error(('Unknown tensorType: %s'):format(opt.tensorType))
    end
    
    opt.isQuantizeBN = opt.isQuantizeBN ~= 'false'
    
    return opt
end

return M
