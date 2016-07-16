local M = { }

function M.option(cmd)
    cmd:text('Torch-7 Quantization Arguments Options:')
    cmd:option('-modelRoot',       'none',    'Externel model folder')
    cmd:option('-convNBits',    -1,        'Number of bits for convolution parameters (including sign)')
    cmd:option('-fcNBits',      -1,        'Number of bits for fc parameters (including sign)')
    cmd:option('-actNBits',        -1,       'Number of bits for activation (including sign)')
    cmd:option('-tensorType',     'float',   'Tensor type of layers')
    cmd:text()
    return cmd
end

function M.parse(cmd, opt)
    if opt.modelRoot == 'none' then
        cmd:error('model root required')
    end
    
    if opt.tensorType ~= 'float' and opt.tensorType ~= 'double' then
        cmd:error(('Unknown tensorType: %s'):format(opt.tensorType))
    end
    return opt
end

return M