local M = { }

function M.option(cmd)
    cmd:text('Torch-7 Quantization Arguments Options:')
    cmd:option('-modelRoot',       'none',    'Externel model folder')
    cmd:option('-convNBits',    -1,        'Number of bits for convolution parameters (including sign)')
    cmd:option('-fcNBits',      -1,        'Number of bits for fc parameters (including sign)')
    cmd:option('-actNBits',        -1,       'Number of bits for activation (including sign)')
    cmd:text()
    return cmd
end

function M.parse(cmd, opt)
    if opt.modelRoot == 'none' then
        cmd:error('model root required')
    end
    return opt
end

return M