local M = { }

function M.option(cmd)
    cmd:text('Torch-7 Quantization Arguments Options:')
    cmd:option('-modelRoot',       'none',    'Externel model folder')
    --cmd:option('-nInt',           -1,         'Number of bits for integer part (including sign)')
    --cmd:option('-nFrac',          -1,         'Number of bits for fraction part')
    cmd:option('-weightNBits',    -1,        'Number of bits for weights (including sign)')
    cmd:option('-activationNFrac', -1,       'Number of bits for activation (including sign)')
    cmd:option('-overFlowRate',    1 * 0.01, 'Overflow rate threshold')
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