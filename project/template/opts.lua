local M = { }

function M.option(cmd)
    cmd:text('Torch-7 Template Arguments Options:')
    cmd:option('-nUnit',            1,       'Number of units for each block')
    cmd:option('-unitType',            'frac',       'Options: frac | fracRes | fracRes2 | casRes')
    cmd:text()
    return cmd
end

function M.parse(opt)
    return opt
end

return M