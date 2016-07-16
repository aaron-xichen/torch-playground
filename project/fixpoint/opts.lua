local M = { }

function M.option(cmd)
    cmd:text('Torch-7 Fixed Point Arguments Options:')
    cmd:option('-modelRoot',       'none',    'Externel model folder')
    cmd:option('-shiftInfoPath',   'none',     'Shift bits info file path')
    cmd:text()
    return cmd
end

function M.parse(cmd, opt)
    if opt.modelRoot == 'none' then
        cmd:error('model root required')
    end
    if opt.accBitWidth ~= -1 then
        assert(opt.shiftInfoPath ~= 'none', 'Please specify shiftInfoPath')
        print(("Reading shift table from %s"):format(opt.shiftInfoPath))
        opt.shiftTable = torch.load(opt.shiftInfoPath)
    end
    
    return opt
end

return M