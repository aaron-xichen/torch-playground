local utee = require 'utee'

local M = { }

function M.getProject(arg)
    local project
    for k, v in ipairs(arg) do
        if v == '-project' then 
            project = arg[k+1]
            break
        end
    end
    assert(project, 'Please specify a project')
    return project
end

function M.option()
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('General Arguments Options:')
    ------------ General options --------------------
    cmd:option('-manualSeed', 11,          'Manually set RNG seed')
    cmd:option('-nGPU',       1,          'Number of GPUs to use by default')
    cmd:option('-backend',    'cudnn',    'Options: cudnn | cunn')
    cmd:option('-cudnn',      'fastest',  'Options: fastest | default | deterministic')
    cmd:option('-gen',        'gen',      'Path to save generated files')
    cmd:option('-project',    'none',     'Project name')
    cmd:option('-dataloader', 'dataloader','Data loader path')
    cmd:option('-snapshotPath',       'none',       'The snapshot path to resume')
    cmd:option('-shareGradInput',  'false', 'Share gradInput tensors to reduce memory usage')
    cmd:option('-experimentsName',  'none',  'The name of this experiment')
    ------------- Data options ------------------------
    cmd:option('-data',              '/work/shadow/',   'imagenet data path')
    cmd:option('-dataset',            'none',       'Dataset want to load')
    cmd:option('-nThreads',        4, 'number of data loading threads')
    ------------- Training options --------------------
    cmd:option('-nEpochs',         1,       'Number of total epochs to run')
    cmd:option('-batchSize',       32,      'mini-batch size (1 = pure stochastic)')
    cmd:option('-testOnly',        'false', 'Run on validation set only')
    cmd:option('-tenCrop',         'false', 'Ten-crop testing')
    ---------- Optimization options ----------------------
    cmd:option('-LR',              0.1,   'initial learning rate')
    cmd:option('-momentum',        0.9,   'momentum')
    cmd:option('-weightDecay',     1e-4,  'weight decay')
    cmd:option('-nesterov',     'true',  'nesterov')
    ---------- Model options ----------------------------------
    cmd:option('-netType',      'baseline', 'Model name')
    cmd:option('-device',       'gpu',     'GPU or CPU mode')
    cmd:text()
    return cmd
end

function M.parse(cmd, arg)
    local opt = cmd:parse(arg or {})
    opt.testOnly = opt.testOnly ~= 'false'
    opt.tenCrop = opt.tenCrop ~= 'false'
    opt.shareGradInput = opt.shareGradInput ~= 'false'
    opt.nesterov = opt.nesterov ~= 'false'
        
    if opt.experimentsName == 'none' then
        cmd:error('Experiment name required')
    end
    
    -- handle workspace and exps dir
    local expsPath = paths.concat('project', opt.project, 'exps')
    opt.workspace = paths.concat(expsPath, opt.experimentsName)    
    if not paths.filep(expsPath) then
        paths.mkdir(expsPath)
    end
    
    -- test only
    if opt.testOnly then
        print('Test Mode')
    end
    
    if opt.device ~= 'cpu' and opt.device ~= 'gpu' then
        cmd:error(('Unknown device : %s'):foramt(opt.device))
    end
    
    return opt
end

return M
