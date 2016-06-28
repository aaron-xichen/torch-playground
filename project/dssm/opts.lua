local M = { }

function M.option(cmd)
    cmd:text('Torch-7 Microsoft DSSM Arguments Options:')
    cmd:option('-activation',       'ReLU', 'Activation function of Linear layer')
    cmd:option('-nFeature',         49284, '#Features of input word hashing result')
    cmd:option('-nTrial',           50,      '#Negative samples per positive sample')
    cmd:option('-offsetRatio',      0.1,    'Sampling offset ratio')
    cmd:option('-trainDataPath',  '/tmp/train2.t7', 'Query and doc file path for training')
    cmd:option('-valDataPath',  '/tmp/val2.t7', 'Query and doc file path for testing')
    cmd:option('-trainQueryFilePath',  '/home/chenxi/dataset/dssm2/query.train.tsv', 'Query file path for training')
    cmd:option('-trainDocFilePath',    '/home/chenxi/dataset/dssm2/doc.train.tsv', 'Doc file path for training')
    cmd:option('-trainNSamples',       10240,     '#samples for training')
    cmd:option('-testQueryFilePath',   '/home/chenxi/dataset/dssm2/query.test.tsv', 'Query file path for testing')
    cmd:option('-testDocFilePath',   '/home/chenxi/dataset/dssm2/doc.test.tsv', 'Doc file path for testing')
    cmd:option('-testNSamples',        10240,     '#samples for testing')
    cmd:option('-nBits',            -1,       'Number bits of quantization')
    cmd:text()
    return cmd
end

function M.parse(cmd, opt)
    opt.memory = opt.memory ~= 'false'
    opt.multiThread = opt.multiThread ~= 'false'      
    return opt
end

return M

