local URL = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'

local M = {}


function M.exec(opt, cacheFile)
    print("=> Downloading MNIST dataset from " .. URL)
    local ok = os.execute('curl ' .. URL .. ' | tar xz -C gen/')
    assert(ok == true or ok == 0, 'error downloading MNIST')

    train_file = 'gen/mnist.t7/train_32x32.t7'
    test_file = 'gen/mnist.t7/test_32x32.t7'

    print(" | combining dataset into a single file")
    local trainData = torch.load(train_file, 'ascii')
    local testData = torch.load(test_file, 'ascii')

    assert(os.execute('rm -r gen/mnist.t7', 'error removing gen/mnist.t7'))
    print(" | saving MNIST dataset to " .. cacheFile)
    torch.save(cacheFile, {
            train = trainData,
            val = testData,
        })
end

return M
