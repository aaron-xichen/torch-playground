require 'pl';


nSlot = 550

qTrainFile = '/tmp/dssm2/query.train.tsv'
dTrainFile = '/tmp/dssm2/doc.train.tsv'
qTrainData = torch.IntTensor(1024000, nSlot, 2):fill(0)
dTrainData = torch.IntTensor(1024000, nSlot, 2):fill(0)

-- read train data
qFileStream = assert(io.open(qTrainFile))
dFileStream = assert(io.open(dTrainFile))
print("process train data")

local curLine = 0
while true do
    qline = qFileStream:lines()()
    dline = dFileStream:lines()()
    if qline == nil or dline == nil then break end
    if #qline > 0 and #dline > 0 then
        curLine = curLine + 1
        
        for i, pair in ipairs(stringx.split(qline)) do
            assert(i <= nSlot, 'out of range')
            spl = stringx.split(pair, ':')
            key = spl[1]
            val = spl[2]
            qTrainData[curLine][i][1] = key + 1
            qTrainData[curLine][i][2] = val
        end
        
        for i, pair in ipairs(stringx.split(dline)) do
            assert(i <= nSlot, 'out of range')
            spl = stringx.split(pair, ':')
            key = spl[1]
            val = spl[2]
            dTrainData[curLine][i][1] = key + 1
            dTrainData[curLine][i][2] = val
        end
    end
    
    if curLine % 1024 == 0 then 
        print(curLine)
    end
end

print(curLine)
qTrainData = qTrainData:narrow(1, 1, curLine):short()
dTrainData = dTrainData:narrow(1, 1, curLine):short()

print(qTrainData:size())
print(dTrainData:size())

print(torch.max(qTrainData))
print(torch.max(dTrainData))

trainData = {
    query = qTrainData,
    doc = dTrainData
}
torch.save('/tmp/train2.t7', trainData)

-- read val data
qValFile = '/tmp/dssm2/query.test.tsv'
dValFile = '/tmp/dssm2/doc.test.tsv'
qValData = torch.IntTensor(1024000, nSlot, 2):fill(0)
dValData = torch.IntTensor(1024000, nSlot, 2):fill(0)

qFileStream = assert(io.open(qValFile))
dFileStream = assert(io.open(dValFile))
print("process val data")

local curLine = 0
while true do
    qline = qFileStream:lines()()
    dline = dFileStream:lines()()
    if qline == nil or dline == nil then break end
    if #qline > 0 and #dline > 0 then
        curLine = curLine + 1
        
        for i, pair in ipairs(stringx.split(qline)) do
            assert(i <= nSlot, 'out of range')
            spl = stringx.split(pair, ':')
            key = spl[1]
            val = spl[2]
            qValData[curLine][i][1] = key + 1
            qValData[curLine][i][2] = val
        end
        
        for i, pair in ipairs(stringx.split(dline)) do
            assert(i <= nSlot, 'out of range')
            spl = stringx.split(pair, ':')
            key = spl[1]
            val = spl[2]
            dValData[curLine][i][1] = key + 1
            dValData[curLine][i][2] = val
        end
    end
    
    if curLine % 1024 == 0 then 
        print(curLine)
    end
end

print(curLine)
qValData = qValData:narrow(1, 1, curLine):short()
dValData = dValData:narrow(1, 1, curLine):short()

print(qValData:size())
print(dValData:size())

print(torch.max(qValData))
print(torch.max(dValData))

valData = {
    query = qValData,
    doc = dValData
}
torch.save('/tmp/val2.t7', valData)
