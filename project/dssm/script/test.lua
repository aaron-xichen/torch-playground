require 'pl'
qFilePath = '/home/chenxi/dataset/dssm/query.test.tsv'
dFilePath = '/home/chenxi/dataset/dssm/doc.test.tsv'
qFileStream = assert(io.open(qFilePath))
dFileStream = assert(io.open(dFilePath))
queries = {}
docs = {}
batchSize = 1024
while true do
    if #queries % 312260 == 0 then 
        print(#queries)
    end
    qline = qFileStream:lines()()
    dline = dFileStream:lines()()
    if not qline or not dline then break end
    if #qline > 0 and #dline > 0 then
        qfields = stringx.split(qline)
        dfields = stringx.split(dline)
        query = {}
        for i, pair in ipairs(qfields) do
            spl = stringx.split(pair, ':')
            key = spl[1]
            val = spl[2]
            query[key] = val
        end
        table.insert(queries, query)
        doc = {}
        for i, pair in ipairs(dfields) do
            spl = stringx.split(pair, ':')
            key = spl[1]
            val = spl[2]
            doc[key] = val
        end
        table.insert(docs, doc)
    end
end