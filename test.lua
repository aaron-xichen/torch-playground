require 'nn';
require 'loadcaffe';
local utee = require 'utee'
d = utee.loadTxt('bitsSetting.config')
for k, v in pairs(d) do
    print(k, v)
end

utee.saveTxt("bits.config", d)