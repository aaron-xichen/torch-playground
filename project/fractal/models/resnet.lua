--  ResNet-1001
--  This is a re-implementation of the 1001-layer residual networks described in:
--  [a] "Identity Mappings in Deep Residual Networks", arXiv:1603.05027, 2016,
--  authored by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.

--  Acknowledgement: This code is contributed by Xiang Ming from Xi'an Jiaotong Univeristy.

--  ************************************************************************
--  This code incorporates material from:

--  fb.resnet.torch (https://github.com/facebook/fb.resnet.torch)
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ************************************************************************

local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
   nUnit = opt.nUnit
   assert(nUnit % 2 == 0, 'Unit must be even')
   
   -- The new Residual Unit in [a]
   local function resUnit(nInputPlane, nOutputPlane, stride)
        local convs = nn.Sequential()
        -- 3x3 conv
        convs:add(Convolution(nInputPlane, nOutputPlane, 3, 3, stride, stride, 1, 1))
        convs:add(SBatchNorm(nOutputPlane))
        convs:add(ReLU(true))

        -- 3x3 conv
        convs:add(Convolution(nOutputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
        convs:add(SBatchNorm(nOutputPlane))
        
        local shortcut
        if nInputPlane == nOutputPlane then -- most Residual Units have this shape     
            shortcut = nn.Identity()
        else -- Residual Units for increasing dimensions        
            shortcut = Convolution(nInputPlane,nOutputPlane, 1, 1, stride, stride, 0, 0)
        end
        
        return nn:Sequential()
            :add(nn.ConcatTable()
                :add(convs)
                :add(shortcut))
            :add(nn.CAddTable())
            :add(ReLU(true))
   end

   -- Stacking Residual Units on the same stage
   local function layer(block, nInputPlane, nOutputPlane, nUnit, stride)
      local s = nn.Sequential()
      s:add(block(nInputPlane, nOutputPlane, stride))
      
      for i = 2, nUnit do
          s:add(block(nOutputPlane, nOutputPlane, 1))
      end
      return s
   end

   local model = nn.Sequential()
   if opt.dataset == 'cifar10' then
      depth = 2 + 3 * nUnit 

      -- The new ResNet-164 and ResNet-1001 in [a]
	  local nStages = {3, 16, 32, 64}
      nUnit = nUnit / 2 -- half
      model:add(Convolution(nStages[1], nStages[2], 3, 3, 1, 1, 1, 1))
      model:add(SBatchNorm(nStages[2]))
      model:add(ReLU(true))  
       
      model:add(layer(resUnit, nStages[2], nStages[2], nUnit, 1)) -- Stage 1 (spatial size: 32x32)
      model:add(layer(resUnit, nStages[2], nStages[3], nUnit, 2)) -- Stage 2 (spatial size: 16x16)
      model:add(layer(resUnit, nStages[3], nStages[4], nUnit, 2)) -- Stage 3 (spatial size: 8x8)
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(nStages[4]):setNumInputDims(3))
      model:add(nn.Linear(nStages[4], 10))
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:cuda()
    
   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil
   print(model)
   print('ResNet-' .. depth .. ' CIFAR-10')
   return model
end

return createModel
