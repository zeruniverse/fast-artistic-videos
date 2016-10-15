require 'torch'
require 'nn'
require 'image'
require 'loadcaffe'
require 'artistic_video_core'

local flowFile = require 'flowFileLoader'

--------------------------------------------------------------------------------

local cmd = torch.CmdLine()

-- Basic options
cmd:option('-content_pattern', 'example/marple8_%02d.ppm',
           'Content target pattern')
cmd:option('-num_images', 0, 'Number of content images. Set 0 for autodetect.')
cmd:option('-start_number', 1, 'Frame index to start with')
cmd:option('-continue_with', 1, 'Continue with the given frame index.')
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
cmd:option('-number_format', '%d', 'Number format of the output images.')

--Flow options
cmd:option('-flow_pattern', 'example/deepflow/backward_[%d]_{%d}.flo',
           'Optical flow files pattern')
cmd:option('-flowWeight_pattern', 'example/deepflow/reliable_[%d]_{%d}.pgm',
           'Optical flow weight files pattern.')

-- Optimization options
cmd:option('-perceptual_weight', 5e0)
cmd:option('-style_weight', 5e0)
cmd:option('-pixel_weight', 1.5e-4)
cmd:option('-temporal_weight', 1e3)
cmd:option('-tv_weight', 1e-3)
cmd:option('-temporal_loss_criterion', 'mse', 'mse|smoothl1')
cmd:option('-num_iterations', '100',
           'Can be set separately for the first and for subsequent iterations, separated by comma, or one value for all.')
cmd:option('-tol_loss_relative', 0.0001, 'Stop if relative change of the loss function is below this value')
cmd:option('-tol_loss_relative_interval', 50, 'Interval between two loss comparisons')
cmd:option('-normalize_gradients', false)
cmd:option('-init', 'image,prevWarped', 'random|image,random|image|prev|prevWarped')
cmd:option('-optimizer', 'lbfgs', 'lbfgs|adam')
cmd:option('-learning_rate', 1e1)

-- Output options
cmd:option('-output_image', 'out.png')
cmd:option('-output_folder', '')

-- Other options
cmd:option('-pooling', 'max', 'max|avg')
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-backend', 'nn', 'nn|cudnn|clnn')
cmd:option('-cudnn_autotune', false)
cmd:option('-seed', -1)
cmd:option('-perceptual_layers', 'relu4_2', 'layers for content')
cmd:option('-style_layers', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', 'layers for style')
cmd:option('-args', '', 'Arguments in a file, one argument per line')


function nn.SpatialConvolutionMM:accGradParameters()
  -- nop.  not needed by our net
end

local function main(params)
  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      require 'cutorch'
      require 'cunn'
      cutorch.setDevice(params.gpu + 1)
    else
      require 'clnn'
      require 'cltorch'
      cltorch.setDevice(params.gpu + 1)
    end
  else
    params.backend = 'nn'
  end

  if params.backend == 'cudnn' then
    require 'cudnn'
    if params.cudnn_autotune then 
      cudnn.benchmark = true
    end
    cudnn.SpatialConvolution.accGradParameters = nn.SpatialConvolutionMM.accGradParameters -- ie: nop
  end

  local loadcaffe_backend = params.backend
  if params.backend == 'clnn' then loadcaffe_backend = 'nn' end
  local cnn = loadcaffe.load(params.proto_file, params.model_file, loadcaffe_backend):float()
  cnn = MaybePutOnGPU(cnn, params)

  -- Set up the network, inserting style losses. Content and temporal loss will be inserted in each iteration.
  local net, losses_indices, losses_type = buildNet(cnn, params)

  -- We don't need the base CNN anymore, so clean it up to save memory.
  cnn = nil
  for i=1,#net.modules do
    local module = net.modules[i]
    if torch.type(module) == 'nn.SpatialConvolutionMM' then
        -- remote these, not used, but uses gpu memory
        module.gradWeight = nil
        module.gradBias = nil
    end
  end
  collectgarbage()

  -- There can be different setting for the first frame and for subsequent frames.
  local num_iterations = params.num_iterations
  local init_split = params.init:split(",")
  local init_first, init_subseq = init_split[1], init_split[2] or init_split[1]
  
  local firstImg = nil

  local num_images = params.num_images
  if num_images == 0 then
    num_images = calcNumberOfContentImages(params)
    print("Detected " .. num_images .. " content images.")
  end

  -- Iterate over all frames in the video sequence
  for frameIdx=params.start_number + params.continue_with - 1, params.start_number + num_images - 1 do

    -- Set seed
    if params.seed >= 0 then
      torch.manualSeed(params.seed)
    end

    local content_image = getContentImage(frameIdx, params)
    if content_image == nil then
      print("No more frames.")
      do return end
    end
    local perceptual_losses, style_losses = {}, {}
    local additional_layers = 0
    local init = frameIdx == params.start_number and init_first or init_subseq
    local imgWarped = nil
    local temporal_reliable = nil
    -- Calculate from which indices we need a warped image
    if frameIdx > params.start_number and params.temporal_weight ~= 0 then
      local prevIndex = frameIdx - 1
      local flowFileName = getFormatedFlowFileName(params.flow_pattern, math.abs(prevIndex), math.abs(frameIdx))
      local weightsFileName = getFormatedFlowFileName(params.flowWeight_pattern, prevIndex, math.abs(frameIdx))
      temporal_reliable = image.load(weightsFileName):float()
      temporal_reliable = MaybePutOnGPU(temporal_reliable, params)
      print(string.format('Reading flow file "%s".', flowFileName))
      local flow = flowFile.load(flowFileName)
      local fileName = build_OutFilename(params, math.abs(prevIndex - params.start_number + 1), -1)
      imgWarped = warpImage(image.load(fileName, 3), flow)
      print(string.format('file "%s".', fileName))
      imgWarped = preprocess(imgWarped):float()
      imgWarped = MaybePutOnGPU(imgWarped, params)
    end

    for i=1, #losses_indices do
      if losses_type[i] == 'perceptual'  then
        local loss_module = getPerceptualLossModuleForLayer(net,
          losses_indices[i] + additional_layers, content_image, params)
        net:insert(loss_module, losses_indices[i] + additional_layers)
        table.insert(perceptual_losses, loss_module)
        additional_layers = additional_layers + 1
      elseif losses_type[i] == 'style' then
        local loss_module = getStyleLossModuleForLayer(net,
          losses_indices[i] + additional_layers, content_image, params)
        net:insert(loss_module, losses_indices[i] + additional_layers)
        table.insert(style_losses, loss_module)
        additional_layers = additional_layers + 1
      end
    end

    -- Initialization
    local img = nil

    
    if init == 'random' then
      img = torch.randn(content_image:size()):float():mul(0.001)
    elseif init == 'image' then
      img = content_image:clone():float()
    elseif init == 'prevWarped' and frameIdx > params.start_number then
      local flowFileName = getFormatedFlowFileName(params.flow_pattern, math.abs(frameIdx - 1), math.abs(frameIdx))
      print(string.format('Reading flow file "%s".', flowFileName))
      local flow = flowFile.load(flowFileName)
      local fileName = build_OutFilename(params, math.abs(frameIdx - params.start_number), -1)
      img = warpImage(image.load(fileName, 3), flow)
      img = preprocess(img):float()
    elseif init == 'prev' and frameIdx > params.start_number then
      local fileName = build_OutFilename(params, math.abs(frameIdx - params.start_number), -1)
      img = image.load(fileName, 3)
      img = preprocess(img):float()
    elseif init == 'first' then
      img = firstImg:clone():float()
    else
      print('ERROR: Invalid initialization method.')
      os.exit()
    end
    
    img = MaybePutOnGPU(img, params)

    -- Run the optimization to stylize the image, save the result to disk
    runOptimization(params, net, perceptual_losses, style_losses, temporal_losses, img, frameIdx, num_iterations, content_image,imgWarped,temporal_reliable)

    if frameIdx == params.start_number then
      firstImg = img:clone():float()
    end
    
    -- Remove this iteration's content layers
    for i=#losses_indices, 1, -1 do
      if frameIdx > params.start_number or losses_type[i] == 'perceptual' or losses_type[i] == 'style' then
        additional_layers = additional_layers - 1
        net:remove(losses_indices[i] + additional_layers)
      end
    end
    
    -- Ensure that all layer have been removed correctly
    assert(additional_layers == 0)
    
  end
end

-- warp a given image according to the given optical flow.
-- Disocclusions at the borders will be filled with the VGG mean pixel.
function warpImage(img, flow)
  -- local mean_pixel = torch.DoubleTensor({123.68/256.0, 116.779/256.0, 103.939/256.0})
  result = image.warp(img, flow, 'bilinear', true, 'pad', -1)
  for x=1, result:size(2) do
    for y=1, result:size(3) do
      if result[1][x][y] == -1 and result[2][x][y] == -1 and result[3][x][y] == -1 then
        result[1][x][y] = img[1][x][y]
        result[2][x][y] = img[2][x][y]
        result[3][x][y] = img[3][x][y]
      end
    end
  end
  return result
end

local tmpParams = cmd:parse(arg)
local params = nil
local file = io.open(tmpParams.args, 'r')

if tmpParams.args == '' or file == nil  then
  params = cmd:parse(arg)
else
  local args = {}
  io.input(file)
  local argPos = 1
  while true do
    local line = io.read()
    if line == nil then break end
    if line:sub(0, 1) == '-' then
      local splits = str_split(line, " ", 2)
      args[argPos] = splits[1]
      args[argPos + 1] = splits[2]
      argPos = argPos + 2
    end
  end
  for i=1, #arg do
    args[argPos] = arg[i]
    argPos = argPos + 1
  end
  params = cmd:parse(args)
  io.close(file)
end

main(params)