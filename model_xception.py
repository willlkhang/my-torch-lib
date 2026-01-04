import torch.nn as nn
import torch

#conv --> batchnorm --> relu
class ConvBlock(nn.Module):
  def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias=False):
    super().__init__()
    self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
    self.bn = nn.BatchNorm2d(out_channel)
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.relu(self.bn(self.conv(x))) #I hope this way of coding will help people understand deeper about sequence in nn
  

#depth --> piont
class seperableConv(nn.Module):
  def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias=False):
    super().__init__()
    self.depthwise = nn.Conv2d(in_channel,
                               in_channel, #look at param groups
                               kernel_size, stride, padding,
                               groups=in_channel, #this is depthwise
                               bias=bias)
    self.pointwise = nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=bias) #kernel, stride, padding

  def forward(self, x): #this is another way to move everything forward the network
    x = self.depthwise(x)
    x = self.pointwise(x)
    return x

#residuel block with 1x1 shortcut
class EntryFlowBlock(nn.Module):
  def __init__(self, in_channel, out_channel):
    super().__init__() # Add this line
    #shortcut path to make x in h(x) = f(x) + x (explain this later)
    self.shortcut_conv = nn.Conv2d(in_channel, out_channel, 1, stride=2, bias=False)
    self.shortcut_bn = nn.BatchNorm2d(out_channel)

    #main path
    self.sep_conv1 = seperableConv(in_channel, out_channel, 3, 1, 1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channel) # This BatchNorm expects out_channel, which is 128
    self.relu1 = nn.ReLU(inplace=True)

    self.sep_conv2 = seperableConv(out_channel, out_channel, 3, 1, 1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channel) # This BatchNorm expects out_channel, which is 128

    self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

  def forward(self, x):
    gx = self.shortcut_bn(self.shortcut_conv(x)) #g(x)

    #mainpath
    fx = self.bn1(self.sep_conv1(x))
    fx = self.relu1(fx)
    fx = self.bn2(self.sep_conv2(fx))
    fx = self.pool1(fx)

    return fx + gx #h(x) = f(x) + g(x)
  
#residule from the middle floW
class MiddleFlowBlock(nn.Module):
  def __init__(self, channels=728):
    super().__init__()

    self.relu = nn.ReLU()
    self.sep_conv1 = seperableConv(channels, channels, 3, 1, 1, bias=False)
    self.bn1 = nn.BatchNorm2d(channels)

    self.sep_conv2 = seperableConv(channels, channels, 3, 1, 1, bias=False)
    self.bn2 = nn.BatchNorm2d(channels)

    self.sep_conv3 = seperableConv(channels, channels, 3, 1, 1, bias=False)
    self.bn3 = nn.BatchNorm2d(channels)

  def forward(self, x):
    shortcut = x

    fx = self.relu(x)
    fx = self.bn1(self.sep_conv1(fx))
    fx = self.relu(fx)
    fx = self.bn2(self.sep_conv2(fx))
    fx = self.relu(fx)
    fx = self.bn3(self.sep_conv3(fx))

    return fx + x
  
class ExitFlowBlock(nn.Module):
  def __init__(self, in_channel=728, out_channel=1028):
    super().__init__()
    #shortcut path g(x)
    self.shortcut_conv = nn.Conv2d(in_channel, out_channel, 1, stride=2, bias=False)
    self.shortcut_bn = nn.BatchNorm2d(out_channel)

    #f(x) path
    self.relu = nn.ReLU()
    self.sep_conv1 = seperableConv(in_channel, out_channel, 3, 1, 1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channel)

    self.sep_conv2 = seperableConv(out_channel, out_channel, 3, 1, 1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channel)

    self.pool = nn.MaxPool2d(3, stride=2, padding=1)

  def forward(self, x):
    gx = self.shortcut_bn(self.shortcut_conv(x))

    fx = self.relu(x)
    fx = self.bn1(self.sep_conv1(fx))
    fx = self.relu(fx)
    fx = self.bn2(self.sep_conv2(fx))
    fx = self.pool(fx)

    return fx + gx
  
#sepconv -> bn -> relu
class sepconvRelu(nn.Module):
  def __init__(self, in_channel, out_channel):
    super().__init__()
    self.sep_conv = seperableConv(in_channel, out_channel, 3, 1, 1, bias=False)
    self.bn = nn.BatchNorm2d(out_channel)
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.relu(self.bn(self.sep_conv(x)))
  

class my_xception(nn.Module):
  def __init__(self, output_size, lr=1e-3):
    super().__init__()

    self.entryflow = nn.Sequential(
        ConvBlock(3, 32, 3, 2, 0),
        ConvBlock(32, 64, 3, 1, 1),


        EntryFlowBlock(64, 128),
        EntryFlowBlock(128, 256),
        EntryFlowBlock(256, 728)
    )

    middle_blocks = []
    for _ in range(8):
      middle_blocks.append(MiddleFlowBlock(channels=728))

    self.middleflow = nn.Sequential(*middle_blocks)

    self.exitflow = nn.Sequential(
        ExitFlowBlock(728, 1024),
        sepconvRelu(1024, 1536),
        sepconvRelu(1536, 2048)
    )

    self.glob_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(2048, output_size)

  def forward(self, x):
    x = self.entryflow(x)
    x = self.middleflow(x)
    x = self.exitflow(x)
    x = self.glob_avg_pool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)

    return x