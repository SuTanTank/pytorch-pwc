#!/usr/bin/env python

import torch

import getopt
import copy
import math
import numpy
import os
import PIL
import PIL.Image
import sys

try:
    from correlation import correlation  # the custom cost volume layer
except:
    sys.path.insert(0, './correlation');
    import correlation  # you should consider upgrading python
# end

##########################################################

assert (int(str('').join(torch.__version__.split('.')[0:3])) >= 41)  # requires at least pytorch version 0.4.1

torch.set_grad_enabled(False)  # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'default'
arguments_strFirst = './images/first.png'
arguments_strSecond = './images/second.png'
arguments_strOut = './out.flo'

for strOption, strArgument in \
getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
    if strOption == '--model' and strArgument != '': arguments_strModel = strArgument  # which model to use
    if strOption == '--first' and strArgument != '': arguments_strFirst = strArgument  # path to the first frame
    if strOption == '--second' and strArgument != '': arguments_strSecond = strArgument  # path to the second frame
    if strOption == '--out' and strArgument != '': arguments_strOut = strArgument  # path to where the output should be stored
# end

##########################################################

Backward_tensorGrid = {}
Backward_tensorPartial = {}

# warping image with flow
def Backward(tensorInput, tensorFlow):
    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(
            tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(
            tensorFlow.size(0), -1, -1, tensorFlow.size(3))

        Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([tensorHorizontal, tensorVertical], 1).cuda()
    # end

    if str(tensorFlow.size()) not in Backward_tensorPartial:
        Backward_tensorPartial[str(tensorFlow.size())] = tensorFlow.new_ones(
            [tensorFlow.size(0), 1, tensorFlow.size(2), tensorFlow.size(3)])
    # end

    tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                            tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)
    tensorInput = torch.cat([tensorInput, Backward_tensorPartial[str(tensorFlow.size())]], 1)

    tensorOutput = torch.nn.functional.grid_sample(input=tensorInput, grid=(
                Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear',
                                                   padding_mode='zeros')

    tensorMask = tensorOutput[:, -1:, :, :];
    tensorMask[tensorMask > 0.999] = 1.0;
    tensorMask[tensorMask < 1.0] = 0.0

    return tensorOutput[:, :-1, :, :] * tensorMask


# end

##########################################################

class NewNetwork(torch.nn.Module):
    def __init__(self):
        super(NewNetwork, self).__init__()

        class Extractor1(torch.nn.Module):
            def __init__(self):
                super(Extractor1, self).__init__()
                self.moduleOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

            def forward(self, tensorInput):
                tensorOne = self.moduleOne(tensorInput)
                return tensorOne

        class Extractor2(torch.nn.Module):
            def __init__(self):
                super(Extractor2, self).__init__()
                self.moduleTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

            def forward(self, tensorOne):
                tensorTwo = self.moduleTwo(tensorOne)
                return tensorTwo

        class Extractor3(torch.nn.Module):
            def __init__(self):
                super(Extractor3, self).__init__()
                self.moduleThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

            def forward(self, tensorTwo):
                tensorThr = self.moduleThr(tensorTwo)
                return tensorThr

        class Extractor4(torch.nn.Module):
            def __init__(self):
                super(Extractor4, self).__init__()
                self.moduleFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

            def forward(self, tensorThr):
                tensorFou = self.moduleFou(tensorThr)
                return tensorFou

        class Extractor5(torch.nn.Module):
            def __init__(self):
                super(Extractor5, self).__init__()
                self.moduleFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

            def forward(self, tensorFou):
                tensorFiv = self.moduleFiv(tensorFou)
                return tensorFiv

        class Extractor6(torch.nn.Module):
            def __init__(self):
                super(Extractor6, self).__init__()
                self.moduleSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

            def forward(self, tensorFiv):
                tensorSix = self.moduleSix(tensorFiv)
                return tensorSix

        class Decoder(torch.nn.Module):
            def __init__(self, intLevel):
                super(Decoder, self).__init__()

                intPrevious = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][intLevel + 1]
                intCurrent = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][intLevel + 0]

                self.moduleUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
                self.moduleUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
                self.dblBackward = [None, None, None, 5.0, 2.5, 1.25, 0.625, None][intLevel + 1]

                self.moduleOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3,
                                    stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3,
                                    stride=1, padding=1)
                )

            # end

            def forward(self, tensorFirst, tensorSecond, objectPrevious):

                tensorFlow = self.moduleUpflow(objectPrevious['tensorFlow'])
                tensorFeat = self.moduleUpfeat(objectPrevious['tensorFeat'])

                warped = Backward(tensorInput=tensorSecond, tensorFlow=tensorFlow * self.dblBackward)
                corr = correlation.FunctionCorrelation(tensorFirst=tensorFirst, tensorSecond=warped)

                tensorVolume = torch.nn.functional.leaky_relu(input=corr, negative_slope=0.1, inplace=False)

                tensorFeat = torch.cat([tensorVolume, tensorFirst, tensorFlow, tensorFeat], 1)

                tensorFeat = torch.cat([self.moduleOne(tensorFeat), tensorFeat], 1)
                tensorFeat = torch.cat([self.moduleTwo(tensorFeat), tensorFeat], 1)
                tensorFeat = torch.cat([self.moduleThr(tensorFeat), tensorFeat], 1)
                tensorFeat = torch.cat([self.moduleFou(tensorFeat), tensorFeat], 1)
                tensorFeat = torch.cat([self.moduleFiv(tensorFeat), tensorFeat], 1)

                tensorFlow = self.moduleSix(tensorFeat)

                return {
                    'tensorFlow': tensorFlow,
                    'tensorFeat': tensorFeat
                }
            # end

        # end

        class DecoderFirst(torch.nn.Module):
            def __init__(self):
                super(DecoderFirst, self).__init__()

                class FeatureProcessor(torch.nn.Module):
                    def __init__(self):
                        super(FeatureProcessor, self).__init__()
                        intCurrent = 81
                        self.moduleOne = torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
                            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                        )

                        self.moduleTwo = torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
                            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                        )

                        self.moduleThr = torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1,
                                            padding=1),
                            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                        )

                        self.moduleFou = torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1,
                                            padding=1),
                            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                        )

                        self.moduleFiv = torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3,
                                            stride=1, padding=1),
                            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                        )

                    def forward(self, tensorVolume):
                        tensorFeat = torch.cat([tensorVolume], 1)

                        tensorFeat = torch.cat([self.moduleOne(tensorFeat), tensorFeat], 1)
                        tensorFeat = torch.cat([self.moduleTwo(tensorFeat), tensorFeat], 1)
                        tensorFeat = torch.cat([self.moduleThr(tensorFeat), tensorFeat], 1)
                        tensorFeat = torch.cat([self.moduleFou(tensorFeat), tensorFeat], 1)
                        tensorFeat = torch.cat([self.moduleFiv(tensorFeat), tensorFeat], 1)

                        return tensorFeat

                class FlowProcessor(torch.nn.Module):
                    def __init__(self):
                        super(FlowProcessor, self).__init__()
                        intCurrent = 81
                        self.moduleSix = torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2,
                                            kernel_size=3,
                                            stride=1, padding=1)
                        )

                    def forward(self, tensorFeat):
                        return self.moduleSix(tensorFeat)

                self.featureProcessor = FeatureProcessor()
                self.flowProcessor = FlowProcessor()
            # end

            def forward(self, tensorFirst, tensorSecond):
                corr = correlation.FunctionCorrelation(tensorFirst=tensorFirst, tensorSecond=tensorSecond)
                tensorVolume = torch.nn.functional.leaky_relu(input=corr, negative_slope=0.1, inplace=False)
                feature1 = torch.jit.trace(self.featureProcessor, tensorVolume)
                feature1.save("feature1.pt")
                tensorFeat = self.featureProcessor(tensorVolume)
                flow1 = torch.jit.trace(self.flowProcessor, tensorFeat)
                flow1.save("flow1.pt")
                tensorFlow = self.flowProcessor(tensorFeat)
                return {
                    'tensorFlow': tensorFlow,
                    'tensorFeat': tensorFeat
                }
            # end

        # end


        class Refiner(torch.nn.Module):
            def __init__(self):
                super(Refiner, self).__init__()

                self.moduleMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128,
                                    kernel_size=3, stride=1, padding=1, dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
                )

            # end

            def forward(self, tensorInput):
                return self.moduleMain(tensorInput)
            # end

        # end

        self.moduleExtractor1 = Extractor1()
        self.moduleExtractor2 = Extractor2()
        self.moduleExtractor3 = Extractor3()
        self.moduleExtractor4 = Extractor4()
        self.moduleExtractor5 = Extractor5()
        self.moduleExtractor6 = Extractor6()
        self.moduleDecoderFirst = DecoderFirst()
        self.moduleFiv = Decoder(5)
        self.moduleFou = Decoder(4)
        self.moduleThr = Decoder(3)
        self.moduleTwo = Decoder(2)
        self.moduleRefiner = Refiner()

    def forward(self, tensorFirst, tensorSecond):

        tensorFeatFirst = []
        extractor1 = torch.jit.trace(newNetwork.moduleExtractor1, tensorFirst)
        tensorFeatFirst.append(self.moduleExtractor1(tensorFirst))
        extractor2 = torch.jit.trace(newNetwork.moduleExtractor2, tensorFeatFirst[-1])
        tensorFeatFirst.append(self.moduleExtractor2(tensorFeatFirst[-1]))
        extractor3 = torch.jit.trace(newNetwork.moduleExtractor3, tensorFeatFirst[-1])
        tensorFeatFirst.append(self.moduleExtractor3(tensorFeatFirst[-1]))
        extractor4 = torch.jit.trace(newNetwork.moduleExtractor4, tensorFeatFirst[-1])
        tensorFeatFirst.append(self.moduleExtractor4(tensorFeatFirst[-1]))
        extractor5 = torch.jit.trace(newNetwork.moduleExtractor5, tensorFeatFirst[-1])
        tensorFeatFirst.append(self.moduleExtractor5(tensorFeatFirst[-1]))
        extractor6 = torch.jit.trace(newNetwork.moduleExtractor6, tensorFeatFirst[-1])
        tensorFeatFirst.append(self.moduleExtractor6(tensorFeatFirst[-1]))

        extractor1.save('extractor1.pt')
        extractor2.save('extractor2.pt')
        extractor3.save('extractor3.pt')
        extractor4.save('extractor4.pt')
        extractor5.save('extractor5.pt')
        extractor6.save('extractor6.pt')
        tensorFeatSecond = []

        tensorFeatSecond.append(self.moduleExtractor1(tensorSecond))
        tensorFeatSecond.append(self.moduleExtractor2(tensorFeatSecond[-1]))
        tensorFeatSecond.append(self.moduleExtractor3(tensorFeatSecond[-1]))
        tensorFeatSecond.append(self.moduleExtractor4(tensorFeatSecond[-1]))
        tensorFeatSecond.append(self.moduleExtractor5(tensorFeatSecond[-1]))
        tensorFeatSecond.append(self.moduleExtractor6(tensorFeatSecond[-1]))

        objectEstimate = self.moduleDecoderFirst(tensorFeatFirst[-1], tensorFeatSecond[-1])
        objectEstimate = self.moduleFiv(tensorFeatFirst[-2], tensorFeatSecond[-2], objectEstimate)
        objectEstimate = self.moduleFou(tensorFeatFirst[-3], tensorFeatSecond[-3], objectEstimate)
        objectEstimate = self.moduleThr(tensorFeatFirst[-4], tensorFeatSecond[-4], objectEstimate)
        objectEstimate = self.moduleTwo(tensorFeatFirst[-5], tensorFeatSecond[-5], objectEstimate)

        return objectEstimate['tensorFlow'] + self.moduleRefiner(objectEstimate['tensorFeat'])

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        class Extractor(torch.nn.Module):
            def __init__(self):
                super(Extractor, self).__init__()

                self.moduleOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

            # end

            def forward(self, tensorInput):
                tensorOne = self.moduleOne(tensorInput)
                tensorTwo = self.moduleTwo(tensorOne)
                tensorThr = self.moduleThr(tensorTwo)
                tensorFou = self.moduleFou(tensorThr)
                tensorFiv = self.moduleFiv(tensorFou)
                tensorSix = self.moduleSix(tensorFiv)

                return [tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix]
            # end

        # end

        class Decoder(torch.nn.Module):
            def __init__(self, intLevel):
                super(Decoder, self).__init__()

                intPrevious = \
                [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][
                    intLevel + 1]
                intCurrent = \
                [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][
                    intLevel + 0]

                if intLevel < 6: self.moduleUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2,
                                                                              kernel_size=4, stride=2, padding=1)
                if intLevel < 6: self.moduleUpfeat = torch.nn.ConvTranspose2d(
                    in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2,
                    padding=1)
                if intLevel < 6: self.dblBackward = [None, None, None, 5.0, 2.5, 1.25, 0.625, None][intLevel + 1]

                self.moduleOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3,
                                    stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3,
                                    stride=1, padding=1)
                )

            # end

            def forward(self, tensorFirst, tensorSecond, objectPrevious):
                tensorFlow = None
                tensorFeat = None

                if objectPrevious is None:
                    tensorFlow = None
                    tensorFeat = None
                    corr = correlation.FunctionCorrelation(tensorFirst=tensorFirst, tensorSecond=tensorSecond)
                    tensorVolume = torch.nn.functional.leaky_relu(input=corr, negative_slope=0.1, inplace=False)

                    tensorFeat = torch.cat([tensorVolume], 1)

                elif objectPrevious is not None:
                    tensorFlow = self.moduleUpflow(objectPrevious['tensorFlow'])
                    tensorFeat = self.moduleUpfeat(objectPrevious['tensorFeat'])

                    warped = Backward(tensorInput=tensorSecond, tensorFlow=tensorFlow * self.dblBackward)
                    corr = correlation.FunctionCorrelation(tensorFirst=tensorFirst, tensorSecond=warped)

                    tensorVolume = torch.nn.functional.leaky_relu(input=corr, negative_slope=0.1, inplace=False)

                    tensorFeat = torch.cat([tensorVolume, tensorFirst, tensorFlow, tensorFeat], 1)

                # end

                tensorFeat = torch.cat([self.moduleOne(tensorFeat), tensorFeat], 1)
                tensorFeat = torch.cat([self.moduleTwo(tensorFeat), tensorFeat], 1)
                tensorFeat = torch.cat([self.moduleThr(tensorFeat), tensorFeat], 1)
                tensorFeat = torch.cat([self.moduleFou(tensorFeat), tensorFeat], 1)
                tensorFeat = torch.cat([self.moduleFiv(tensorFeat), tensorFeat], 1)

                tensorFlow = self.moduleSix(tensorFeat)

                return {
                    'tensorFlow': tensorFlow,
                    'tensorFeat': tensorFeat
                }
            # end

        # end

        class Refiner(torch.nn.Module):
            def __init__(self):
                super(Refiner, self).__init__()

                self.moduleMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128,
                                    kernel_size=3, stride=1, padding=1, dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
                )

            # end

            def forward(self, tensorInput):
                return self.moduleMain(tensorInput)
            # end

        # end

        self.moduleExtractor = Extractor()

        self.moduleTwo = Decoder(2)
        self.moduleThr = Decoder(3)
        self.moduleFou = Decoder(4)
        self.moduleFiv = Decoder(5)
        self.moduleSix = Decoder(6)

        self.moduleRefiner = Refiner()

    # end

    def forward(self, tensorFirst, tensorSecond):
        tensorFirst = self.moduleExtractor(tensorFirst)
        tensorSecond = self.moduleExtractor(tensorSecond)

        objectEstimate = self.moduleSix(tensorFirst[-1], tensorSecond[-1], None)
        objectEstimate = self.moduleFiv(tensorFirst[-2], tensorSecond[-2], objectEstimate)
        objectEstimate = self.moduleFou(tensorFirst[-3], tensorSecond[-3], objectEstimate)
        objectEstimate = self.moduleThr(tensorFirst[-4], tensorSecond[-4], objectEstimate)
        objectEstimate = self.moduleTwo(tensorFirst[-5], tensorSecond[-5], objectEstimate)

        return objectEstimate['tensorFlow'] + self.moduleRefiner(objectEstimate['tensorFeat'])
# end


# end

moduleNetwork = Network().cuda().eval()
newNetwork = NewNetwork().cuda().eval()
weights = torch.load('./network-' + arguments_strModel + '.pytorch')
moduleNetwork.load_state_dict(weights)
newNetwork.moduleExtractor1.moduleOne = copy.deepcopy(moduleNetwork.moduleExtractor.moduleOne)
newNetwork.moduleExtractor2.moduleTwo = copy.deepcopy(moduleNetwork.moduleExtractor.moduleTwo)
newNetwork.moduleExtractor3.moduleThr = copy.deepcopy(moduleNetwork.moduleExtractor.moduleThr)
newNetwork.moduleExtractor4.moduleFou = copy.deepcopy(moduleNetwork.moduleExtractor.moduleFou)
newNetwork.moduleExtractor5.moduleFiv = copy.deepcopy(moduleNetwork.moduleExtractor.moduleFiv)
newNetwork.moduleExtractor6.moduleSix = copy.deepcopy(moduleNetwork.moduleExtractor.moduleSix)

newNetwork.moduleDecoderFirst.featureProcessor.moduleOne = copy.deepcopy(moduleNetwork.moduleSix.moduleOne)
newNetwork.moduleDecoderFirst.featureProcessor.moduleTwo = copy.deepcopy(moduleNetwork.moduleSix.moduleTwo)
newNetwork.moduleDecoderFirst.featureProcessor.moduleThr = copy.deepcopy(moduleNetwork.moduleSix.moduleThr)
newNetwork.moduleDecoderFirst.featureProcessor.moduleFou = copy.deepcopy(moduleNetwork.moduleSix.moduleFou)
newNetwork.moduleDecoderFirst.featureProcessor.moduleFiv = copy.deepcopy(moduleNetwork.moduleSix.moduleFiv)
newNetwork.moduleDecoderFirst.flowProcessor.moduleSix = copy.deepcopy(moduleNetwork.moduleSix.moduleSix)

newNetwork.moduleTwo = copy.deepcopy(moduleNetwork.moduleTwo)
newNetwork.moduleThr = copy.deepcopy(moduleNetwork.moduleThr)
newNetwork.moduleFou = copy.deepcopy(moduleNetwork.moduleFou)
newNetwork.moduleFiv = copy.deepcopy(moduleNetwork.moduleFiv)

newNetwork.moduleRefiner = copy.deepcopy(moduleNetwork.moduleRefiner)
##########################################################

def estimate(tensorFirst, tensorSecond):
    assert (tensorFirst.size(1) == tensorSecond.size(1))
    assert (tensorFirst.size(2) == tensorSecond.size(2))

    intWidth = tensorFirst.size(2)
    intHeight = tensorFirst.size(1)

    assert (
                intWidth == 1024)  # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    assert (
                intHeight == 436)  # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    tensorPreprocessedFirst = tensorFirst.cuda().view(1, 3, intHeight, intWidth)
    tensorPreprocessedSecond = tensorSecond.cuda().view(1, 3, intHeight, intWidth)

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

    tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorPreprocessedFirst,
                                                              size=(intPreprocessedHeight, intPreprocessedWidth),
                                                              mode='bilinear', align_corners=False)
    tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorPreprocessedSecond,
                                                               size=(intPreprocessedHeight, intPreprocessedWidth),
                                                               mode='bilinear', align_corners=False)

    tensorFlow = 20.0 * torch.nn.functional.interpolate(
        input=newNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond), size=(intHeight, intWidth),
        mode='bilinear', align_corners=False)

    tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    return tensorFlow[0, :, :, :].cpu()

# end

##########################################################

if __name__ == '__main__':
    tensorFirst = torch.FloatTensor(
        numpy.array(PIL.Image.open(arguments_strFirst))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (
                    1.0 / 255.0))
    tensorSecond = torch.FloatTensor(
        numpy.array(PIL.Image.open(arguments_strSecond))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (
                    1.0 / 255.0))

    tensorOutput = estimate(tensorFirst, tensorSecond)

    objectOutput = open(arguments_strOut, 'wb')

    numpy.array([80, 73, 69, 72], numpy.uint8).tofile(objectOutput)
    numpy.array([tensorOutput.size(2), tensorOutput.size(1)], numpy.int32).tofile(objectOutput)
    numpy.array(tensorOutput.numpy().transpose(1, 2, 0), numpy.float32).tofile(objectOutput)

    objectOutput.close()
# end
