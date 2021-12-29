import torch
from torch import nn
from collections import OrderedDict

DISPLAY_EVERY = 100

class SilNet(nn.Module):
    """
    A class used instantiate SilNet

    ...

    Inheritance
    -----------
    Inherits from : pytorch nn module

    Methods
    -------
    _init__(self, ic, mc, oc) : instantiates local variables, calls super init, 
    creates neural network architecture
    CBLK(ic, oc){OrderedDict} : creates a convolutional block 
    DBLK(ic, mc, oc){OrderedDict} : creates a deconvolutional block
    """

    def __init__(self):
        super().__init__()
        
        self.cblk_branch1 = nn.Sequential(
            self.CBLK(19, 64),
            self.CBLK(64, 128),
            self.CBLK(128, 256),
            self.CBLK(256, 512),
            self.CBLK(512, 512)
        )


        self.cblk_branch2_1 = self.CBLK(18, 64)
        self.cblk_branch2_2 = self.CBLK(64, 128)
        self.cblk_branch2_3 = self.CBLK(128, 256)
        self.cblk_branch2_4 = self.CBLK(256, 512)
        self.cblk_branch2_5 = self.CBLK(512, 512)

        self.out1 = self.DBLK(1024, 512, 512)
        self.out2 = self.DBLK(1024, 512, 256)
        self.out3 = self.DBLK(512, 256, 128)
        self.out4 = self.DBLK(256, 128, 64)
        self.out5 = self.DBLK(128, 64, 64)

        self.pred = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Conv2d(64, 2, 1, 1, 0),
            nn.Sigmoid()
        )

    def CBLK(self, ic, oc):
        """
        Parameters
        ----------
        ic : input channels
        oc : output channels
        """
        return nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(ic, oc, 3, 1, 1)),
                ('inorm1', nn.InstanceNorm2d(oc)),
                ('relu1', nn.LeakyReLU(0.2)),
                ('conv2', nn.Conv2d(oc, oc, 3, 1, 1)),
                ('inorm2', nn.InstanceNorm2d(oc)),
                ('relu2', nn.LeakyReLU(0.2)),
                ('conv3', nn.Conv2d(oc, oc, 4, 2, 1)),
                ('inorm3', nn.InstanceNorm2d(oc)),
                ('relu3', nn.LeakyReLU(0.2)),
        ]))

    def DBLK(self, ic, mc, oc):
        """
        Parameters
        ----------
        ic : input channels
        mc : medium channels
        oc : output channels
        """
        return nn.Sequential(
            OrderedDict([
            ('conv1', nn.Conv2d(ic, mc, 3, 1, 1)),
            ('inorm1', nn.InstanceNorm2d(mc)),
            ('relu1', nn.LeakyReLU(0.2)),
            ('conv2', nn.Conv2d(mc, mc, 3, 1, 1)),
            ('inorm2', nn.InstanceNorm2d(mc)),
            ('relu2', nn.LeakyReLU(0.2)),
            ('conv3', nn.Conv2d(mc, mc, 3, 1, 1)),
            ('inorm3', nn.InstanceNorm2d(mc)),
            ('relu3', nn.LeakyReLU(0.2)),
            ('conv3', nn.ConvTranspose2d(mc, oc, 4, 2, 1)),
            ('relu3', nn.LeakyReLU(0.2)),
        ]))

    def forward(self, data):
        M1 = data['M1'].unsqueeze(dim=1)
        BP1 = data['BP1']
        input_branch1 = torch.cat((M1.to(torch.float32), BP1), dim=1).cuda()
        input_branch2 = data['BP2'].cuda()

        ## Branch 1 - sequential
        x_branch1 = self.cblk_branch1(input_branch1)

        ## Branch 2 - outputs concat to DBLK
        x_branch2_1 = self.cblk_branch2_1(input_branch2)
        x_branch2_2 = self.cblk_branch2_2(x_branch2_1)
        x_branch2_3 = self.cblk_branch2_3(x_branch2_2)
        x_branch2_4 = self.cblk_branch2_4(x_branch2_3)
        x_branch2_5 = self.cblk_branch2_5(x_branch2_4)

        x_out = self.out1(torch.cat((x_branch1, x_branch2_5), dim=1))
        x_out = self.out2(torch.cat((x_out, x_branch2_4), dim=1))
        x_out = self.out3(torch.cat((x_out, x_branch2_3), dim=1))
        x_out = self.out4(torch.cat((x_out, x_branch2_2), dim=1))
        x_out = self.out5(torch.cat((x_out, x_branch2_1), dim=1))

        y_pred = self.pred(x_out)

        return y_pred