import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self,in_ch,out_ch) :
        #继承DoubleConv父类
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1),
            nn.BatchNorm2d(out_ch),
            #在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，
            #这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.Relu(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.Relu(inplace=True)

        )
    def forward(self,input):
        return self.conv(input)

    def forward(self,input):
        return self.conv(input)
class Unet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Unet,self).__init__()
        #nn.MaxPool2d(kernel_size,stride=None,padding=0,dilation=1,
        # return_indices=False,ceil_model=False)
        #kernel_size() max_pool的窗口大小
        #stride移动的步长，默认kernel_sizeza
        self.conv1 = DoubleConv(in_ch,64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2=DoubleConv(64,128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(128,256)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = DoubleConv(512, 1024)
#--------------------------反卷积---------------------------------------
#class torch.nn.ConvTranspose2d(
# in_channels, out_channels, kernel_size, stride=1, padding=0, 
# output_padding=0, groups=1, bias=True, dilation=1)
#in_channels:输入信号的通道数
#out_channels:卷积产生的通道数
#kerer_size：卷积核的大小
#stride：卷积的步长，也就是将输入扩大的倍数
#padding：输入的每一条边补充0的层数，高宽都增加2*padding
#output_padding:输出边补充0的层数，高宽都增加padding
#groups：从输入通道到输出通道的阻塞连接数
#bias：如果bias=True，添加偏置
#dilation:卷积核元素之间的间距
#output=（intput-1)*stride + outputpadding - 2*padding + kernelsize
        self.up6 = nn.ConvTranspose2d(1024,521,2,stride=2)
        #512=(512-1)*2+0-2*0+2
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10=nn.Conv2d(64,out_ch,1)
        #class torch.nn.Conv2d(
        # in_channels, out_channels, kernel_size, 
        # stride=1, padding=0, dilation=1, groups=1, bias=True)
    def forward(self,x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        return c10


        

