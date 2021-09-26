# coding:utf-8

from torch import nn


class NetG(nn.Module):
    '''
    生成器定义
     '''

    def __init__(self,opt):

        super(NetG,self).__init__()
        ngf=opt.ngf  #生成器feature map数


        self.main=nn.Sequential(
            # 输入是一个nz维度的噪音，我们可以认为它是一个1*1*nz的feature map
            nn.ConvTranspose2d(opt.nz,ngf*8,4,1,0,bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            #上一步的输出形状：(ngf*8) *4 *4

            nn.ConvTranspose2d(ngf*8,ngf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            #上一步的输出形状：(ngf*4)*8*8


            nn.ConvTranspose2d(ngf*4,ngf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            #上一步的输出形状：(ngf*2)*16*16


            nn.ConvTranspose2d(ngf*2,ngf,4,2,1,bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            #上一步的输出形状：(ngf)*32*32

            nn.ConvTranspose2d(ngf,3,5,3,1,bias=False),
            nn.Tanh() #输出范围负1至1，故而采用Tanh
            #输出形状为：3*96*96
            )


    def forward(self,imm):
        return self.main(imm)



class NetD(nn.Module):

    '''
    判别器定义
    '''

    def __init__(self,opt):
        super(NetD,self).__init__()
        ndf=opt.ndf
        self.main=nn.Sequential(

            #输入 3*96*96
            nn.Conv2d(3,ndf,5,3,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            #输出 (ndf)*32*32

            nn.Conv2d(ndf,ndf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,inplace=True),
            #输出为(ndf*2)*16*16


            nn.Conv2d(ndf*2,ndf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2,inplace=True),
            #输出为(ndf*4)*8*8


            nn.Conv2d(ndf*4,ndf*8,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2,inplace=True),
            #输出为(ndf*8)*4*4

            nn.Conv2d(ndf*8,1,4,1,0,bias=False),
            nn.Sigmoid()  #输出一个数(概率)
            )
        

    def forward(self,imm):
        return self.main(imm).view(-1)
            
