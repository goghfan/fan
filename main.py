from matplotlib.transforms import Transform
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from unet import Unet
from DataHelper import *
from tqdm import tqdm
import numpy as np
import skimage.io as io

PATH = './modle/unet_model.pt'

#是否使用cuda（gpu）进行训练
#对于老显卡不推荐进行gpu训练，容易显存溢出
#使用cpu速度较慢
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#tensforms是torchvision中的函数，用于常见的图形变换。
#这里的transforms.Compose用于串联多个图片变换操作
#transforms.ToTensor()将shape为(H, W, C)的nump.ndarray或img转为shape为(C, H, W)的tensor，其将每一个数值归一化到[0,1]，其归一化方法比较简单，直接除以255即可。
#transforms.Normalize(mean,std,inplace) 逐channel的对图像进行标准化（均值变为0，标准差变为1），可以加快模型的收敛
x_transforms = transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])

#
y_transforms = transforms.ToTensor()

def train_model(model,criterion,optimizer,dataload,num_epochs=10):
    best_model = model
    min_loss= 100000
    for epoch in range(num_epochs):
        print('Epoch{}/{}'.format(epoch,num_epochs-1))
        dt_size=len(dataload.dataset)
        epoch_loss=0
        step =0
        for x,y in tqdm(dataload):
        #Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，
        #用户只需要封装任意的迭代器 tqdm(iterator)。
            step+=1
            inputs = x.to(device)
            lables = y.to(device)
            #清空梯度
            optimizer.zero_grad()
            #前向传播
            outputs=model(inputs)

            loss = criterion(outputs,lables)
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
    print("epoch %d loss:%0.3f" % (epoch, epoch_loss/step))
    if(epoch_loss/step)<min_loss:
        min_loss=(epoch_loss/step)
        best_model=model
    torch.save(best_model.state_dict(),PATH)
    return best_model



#训练模型
def train():
    #从Unet中引用Unet网络
    model = Unet(1,1).to(device)
    batch_size=1
    criterion = nn.BCEWithLogitsLoss()
    #本质上和nn.BCELoss()没有区别，
    #只是在BCELoss上加了个logits函数(也就是sigmoid函数)
    #也就是二值交叉熵做了sigmod，数值计算稳定性更好    
    # #model.parameters()保存的是Weights和Bais参数的值。
    optimizer = optim.Adam(model.parameters())
    #优化函数，model.parameters()为该实例中可优化的参数，lr为参数优化的选项（学习率等）
    train_dataset = TrainDataset("dataset/train/image", "dataset/train/label", transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
    #ps.这里使用num_works数量量力而行，太多容易出候选页面不够的问题，可以将num_workers改为0
    #DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, 
    #num_workers=0, collate_fn=default_collate, pin_memory=False, 
    #drop_last=False)
    #dataset:加载的数据集
    #batch_size:batch size
    #shuffle: 是否打乱数据
    #sample：样品抽样
    #num_works：使用多进程加载的进程数
    #collate_fn:如何将多个样本数据拼接成一个batch，一般使用默认的拼接方式即可
    #pin_memory:是否将数据保存在pin memory区中，pin memory中的数据转换gpu中更快
    #drop_last:dataset中的数据个数可能不是batch_size的整数倍，drop_last为True会将多出来不足一个batch的数据丢弃
    train_model(model, criterion, optimizer, dataloaders)
def test():
    model = Unet(1,1)
    model.load_state_dict(torch.load(PATH))
    test_dataset = TestDataset("dataset/test", transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(test_dataset, batch_size=1)
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()
    with torch.no_grad():
        for index, x in enumerate(dataloaders):
            y = model(x)
            img_y = torch.squeeze(y).numpy()
            img_y = img_y[:, :, np.newaxis]
            img = labelVisualize(2, COLOR_DICT, img_y) if False else img_y[:, :, 0]
            io.imsave("./dataset/test/" + str(index) + "_predict.png", img)
            plt.pause(0.01)
        plt.show()
if __name__=='__main__':
    print("开始训练")
    train()
    print('训练完成，保存模型')
    print('-'*20)
    print('开始预测')
    test()