# U-net
本项目为基于U-net的医学影像分割的pytorch实现
## Main Code
本实验环境如下：
    * windows 10 64bit
    * python 3.7.9
    * cuda 11.4
    * pytorch 1.9.2
## 使用前必读
本项目来自于： 
  
        大佬的博客：https://blog.csdn.net/ykben/article/details/103118619)  
        大佬的GitHub：https://github.com/Czt1998/U-net  
           
因为在研究U-NET网络的时候发现该项目，所以下载并研究，发现原项目并不能即时运行使用，因此在对源代码进行详细研读和仔细批注之后，修正了原本的问题，包括：  
        1.对原有代码进行详细注释，对关键性的函数调用语句进行详细解读  
        2.修正原有代码读取图片文件不正确的问题  
        3.完善原有的图片集合，在训练完成，进行预测之后会在test文件夹下自动生成预测的图片文件    
          
通过运行main.py即可实现训练，并对test中图片进行分割。
  
注释：模型数据集合来自ISBI Challenge2012的数据集合。

