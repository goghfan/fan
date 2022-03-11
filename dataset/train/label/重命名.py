from locale import strxfrm
import os,sys                       #导入模块
def add_prefix_subfolders():        #定义函数名称
    mark = '_mask'                  #准备添加的前缀内容
    old_names = os.listdir( path )  #取路径下的文件名，生成列表
    for old_name in old_names:
        if old_name.endswith('.tif_mask'):
            str=old_name.split('.')
            str[1]='.tif'
            str[0]=str[0]+'_mask'
            strx=str[0]+str[1]
            os.rename(os.path.join(path,old_name),os.path.join(path,strx))
    for old_name in old_names:
        if old_name.endswith('.tif_mask'):
            print(old_name)
if __name__ == '__main__': 
        path = 'D:/BaiduNetdiskDownload/ISBI Challenge 2012/train/label'   #运行程序前，记得修改主文件夹路径！
        add_prefix_subfolders()            #调用定义的函数

        