import os
import shutil
#对训练集中的猫狗图片进行分类

def redistribution():
    data_file = os.listdir('./DogsVSCats/Dataset')#浏览目录
    #筛选，把data_file中前三个标签为dog和cat分开
    dogs_file = list(filter(lambda x: x[:3] == 'dog', data_file))
    cats_file = list(filter(lambda x: x[:3] == 'cat', data_file))
    
    #新的存放地址
    data_root = './DogsVSCats/'
    train_root = './DogsVSCats/train'
    valid_root = './DogsVSCats/valid'

    for i in range(len(cats_file)):
        #原始图片地址
        image_path = data_root + 'Dataset/' + cats_file[i]
        #取90%到训练集，10%到验证集
        if i < len(cats_file) * 0.9:
            new_path = train_root + '/cat/' + cats_file[i]
        else:
            new_path = valid_root + '/cat/' + cats_file[i]
        #shutil.move用于移动图片
        shutil.move(image_path, new_path)
 
    for i in range(len(dogs_file)):
        image_path = data_root + 'Dataset/' + dogs_file[i]
        if i < len(dogs_file) * 0.9:
            new_path = train_root + '/dog/' + dogs_file[i]
        else:
            new_path = valid_root + '/dog/' + dogs_file[i]
        shutil.move(image_path, new_path)
 
if __name__ == '__main__':
    redistribution()