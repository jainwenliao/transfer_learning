import torch
import torchvision
from torchvision import datasets, transforms,models
import torchvision.transforms
from torch.autograd import Variable
import time
import cv2#matplotlib没法用，改用opencv

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'#gpu指定

data_dir = "DogsVSCats"

data_transform = {
    x:transforms.Compose(
        [   
            transforms.Resize([64,64]), transforms.ToTensor()#Scale会报错提示改用resize，这里将图片都变为64*64大小格式
        ]
    )
    for x in ["train","valid"]
}

image_datasets = {
    x:datasets.ImageFolder(root = os.path.join(data_dir, x), 
    transform = data_transform[x]
    )
    for x in ["train","valid"]
}

#z这里对图片进行了one-hot encoding处理，其中cat为0，dog为1
dataloader = {
    x:torch.utils.data.DataLoader(
    dataset = image_datasets[x],
    batch_size = 16, 
    shuffle = True
    ) 
    for x in ["train","valid"]
    }

x_examples, y_examples = next(iter(dataloader['train']))

#print(u"x_example 个数{}".format(len(x_examples)))
#print(u"y_example 个数{}".format(len(y_examples)))

index_classses = image_datasets['train'].class_to_idx#one-hot对应类别
#rint(index_classses)

#将原始标签结果存储在example_clsaaes中
example_classes = image_datasets['train'].classes
#print(example_classes)

#绘制一个批次的图片
#img = torchvision.utils.make_grid(x_examples)
#img = img.numpy().transpose([1,2,0])
#print([example_classes[i] for i in y_examples])
#cv2.imshow('dogs_vs_cats',img)
#key_pressed = cv2.waitKey(0)


use_gpu = torch.cuda.is_available()
print(use_gpu)
model = models.vgg16(pretrained = True)
#print(model)

for parma in model.parameters():
    parma.requires_grad = False#对应参数不计算梯度

#重新设计分类器即新的全连接层
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(25088,4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(4096,4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(p = 0.5),
    torch.nn.Linear(4096, 2)
)

#定义损失函数和优化参数
loss_f = torch.nn.CrossEntropyLoss()

optimitzer = torch.optim.Adam(model.classifier.parameters(),lr = 0.00001)

if use_gpu:
    model = model.cuda()


loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(),lr = 0.00001)


epoch_n = 5
time_open =time.time()

for epoch in range(epoch_n):  
    print('epoch{}/{}'.format(epoch, epoch_n - 1))
    print("-"*10)

    for phase in ["train","valid"]:  
        if phase == "train":  
            print("training...")
            model.train(True)

        else:  
            print("Validing...")
            model.train(False)

        running_loss = 0.0
        running_corrects = 0

        for batch, data in enumerate(dataloader[phase], 1):  
            X, y = data
            if use_gpu:  
                X, y =Variable(X.cuda()), Variable(y.cuda())

            else:  
                X, y =Variable(X), Variable(y)

            y_pred = model(X)

            _,pred = torch.max(y_pred.data, 1)

            optimizer.zero_grad()

            loss = loss_f(y_pred,y)

            if phase == "train":  

                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            running_corrects += torch.sum(pred == y.data)

            if  batch % 500 == 0 and phase == "train":  
                print(
                    "batch{}, train loss:{:.4f},train acc:{:.4f}".format(
                    batch, running_loss/batch, 100.0*running_corrects/(16*batch)
                    )
                )

        epoch_loss = running_loss*16 / len(image_datasets[phase])
        epoch_acc = 100*running_corrects/len(image_datasets[phase])

print("{} loss:{:.4f} acc:{:.4f}%".format(phase,epoch_loss,epoch_acc))

time_end = time.time() - time_open 
print(time_end)

