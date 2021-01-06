import torch
import torchvision
from torchvision import datasets, transforms
import torchvision.transforms
import os
import time
import cv2#matplotlib没法用，改用opencv

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
img = torchvision.utils.make_grid(x_examples)
img = img.numpy().transpose([1,2,0])
print([example_classes[i] for i in y_examples])
#cv2.imshow('dogs_vs_cats',img)
#key_pressed = cv2.waitKey(0)



class Models(torch.nn.Module):

    def __init__(self):
        super(Models,self).__init__()
        self.Conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),


            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),


            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),


            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
    

        self.Classes = torch.nn.Sequential(
            torch.nn.Linear(4*4*512,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024,2)
        )
    
    def forward(self,input):
        x = self.Conv(input)
        x = x.view(-1,4*4*512)
        x = self.Classes(x)
        return x

model = Models()
#print(model)

#定义损失函数和优化参数
loss_f = torch.nn.CrossEntropyLoss()

optimitzer = torch.nn.Adam(model.parameters(),lr = 0.00001)

use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()

epoch_n = 10
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
        x, y = data
        x, y =Variable(x), Variable(y)

    y_pred = model(x)

    _,pred = torch.max(y_pred.data, 1)

    optimizer.zero_grad()

    loss = loss_f(y_pred,y)

    if phase == "train":  
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        running_corrects += torch.sum(pred == y.data)

    if  batch%500 = 0 and phase ="train":  
        print("batch{}, train loss:{:.4f},train acc:{:.4f}".format(batch, running_loss/batch, 100*running_corrects/(16*batch)))

epoch_loss = running_loss*16 / len(image_datasets[phase])
epoch_acc = 100*running_corrects/len(image_datasets[phase])

print("{} loss:{:.4f} acc:{:.4f}%".format(phase,epoch_loss,epoch_acc))

timr_end = time.time() - time_open 
print(time_end)

