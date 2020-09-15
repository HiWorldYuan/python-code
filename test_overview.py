'''
python功能汇总
@author jyuan
'''
#1.python下载文件
import urllib.request

url =  "http://csegroups.case.edu/sites/default/files/bearingdatacenter/files/Datafiles/118.mat"
downPath = "./1.mat"
urllib.request.urlretrieve(url,downPath)

#2.迭代器
#排列
from itertools import permutations
items=['a','b','c']
for p in permutations(items):   #for p in permutations(items,2):
    print(p)

#组合
from itertools import combinations
items=['a','b','c']
for p in combinations(items,2):
    print(p)

#索引-值
items=['a','b','c']
for row,item in enumerate(items):
    print(row,item)

#迭代多个序列
x=[1,2,3,4,5]
y=[6,7,8,9,10,11]
for a,b in zip(x,y):
    print(a,b)

#dict(zip(keys,values))
#chain(a,b,c)
#heapq.merge(a,b,c)

#3.归一化
import numpy as np
x=np.array([[1,2,3],[4,5,6]])
x=(x-np.min(x))/(np.max(x)-np.min(x))
print(x)

#4.导入自己的模块
import numpy as np
import sys
sys.path.append('../')
print(sys.path)
from pytorch_test.RNN.myDataset import normalizaition
x=np.array([[1,2,3],[4,5,6]])
x=normalizaition(x)
print(x)

#5.混淆矩阵
import itertools
import matplotlib.pyplot as plt
import numpy as np
# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes)+1)-0.5
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

cnf_matrix = np.array([[8707, 64, 731, 164, 45],
    [1821, 5530, 79, 0, 28],
    [266, 167, 1982, 4, 2],
    [691, 0, 107, 1930, 26],
    [30, 0, 111, 17, 42]])
attack_types = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']

plot_confusion_matrix(cnf_matrix, classes=attack_types, normalize=False, title='Normalized confusion matrix')

#6.python导入数据（txt，每行一条数据）
file1 = open("./test/400.txt")
f = []
for line in file1.readlines():
    f.append(line.strip('\n'))


#7.计算多分类的P，R，F1值
precision = precision_score(test_y,pred_y,labels=[0,1,2,3,4,5,6,7,8,9],average='macro')
recall = recall_score(test_y,pred_y,labels=[0,1,2,3,4,5,6,7,8,9],average='macro')
f1 = f1_score(test_y,pred_y,labels=[0,1,2,3,4,5,6,7,8,9],average='macro')

#8.matplotlib 画动态图
'''
在交互模式下：

1、plt.plot(x)或plt.imshow(x)是直接出图像，不需要plt.show()

2、如果在脚本中使用ion()命令开启了交互模式，没有使用ioff()关闭的话，则图像会一闪而过，并不会常留。要想防止这种情况，需要在plt.show()之前加上ioff()命令。

在阻塞模式下：

1、打开一个窗口以后必须关掉才能打开下一个新的窗口。这种情况下，默认是不能像Matlab一样同时开很多窗口进行对比的。

2、plt.plot(x)或plt.imshow(x)是直接出图像，需要plt.show()后才能显示图像
'''
    import matplotlib.pyplot as plt
    plt.ion()    # 打开交互模式
    # 同时打开两个窗口显示图片
    plt.figure()  #图片一
    plt.imshow(i1)
    plt.figure()    #图片二
    plt.imshow(i2)
    # 显示前关掉交互模式
    plt.ioff()
    plt.show()


#9.numpy.array保存到excal
import xlwt
def saveBest(num):
    wbk = xlwt.Workbook()
    sheet = wbk.add_sheet('sheet 1')
    # indexing is zero based, row then column
    for i in range(num.shape[0]):
        for j in range(num.shape[1]):
            sheet.write(i,j,num[i,j].astype(str))
    wbk.save('test2.xls')  #默认保存在当前根目录

#10.数据保存到csv文件
import csv            
with open(r"C:\Users\Jensen\Desktop\python_py\pytorch_test\prognostics\bearing_1.csv", 'w', newline='') as csvfile:
    writer  = csv.writer(csvfile)
    for i in range(len(data)):
        writer.writerow(data[i:i+1])  

#11.matplotlib相关参数设置
# 中设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=23) 

# 设置横纵坐标的名称以及对应字体格式
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 30,
         }
plt.xlabel('round', font2)
plt.ylabel('value', font2)
