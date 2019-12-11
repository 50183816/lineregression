# _*_ codig utf8 _*_

import numpy as np

'''
x1  x2  x3  x4  y
1   1   1   1   1
1   2   2   1   1
2   0   5   3   2
3   2   6   7   3
4   3   1   1   2
5   2   1   1   1
7   8   6   5   3
0   1   2   8   2
1   6   4   2   3
4   3   1   5   2
-----------------------------------
y={1,2,3}
py1 = 0.3
PY2 = 0.4
PY3 = 0.3

P(X1=1|Y=1) = 2/3
P(x1=5) = 1/3

P(x2=1|y=1) = 1/3
p(x2=2|y=1)=2/3
'''
data = np.array([
    [1, 1, 1, 1, 1],
    [1, 2, 2, 1, 1],
    [2, 0, 5, 3, 2],
    [3, 2, 6, 7, 3],
    [4, 3, 1, 1, 2],
    [5, 2, 1, 1, 1],
    [7, 8, 6, 5, 3],
    [0, 1, 2, 8, 2],
    [1, 6, 4, 2, 3],
    [4, 3, 1, 5, 2]
])

ys = np.unique(data[:, 4])
py = {}
totalcount = len(data)
# 计算类别y的概率
for y in ys:
    c = len([d for d in data if d[4] == y])
    py[y] = (c + 1) / (totalcount + len(ys))
print(py)
# 对每个类别，分别计算各特征值的条件概率
pxy = {}  # 字典： {y:{x:条件概率}}
pxy_numbers = {}
for y in ys:
    tmp = np.array([d for d in data if d[4] == y])  # 训练样本中类别为y的样本
    pxy[y] = []
    pxy_numbers[y] = []
    for i in np.arange(4):  # 对每个属性计算条件概率
        pxi = 0
        xs = np.unique(tmp[:, i])  # 属性i中的可能取值
        p_of_x_values = {}
        p_of_x_numbers = {}
        for x in xs:  # 处理Xi的每一个可能取值
            c = len([tx for tx in tmp if tx[i] == x])
            #pxi = (c + 1) / (len(tmp) + len(xs))
            pxi = (len(tmp) + len(xs)) #平滑處理，防止出現屬性值沒有而導致概率為0的情況，叫拉普拉斯修正
            p_of_x_values[x] = pxi
            p_of_x_numbers[x] = c+1 #平滑處理，防止出現屬性值沒有而導致概率為0的情況，叫拉普拉斯修正
        pxy[y].append((len(tmp) + len(xs)))
        pxy_numbers[y].append(p_of_x_numbers)

print(pxy)
print(pxy_numbers)
# 预测
test = [40,3,1,1]
for y in ys:
    test_pxi = py[y]
    for i, px in enumerate(pxy_numbers[y]):
        c= px.get(test[i],0.05)
        total = pxy[y][i]
        test_pxi *= c/total
    print('class={},prob={}'.format(y, test_pxi))
