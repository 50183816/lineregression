# _*_ codig utf8 _*_
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def entropy(p):
    return -1 * np.sum(pi * np.log2(pi) for pi in p)


# 计算DataFramne中某属性按某值划分的概率分布
def calculate_prob(x, attributename, value):
    pass


# 划分数据，排除掉atts中列出的属性
def split_data(x, atts, y_label, tree, index=1):
    if len(x) == 0:
        return
    labelas = [c for c in x.columns if c not in atts]
    print('lablas={}'.format(labelas))
    yvalues = set(x[y_label])
    if len(yvalues) == 1:
        return

    ygroupsize = x.groupby(y_label).size()
    total = ygroupsize.sum()
    yprob = [ygroupsize[c] / total for c in yvalues]
    print('根节点的概率分布：{}'.format(yprob))
    hd = entropy(yprob)
    g0 = 0
    node = {}
    for labela in labelas:
        if x[labela].dtype == 'object':  # 字符类型属性的判断
            values = set(x[labela])
            # 字符串类型数据，分别计算
            groupsize = x.groupby(labela).size()
            total = groupsize.sum()
            prob = [groupsize[c] / total for c in values]
            print(prob)
            h = []
            # 计算Y的信息熵
            children = []
            for v in values:
                x_of_v = x[x[labela] == v]
                yvalues = set(item for item in x_of_v[y_label])
                ygroupsize = x_of_v.groupby(y_label).size()
                total = ygroupsize.sum()
                yprob = [ygroupsize[c] / total for c in yvalues]
                print('{} ={} 节点的概率分布：{}'.format(labela, v, yprob))
                hv = entropy(yprob)
                print('{} ={} 节点的信息熵：{}'.format(labela, v, hv))
                h.append(hv)
                children.append(x_of_v)
            gain = hd - np.sum(pi * hi for pi, hi in zip(prob, h))
            print('{} 划分的信息增益量：{}'.format(labela, gain))

        elif is_numeric_dtype(x[labela].dtype):
            children = []
            # 处理数字类型属性，需要找到一个划分的阀值
            # print('处理{},类型为{}'.format(labela,x[labela].dtype))
            mean = np.mean(x[labela])
            mead = np.median(x[labela])
            print('{}的平均数是{},中位数是{}'.format(labela, mean, mead))
            x_of_mead = x[x[labela] < mead]  # 按中位数划分数据，小于中位数
            yvalues = set(item for item in x_of_mead[y_label])
            ygroupsize = x_of_mead.groupby(y_label).size()
            total = ygroupsize.sum()
            yprob = [ygroupsize[c] / total for c in yvalues]
            p0 = len(x_of_mead) / len(x)
            print('{} <{} 节点的概率分布：{}'.format(labela, mead, yprob))
            hv0 = entropy(yprob)
            print('{} <{} 节点的信息熵：{}'.format(labela, mead, hv0))
            children.append(x_of_mead)
            x_of_mead = x[x[labela] >= mead]  # 按中位数划分数据 大于等于中位数
            yvalues = set(item for item in x_of_mead[y_label])
            ygroupsize = x_of_mead.groupby(y_label).size()
            total = ygroupsize.sum()
            yprob = [ygroupsize[c] / total for c in yvalues]
            p1 = len(x_of_mead) / len(x)
            print('{} >={} 节点的概率分布：{}'.format(labela, mead, yprob))
            hv1 = entropy(yprob)
            print('{} >={} 节点的信息熵：{}'.format(labela, mead, hv1))
            children.append(x_of_mead)
            gain = hd - (p0 * hv0 + p1 * hv1)
            print('{}划分的信息增益量为{}'.format(labela, gain))
        if g0 <= gain:
            g0 = gain
            node['choosen_label'] = labela
            node['children'] = children
            node['count'] = [c.shape[0] for c in children]
            node['index'] = index
    atts.append(node['choosen_label'])
    # print('本次划分数据集的节点为{} '.format(node))
    tree.append(node)
    print('本次得到的节点是：{}'.format(tree))
    for c in node['children']:
        print(c)
        print(atts)
        split_data(c,atts,y_label,tree,index+1)



if __name__ == '__main__':
    #    data = pd.DataFrame(['是','否','否','是','是','否','否','否','否','是'])
    data = pd.DataFrame([
        ['是', '单身', 125, '否'],
        ['否', '已婚', 100, '否'],
        ['否', '单身', 100, '否'],
        ['是', '已婚', 110, '否'],
        ['是', '离婚', 60, '否'],
        ['否', '离婚', 95, '是'],
        ['否', '单身', 85, '是'],
        ['否', '已婚', 75, '否'],
        ['否', '单身', 90, '是'],
        ['是', '离婚', 220, '否'],
    ], columns=['拥有房产', '婚姻状况', '年收入', '无法偿还'])
    tree = []
    split_data(data, ['无法偿还'], '无法偿还', tree)
