# _*_ codig utf8 _*_
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

'''
步骤：
1. 遍历样本属性，找到信息增益最大的属性。信息增益G = HD - HD_A.
    计算原始目标属性的信息熵HD。
    对属性X，如果是离散的，取值为x1,x2,...,xn,分别计算X=xi的信息熵HD_A,
    以此得到信息增益 G= HD-HD_A
2. 以此属性作为划分点，划分样本，作为子节点
3. 对各个子节点重复1,2，子节点中中样本类别相同
'''


class customdt:
    def __init__(self):
        self.tree = None
        pass

    def __fit__(self, x, y, tree,index,usedcolumns,pnode,split_value=0):
        '''
        构建决策树
        :param x:
        :param y:
        :return:
        '''
        hd = self.Entropy(y)
        if len(x) <2 or hd ==0:
            #找到叶子节点后，生成叶子节点
            print('找到叶子节点。')
            node = {}
            node['children'] = (split_value,x)
            node['split_value'] = split_value #节点划分的值
            node['label'] = ''
            node['gain'] = 0
            node['index'] = index
            node['class'] = y[0] #当前叶子节点所属的类别
            node['pnode'] = pnode #父节点
            tree.append(node)
            return tree,index
        print('原始殤：{},len(x)={}'.format(hd,len(x)))
        m, n = np.shape(x)
        # gainresults = []
        node = {}
        gain = 0
        tc=[]
        for c in x.columns:
            if c in usedcolumns:
                continue
            xi = x[c]
            # print(is_numeric_dtype(xi.dtype))
            gaintotal = 0
            children = []
            targetchildres = []
            threshold = 0
            if is_numeric_dtype(xi):
                # 如果是连续值，使用中位数作为划分
                splitshreshold = np.median(xi)
                yvleft = y[xi < splitshreshold]
                children.append((0,x[xi<splitshreshold]))
                targetchildres.append(yvleft)
                hdv_left = self.Entropy(yvleft)
                # info_gain = hd-hdv
                p0 = len(yvleft) / len(y)
                yvright = y[xi >= splitshreshold]
                hdv_right = self.Entropy(yvright)
                # info_gain = hd - hdv
                p1 = len(yvright) / len(y)
                children.append((1,x[xi >= splitshreshold]))
                targetchildres.append(yvright)
                gaintotal = hd - (p0 * hdv_left + p1 * hdv_right)
                # gainresults.append([c, 'threshold %s' % splitshreshold, gaintotal])
                threshold = splitshreshold
            else:
                uniquevalues = set(xi)
                hdv_sum = 0
                for v in uniquevalues:
                    yv = y[xi == v]
                    hdv = self.Entropy(yv)
                    hdv_sum += len(yv)/len(y) * hdv
                    children.append((v,x[xi==v]))
                    targetchildres.append(yv)

                gaintotal = hd-hdv_sum
                # gainresults.append([c, v, gaintotal])
            if gain < gaintotal:
                #记录产生最大信息增益的节点划分
                gain = gaintotal
                node['children'] = children
                tc=targetchildres
                node['label'] = c
                node['gain'] = gaintotal
                node['index'] = index
                node['threshold'] = threshold
        node['pnode'] = pnode
        node['split_value'] = split_value
        tree.append(node)
        usedcolumns.append(node['label'])
        # s = np.array(gainresults)[:, 2]
        # max = np.argmax(s, axis=0)
        # print(targetchildres)
        # print('\n')
        # children = node['children']
        for i in np.arange(len(children)):
            index = index + 1
            _,index = self.__fit__(children[i][1],tc[i],tree,index,usedcolumns,node['index'],children[i][0])
        return tree,index

    def fit(self, x, y):
        tree,_ =self.__fit__(x,y,[],0,[],-1)
        self.tree = tree
        return tree

    def Entropy(self, y):
        '''
        计算信息熵
        :param y:目标属性
        :return:
        '''
        uniqueValues = set(y)
        totalitems = len(y)
        df_y = pd.DataFrame(y, columns=['class'])
        grouped_df_y = df_y.groupby('class')
        count = grouped_df_y.size()
        prob = [count[v] / totalitems for v in uniqueValues]
        entry = -1 * np.sum([p * np.log2(p) for p in prob])
        return entry

    def __FilterClass(self, x, y):
        pass

    def predict_single(self,x):
        pindex = 0
        node = self.tree[pindex]
        pnode = None
        while node is not None:
            pnode = node
            pindex = node['index']
            label = node['label']
            if node.get('class','') != '' or node['label']=='':
                break
            print(x)
            x_split=x[label]
            if is_numeric_dtype(x_split):
                threshold =  node.get('threshold',0)
                split_value = 0 if x_split <threshold else 1
            else:
                split_value = x_split
            node = [n for n in self.tree if n['pnode']==pindex and n['split_value']==split_value]
            if len(node)>0:
                node = node[0]
            else:
                node = None
        if pnode != None:
            print('所属叶子结点为：{}'.format(pnode['index']))
            return pnode.get('class',0)

    def predict(self,x):
        m = x.shape[0]
        result = []
        for i in np.arange(m):
            result.append(self.predict_single(x.ix[i]))
        return result

if __name__ == '__main__':
    X = pd.DataFrame([
        ['是', '单身', 125],
        ['否', '已婚', 100],
        ['否', '单身', 100],
        ['是', '已婚', 110],
        ['是', '离婚', 60],
        ['否', '离婚', 95],
        ['否', '单身', 85],
        ['否', '已婚', 75],
        ['否', '单身', 90],
        ['是', '离婚', 220],
    ], columns=['拥有房产', '婚姻状况', '年收入'])
    Y = np.array([
        '否',
        '否',
        '否',
        '否',
        '否',
        '是',
        '是',
        '否',
        '是',
        '否',
    ])

    cdt = customdt()

    r = cdt.fit(X, Y)
    print('决策树构建完毕：\n{}'.format([node for node in r ]))
    # print('决策树构建完毕：\n{}'.format([node for node in r if node['index'] == 0]))
    t = pd.DataFrame([
        ['否', '单身', 100]
    ], columns=['拥有房产', '婚姻状况', '年收入'])
    cls = cdt.predict(X)
    print('预测结果为：{}'.format(cls))