# _*_ codig utf8 _*_

import numpy as np


# HMM前向算法
#
def hmm_forward(pi, A, B, s):
    '''

    :param pi:
    :param A:
    :param B:
    :param s:
    :return:
    '''
    alpha = pi * B[:, s[0]]
    print(alpha)
    for index in np.arange(1, len(s)):
        alpha = np.dot(alpha, A) * B[:, s[index]]
        print(alpha)

    final = np.sum(alpha)
    return final


# viterbi算法预测概率最大的路径
def hmm_viterbi(pi, A, B, s):
    """
    viterbi 算法计算隐状态
    :param pi:初始状态概率
    :param A: 状态转移矩阵
    :param B: 状态至观测值转移矩阵
    :param s: 观测序列
    :return:
    """
    delta = pi * B[:, s[0]]  # 初始化delta = Pi(i) * B(iq1)
    # print(delta)
    path = [np.argmax(delta) + 1]  # 保存每一步的最优路径
    for index in np.arange(1, len(s)):
        delta = delta * A.T  # 计算从之前状态转到当前状态的概率，数乘
        # print('delta*A.T:\n{}'.format(delta))
        path.append(np.argmax(delta, axis=1))  # 记录当前步每一个状态下的最优路径，也就是从之前状态转到当前状态的最高概率
        delta = np.array([np.max(t) for t in delta[:]] * B[:, s[index]])
        # 计算当前状态下产生预期观察结果的概率。状态的概率取从上一状态到达当前状态的最大概率
    status_index_bestProb = np.argmax(delta, axis=0)
    finalist = [status_index_bestProb + 1]
    index = len(s) - 1
    # 从结果反推，找出路径。当前最优是从状态S3得到结果Q，而状态S3是从上一状态S2得到最优结果，S2又从它上一状态Sx得到最优结果
    # 依次类推，反推找出最优路径
    while index > 0:
        # print(path[index])
        next = path[index][status_index_bestProb]
        finalist.insert(0, next + 1)
        index -= 1
        status_index_bestProb = next
    return finalist, np.max(delta)


if __name__ == '__main__':
    pi = np.array([0.2, 0.5, 0.3])
    A = np.array([
        [0.5, 0.4, 0.1],
        [0.2, 0.2, 0.6],
        [0.2, 0.5, 0.3]
    ])

    B = np.array([
        [0.4, 0.6],
        [0.8, 0.2],
        [0.5, 0.5]
    ])
    s = [0, 1, 0, 0, 1]
    final, p = hmm_viterbi(pi, A, B, s)
    print('最大概率为{}，最优路径为{}'.format(p, final))
# finalP = 0;
# p = np.dot(pi,B)
# print(pi)
# print(p)
# finalP = p[0]
#
# #白黑白白黑
# pi = np.dot(pi,A)
# p = np.dot(pi,B)
# print(pi)
# print(p)
# finalP *= p[1]
#
# pi = np.dot(pi,A)
# p = np.dot(pi,B)
# print(pi)
# print(p)
# finalP *= p[0]
#
# pi = np.dot(pi,A)
# p = np.dot(pi,B)
# print(pi)
# print(p)
# finalP *= p[0]
#
# pi = np.dot(pi,A)
# p = np.dot(pi,B)
# print(pi)
# print(p)
# finalP *= p[1]
#
# print(finalP)
