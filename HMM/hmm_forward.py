# _*_ codig utf8 _*_

import numpy as np


# HMM前向算法
#
def hmm_forward(pi, A, B, s):
    '''
    HMM概率计算：前向概率
    :param pi:初始状态概率
    :param A: 状态转移矩阵
    :param B:状态到观测值转移矩阵
    :param s:观测状态序列
    :return: alpha:各个时刻alpha值的列表,final:最终概率
    '''
    tmp_alpha = pi * B[:, s[0]]
    alpha = []
    alpha.append(tmp_alpha)
    for index in np.arange(1, len(s)):
        tmp_alpha = np.dot(tmp_alpha, A) * B[:, s[index]]
        # print(alpha)
        alpha.append(tmp_alpha)
    final = np.sum(tmp_alpha)
    return np.array(alpha),final



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
    alpha,final = hmm_forward(pi, A, B, s)
    print(alpha)
    print('P(Q;λ)={}'.format(final))