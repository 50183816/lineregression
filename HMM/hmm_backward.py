# _*_ codig utf8 _*_

import numpy as np


def hmm_backward(pi, A, B, Q):
    '''
    HMM概率计算：后向向概率
    :param pi:初始状态概率
    :param A: 状态转移矩阵
    :param B:状态到观测值转移矩阵
    :param Q:观测状态序列
    :return: beta:各个时刻beta值的列表,final:最终概率
    '''
    N = A.shape[0]
    T = len(Q)
    M = B.shape[1]
    beta = np.zeros((T, N))
    beta[T - 1, :] = 1
    for t in np.arange(T - 2, -1, -1):
        for i in np.arange(N):
            for j in np.arange(N):
                beta[t][i] += A[i][j] * B[j][Q[t + 1]] * beta[t + 1][j]

    final = np.sum(pi*B[:,Q[0]]*beta[0,:])
    # for i in np.arange(N):
    #     final += pi[i] * B[i][Q[0]] * beta[0][i]
    return beta, final


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
    Q = [0, 1, 0, 0, 1]
    beta, final = hmm_backward(pi, A, B, Q)
    print(beta)
    print('P(Q;λ)={}'.format(final))
