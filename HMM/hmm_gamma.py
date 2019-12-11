# _*_ codig utf8 _*_
import numpy as np
import hmm_backward, hmm_forward


def hmm_gamma(alpha, beta):
    '''
    计算观测序列中各个时刻各个状态的的概率
    :param alpha: 前向概率矩阵
    :param beta: 后续概率矩阵
    :return: γ矩阵
    '''
    N = np.shape(alpha)[1]
    T = np.shape(alpha)[0]
    gamma = np.zeros((T, N))

    # 求α* β的和
    for t in np.arange(T):
        sum = 0
        for j in np.arange(N):
            sum += alpha[t][j] * beta[t][j]
        for i in np.arange(N):
            gamma[t][i] = alpha[t][i] * beta[t][i] / sum

    return gamma


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
    beta, _ = hmm_backward.hmm_backward(pi, A, B, Q)
    alpha, _ = hmm_forward.hmm_forward(pi, A, B, Q)
    gamma = hmm_gamma(alpha, beta)
    print('γ={}'.format(gamma))
