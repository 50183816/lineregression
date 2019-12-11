# _*_ codig utf8 _*_
import numpy as np
import hmm_backward, hmm_forward


def hmm_kesai(alpha, beta, A, B, Q):
    '''
    计算观测序列中t时刻和t+1时刻状态分别为si 和sj的概率
    :param alpha: 前向概率矩阵
    :param beta: 后续概率矩阵
    :param A: 状态转移矩阵
    :param B:状态到观测值转移矩阵
    :param Q:观测状态序列
    :return: ξ矩阵
    '''
    N = alpha.shape[1]
    T = alpha.shape[0]
    kesai = np.zeros((T-1, N, N))

    # 求α* β的和
    for t in np.arange(T-1):
        sum = 0
        for i in np.arange(N):
            for j in np.arange(N):
                sum += alpha[t][i] * A[i][j] * B[j][Q[t + 1]] * beta[t + 1][j]
        for i in np.arange(N):
            for j in np.arange(N):
                kesai[t][i][j] = (alpha[t][i] * A[i][j] * B[j][Q[t + 1]] * beta[t + 1][j]) / sum
    return kesai


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
    kesai = hmm_kesai(alpha, beta,A,B,Q)
    print('ξ={}'.format(kesai))
