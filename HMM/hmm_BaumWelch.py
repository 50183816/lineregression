# _*_ codig utf8 _*_
import numpy as np
import hmm_kesai, hmm_gamma, hmm_forward, hmm_backward


def BaumWelch(pi, A, B, Q, maxiter=100, alpha=None, beta=None, gamma=None, kesai=None):
    N = np.shape(A)[1]
    T = np.shape(Q)[0]
    M = np.shape(B)[1]

    # 基于初始pi,A,B,迭代计算相关参数
    for m in np.arange(maxiter):
        alpha, _ = hmm_forward.hmm_forward(pi, A, B, Q)
        beta, _ = hmm_backward.hmm_backward(pi, A, B, Q)
        gamma = hmm_gamma.hmm_gamma(alpha, beta)
        kesai = hmm_kesai.hmm_kesai(alpha, beta, A, B, Q)
        # 迭代
        for i in np.arange(N):
            pi[i] = gamma[0][i]
            # 更新A
            for j in np.arange(N):
                sum_gamma = 0
                sum_kesai = 0
                for t in np.arange(T - 1):
                    sum_gamma += gamma[t][i]
                    sum_kesai += kesai[t][i][j]
                A[i][j] = sum_kesai / sum_gamma #if sum_gamma != 0 else 0
            # 更新B
            for j in np.arange(M):
                sum_gamma_ij = 0
                sum_gamma = 0
                for t in np.arange(T):
                    sum_gamma += gamma[t][i]
                    if Q[t] == j:
                        sum_gamma_ij += gamma[t][i]
                B[i][j] = sum_gamma_ij / sum_gamma #if sum_gamma != 0 else 0

    return pi, A, B


if __name__ == '__main__':
    pi = np.array([0.2, 0.2, 0.6])
    A = np.array([
        # [0.5, 0.4, 0.1],
        # [0.2, 0.2, 0.6],
        # [0.2, 0.5, 0.3]
        [0.1, 0.1, 0.9],
        [0.2, 0.2, 0.6],
        [0.3, 0.4, 0.3]
    ])

    B = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5]
    ])
    Q = [0, 1, 0, 0, 1]
    pi,A,B = BaumWelch(A=A, B=B, Q=Q, pi=pi,maxiter=5)

    print('π={}'.format(pi))
    print('A={}'.format(A))
    print('B={}'.format(B))
