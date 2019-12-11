# _*_ codig utf8 _*_
import numpy as np

class ReleaseRing():

    def __init__(self,n):
        self.ring = np.ones(n)

    def Release(self,n,cycle):
        '''
        解九连环
        :param ring: 九连环
        :param n: 解第几环
        :return:
        '''
        cycle += 1
        if n == 1:
            print('解下1环')
            self.ring[n-1]=0
            print(self.ring)
            return cycle
        if n == 2:
            print('解下1，2环')
            cycle += 1
            self.ring[n - 1] = 0
            self.ring[n - 2] = 0
            print(self.ring)
            return cycle
        if n==3:
            if self.ring[1]==0:
                cycle = self.Puton(2,cycle)
        cycle = self.Release(n - 2,cycle)
        print('解下%d环' % n)
        self.ring[n - 1] = 0
        print(self.ring)
        # 要接下n-1环需要先把n-2环套上
        cycle = self.Puton(n - 2,cycle)
        cycle = self.Release(n - 1,cycle)
        return cycle


    def Puton(self,n,cycle):
        cycle += 1
        if n == 2:
            print('套上1,2环')
            self.ring[n - 1] = 1
            self.ring[n - 2] = 1
            cycle += 1
            print(self.ring)
            return cycle
        if n == 1:
            print('套上1环')
            self.ring[n - 1] = 1
            print(self.ring)
            return cycle
        cycle = self.Puton(n - 1,cycle)
        cycle = self.Release(n-2,cycle)
        print('套上%d环' % n)
        self.ring[n - 1] = 1
        print(self.ring)
        cycle = self.Puton(n - 2, cycle)
        return cycle


if __name__ == '__main__':
    n =9
    cycle =0
    steps=[]
    for i in np.arange(9,10):
        n=i
        cycle=0
        rr  = ReleaseRing(n)
        cycle = rr.Release(n,cycle)
        print('共需要用%d步'%cycle)
        steps.append(cycle)


    print(steps)
#從1到19连环的拆解步骤：[1, 1, 4, 7, 16, 31, 64, 127, 256, 511, 1024, 2047, 4096, 8191, 16384, 32767, 65536, 131071, 262144]
#規律： Steps = 2^(n-1) - (n-1)%2  这种统计方法时把1,2同上同下视为两步