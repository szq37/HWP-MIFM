import numpy as np
def HP(Z,ran_seed):
    def hyper_connection(Z,k,rand_w):
        C= np.linalg.matrix_power(Z.T@Z,k - 1)
        Z_1= Z @ C
        Z_1[Z_1 > 1] = 1
        W = (0.01 ** (k - 2)) * rand_w
        Hop = (Z_1 - Z) * W
        return Hop
    Z_result=np.zeros(Z.shape)
    np.random.seed(ran_seed)
    rand_w = np.random.rand(*Z.shape)
    zero_count_1= np.count_nonzero(Z_result == 0)
    for i in range(2,100):
        Z_result+=hyper_connection(Z,i,rand_w)
        zero_count_2 = np.count_nonzero(Z_result == 0)
        if zero_count_1==zero_count_2:
            print("高阶扰动保持不变的阶数：",i-1)
            Z_result=Z_result-hyper_connection(Z,i,rand_w)
            break
        zero_count_1 = zero_count_2
    Z_result[Z_result > 1] = 1
    ZZ=Z_result+Z
    np.savetxt('../data/Dataset1/asso_hp.txt', ZZ, fmt='%6f', delimiter='\t')
    return ZZ