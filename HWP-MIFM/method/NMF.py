#######################使用L1范数##############################
# import numpy as np
# def nmf(M,D,lam):
#     def get_low_feature(k,lam, th, A):
#         # #随机初始化
#         # m, n = A.shape
#         # rng = np.random.RandomState(1)
#         # arr1 = rng.randint(0,100,size=(m,k))
#         # arr2 = rng.randint(0,100,size=(k,n))
#         # U = arr1 / 100
#         # V = arr2 / 100
#         #SVD分解
#         U_init, S, V_init = np.linalg.svd(A)
#         # 取前rank个奇异值和向量
#         U_r = U_init[:, :k]
#         S_r = np.diag(S[:k])
#         VT_r = V_init[:k, :]
#         # 初始化U和V矩阵
#         U = np.dot(U_r, S_r)
#         V = VT_r
#         # 确保U和V矩阵非负
#         U[U < 0] = 0
#         V[V < 0] = 0
#         obj_value = objective_function(A, A, U, V, lam)
#         i = 0
#         while i < 10000:
#             i =i + 1
#             U = updating_U(A, A, U, V, lam)
#             V = updating_V(A, A, U, V, lam)
#             obj_value1 = objective_function(A, A, U, V, lam)
#             relative_diff = abs(obj_value1 - obj_value) / (abs(obj_value) + 1e-10)
#             if relative_diff<th:#终止条件
#                 print("迭代次数：",i)
#                 break
#             obj_value=obj_value1
#         return U, V.transpose()
#     def objective_function(W, A, U, V, lam):
#         m, n = A.shape
#         sum_obj = 0
#         for i in range(m):
#             for j in range(n):
#                 sum_obj = sum_obj + W[i,j]*(A[i,j] - U[i,:].dot(V[:,j]))+ lam*(np.linalg.norm(U[i, :], ord=1,keepdims= False) + np.linalg.norm(V[:, j], ord =1, keepdims = False))
#         return  sum_obj
#     def updating_U (W, A, U, V, lam):
#         m, n = U.shape
#         upper = (W*A).dot(V.T)
#         down = (W*(U.dot(V))).dot((V.T)) + (lam/2) *(np.ones([m, n]))
#         U_new = U
#         for i in range(m):
#             for j in range(n):
#                 U_new[i,j] = U[i, j]*(upper[i,j]/down[i, j])
#         return U_new
#     def updating_V (W, A, U, V, lam):
#             m,n = V.shape
#             upper = (U.T).dot(W*A)
#             down = (U.T).dot(W*(U.dot(V)))+(lam/2)*(np.ones([m,n]))
#             V_new = V
#             for i in range(m):
#                 for j in range(n):
#                     V_new[i,j] = V[i, j]*(upper[i,j]/down[i,j])
#             return V_new
#     ##################################程序起点！#####################################################
#     NMF_cfeature, NMF_dfeature = get_low_feature(D, lam, pow(10, -4), M)
#     #NMF有参数，两个参数lam和D，在这里lam取0.01，D取得是30
#     np.savetxt('../data/Dataset2/NMF_cfeature.txt',NMF_cfeature,fmt='%6f',delimiter='\t')
#     np.savetxt('../data/Dataset2/NMF_dfeature.txt',NMF_dfeature,fmt='%6f',delimiter='\t')
#     return NMF_cfeature,NMF_dfeature
# #SVD分解
# # arr1=np.random.randint(0,100,size=(m,k))
# # U = arr1/100
# # arr2=np.random.randint(0,100,size=(k,n))
# # V = arr2/100
# # SVD初始化
# # U, sigma, V = np.linalg.svd(A, full_matrices=False)
# # # 只选择前k个特征
# # U = U[:, :k]
# # sigma = sigma[:k]
# # V = V[:k, :]
# # # 归一化U和V
# # U = U * sigma  # 将奇异值应用于左特征矩阵
#######################使用L2范数##############################
import numpy as np
def nmf(A, D, lam1, lam2):  # 修改1：分离参数
    def get_low_feature(k, lam1, lam2, th, A,epsilon):  # 修改2：增加参数
        # 保持原初始化逻辑
        U1, S1, VT1 = np.linalg.svd(A, full_matrices=False)
        # 2. 截断到前k个奇异值
        m, n = A.shape
        U_trunc = U1[:, :k]  # (m, k)
        S_trunc = S1[:k]  # (k,)
        VT_trunc = VT1[:k, :]  # (k, n)
        # 3. 构造初始化矩阵（添加数值稳定性）
        S_sqrt = np.sqrt(np.maximum(S_trunc, epsilon))  # 防止sqrt(0)
        U_scaled = U_trunc @ np.diag(S_sqrt)  # (m, k)
        V_scaled = (VT_trunc.T @ np.diag(S_sqrt))  # (n, k)
        # 方式1：简单截断
        U = np.maximum(U_scaled, 0)
        V = np.maximum(V_scaled.T, 0)
        # 修改3：使用新参数计算目标
        obj_value = objective_function(A, A, U, V, lam1, lam2)
        i = 0
        while i < 10000:
            i += 1
            # 修改4：分别传递正则化参数
            U = updating_U(A, A, U, V, lam1)
            V = updating_V(A, A, U, V, lam2)
            # 修改5：更新目标计算
            obj_value1 = objective_function(A, A, U, V, lam1, lam2)
            relative_diff = abs(obj_value1 - obj_value) / (abs(obj_value) + 1e-10)
            if relative_diff < th:
                print("迭代次数：", i)
                break
            obj_value = obj_value1
        return U, V.transpose()
    # 修改6：新目标函数
    def objective_function(W, A, U, V, lam1, lam2):
        reconstruction = np.sum(W * (A - U.dot(V)) ** 2)
        reg_term = lam1 * np.sum(U ** 2) + lam2 * np.sum(V ** 2)  # 等效Frobenius范数平方
        return reconstruction + reg_term
    # 修改7：U更新规则（仅λ1）
    def updating_U(W, A, U, V, lam1):
        m, n = U.shape
        upper = (W * A).dot(V.T)
        down = (W * (U.dot(V))).dot(V.T) + lam1 * U  # 正则化项修改
        return U * (upper / np.maximum(down, 1e-8))  # 数值稳定性
    # 修改8：V更新规则（仅λ2）
    def updating_V(W, A, U, V, lam2):
        m, n = V.shape
        upper = U.T.dot(W * A)
        down = U.T.dot(W * (U.dot(V))) + lam2 * V  # 正则化项修改
        return V * (upper / np.maximum(down, 1e-8))  # 数值稳定性
    # 修改9：调用时传入两个参数
    NMF_cfeature, NMF_dfeature = get_low_feature(D,lam1,lam2,1e-4,A,epsilon=1e-8)
    # 以下保持原样
    np.savetxt('../data/Dataset1/NMF_cfeature.txt', NMF_cfeature, fmt='%6f', delimiter='\t')
    np.savetxt('../data/Dataset1/NMF_dfeature.txt', NMF_dfeature, fmt='%6f', delimiter='\t')
    return NMF_cfeature, NMF_dfeature