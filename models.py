'''
Created on Dec 8, 2015

@author: donghyun
'''

import os
import time

from util import eval_RMSE
import math
import numpy as np
from text_analysis.models import CNN_module


def ConvMF(res_dir, train_user, train_item, valid_user, test_user,
           R, CNN_X, vocab_size, init_W=None, give_item_weight=True,
           max_iter=50, lambda_u=1, lambda_v=100, dimension=50,
           dropout_rate=0.2, emb_dim=200, max_len=300, num_kernel_per_ws=100):
    '''
    构造并训练卷积矩阵分解模型
    :param res_dir:结果文件路径
    :param train_user:训练集用户稀疏评分向量（libSVM format）
    :param train_item:训练集物品稀疏评分向量（libSVM format）
    :param valid_user:测试集用户稀疏评分向量（libSVM format）
    :param test_user:测试集物品稀疏评分向量（libSVM format）
    :param R:原始评分数据,format: user id::item id::rating
    :param CNN_X:物品描述词序列
    :param vocab_size:词表大小
    :param init_W:如果为None则动态训练词向量权重
    :param give_item_weight:如果为True则使用静态词向量，否则动态训练词向量
    :param max_iter:最大迭代次数
    :param lambda_u:用户端正则惩罚项系数
    :param lambda_v:用户端正则惩罚项系数
    :param dimension:隐变量维度
    :param dropout_rate:丢弃率
    :param emb_dim:词嵌入维度
    :param max_len:物品文本描述序列最大长度
    :param num_kernel_per_ws:CNN的卷积核个数
    :return:None
    '''
    # explicit setting
    a = 1
    b = 0

    num_user = R.shape[0]
    num_item = R.shape[1]
    PREV_LOSS = 1e-50
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    f1 = open(res_dir + '/state.log', 'w')

    Train_R_I = train_user[1] #user rating_list
    Train_R_J = train_item[1] #item rating_list
    Test_R = test_user[1]
    Valid_R = valid_user[1]

    if give_item_weight is True:
        item_weight = np.array([math.sqrt(len(i))
                                for i in Train_R_J], dtype=float)
        item_weight = (float(num_item) / item_weight.sum()) * item_weight
    else:
        item_weight = np.ones(num_item, dtype=float)

    pre_val_eval = 1e10
    ## init CNN model
    cnn_module = CNN_module(dimension, vocab_size, dropout_rate,
                            emb_dim, max_len, num_kernel_per_ws, init_W)
    theta = cnn_module.get_projection_layer(CNN_X)
    np.random.seed(133)
    # user-latent matrix
    U = np.random.uniform(size=(num_user, dimension))
    # item-latent matrix
    V = theta

    endure_count = 5
    count = 0
    for iteration in range(max_iter):
        loss = 0
        tic = time.time()
        print("%d iteration\t(patience: %d)" % (iteration, count))

        ##get user-latent matirx loss
        VV = b * (V.T.dot(V)) + lambda_u * np.eye(dimension) #theta V
        sub_loss = np.zeros(num_user)

        for i in range(num_user):
            idx_item = train_user[0][i]
            V_i = V[idx_item]
            R_i = Train_R_I[i]
            A = VV + (a - b) * (V_i.T.dot(V_i))
            B = (a * V_i * (np.tile(R_i, (dimension, 1)).T)).sum(0)

            U[i] = np.linalg.solve(A, B)

            sub_loss[i] = -0.5 * lambda_u * np.dot(U[i], U[i])

        loss = loss + np.sum(sub_loss)
        ##get item-latent matirx loss
        sub_loss = np.zeros(num_item)
        UU = b * (U.T.dot(U))
        for j in range(num_item):
            idx_user = train_item[0][j]
            U_j = U[idx_user]
            R_j = Train_R_J[j]

            tmp_A = UU + (a - b) * (U_j.T.dot(U_j))
            A = tmp_A + lambda_v * item_weight[j] * np.eye(dimension)
            B = (a * U_j * (np.tile(R_j, (dimension, 1)).T)
                 ).sum(0) + lambda_v * item_weight[j] * theta[j]
            V[j] = np.linalg.solve(A, B)

            sub_loss[j] = -0.5 * np.square(R_j * a).sum()
            sub_loss[j] = sub_loss[j] + a * np.sum((U_j.dot(V[j])) * R_j)
            sub_loss[j] = sub_loss[j] - 0.5 * np.dot(V[j].dot(tmp_A), V[j])

        loss = loss + np.sum(sub_loss)
        seed = np.random.randint(100000)

        # get cnn loss
        history = cnn_module.train(CNN_X, V, item_weight, seed)
        theta = cnn_module.get_projection_layer(CNN_X)
        cnn_loss = history.history['loss'][-1]

        loss = loss - 0.5 * lambda_v * cnn_loss * num_item

        # get rmse eval
        tr_eval = eval_RMSE(Train_R_I, U, V, train_user[0])
        val_eval = eval_RMSE(Valid_R, U, V, valid_user[0])
        te_eval = eval_RMSE(Test_R, U, V, test_user[0])

        toc = time.time()
        elapsed = toc - tic

        converge = abs((loss - PREV_LOSS) / PREV_LOSS)
        #save u,v,w weight
        if (val_eval < pre_val_eval):
            cnn_module.save_model(res_dir + '/CNN_weights.hdf5')
            np.savetxt(res_dir + '/U.dat', U)
            np.savetxt(res_dir + '/V.dat', V)
            np.savetxt(res_dir + '/theta.dat', theta)
        else:
            count = count + 1

        pre_val_eval = val_eval
        print("Loss: %.5f Elpased: %.4fs Converge: %.6f Tr: %.5f Val: %.5f Te: %.5f" % (
            loss, elapsed, converge, tr_eval, val_eval, te_eval))
        f1.write("Loss: %.5f Elpased: %.4fs Converge: %.6f Tr: %.5f Val: %.5f Te: %.5f\n" % (
            loss, elapsed, converge, tr_eval, val_eval, te_eval))
        if (count == endure_count):
            break
        PREV_LOSS = loss

    f1.close()
