import itertools
import random
import numpy as np
from keras import Input, Model
from keras.src.callbacks import EarlyStopping
from keras.src.initializers import RandomNormal
from keras.src.layers import Dense
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve,auc
from scipy import interp
import warnings
import tensorflow as tf
from CMD_DNMF import cmd_dnmf
from NMF import nmf
from HP import HP
warnings.filterwarnings("ignore")
def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]
    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN
    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])
    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])
    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return [auc[0, 0], aupr[0, 0],f1_score, accuracy, recall, specificity, precision]
def BuildModel(train_x, train_y, epochs):
    l = len(train_x[1])
    inputs = Input(shape=(l,))
    x = Dense(128, activation='relu', kernel_initializer=RandomNormal(seed=random_seed))(inputs)
    x = Dense(64, activation='relu', kernel_initializer=RandomNormal(seed=random_seed))(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='nadam',#adam,SGD
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, restore_best_weights=True)
    model.fit(train_x, train_y, epochs=epochs, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    return model
def get_all_samples(conjunction):
    pos = []
    neg = []
    for index in range(conjunction.shape[0]):
        for col in range(conjunction.shape[1]):
            if conjunction[index, col] == 1:
                pos.append([index, col, 1])
            else:
                neg.append([index, col, 0])
    pos_len = len(pos)
    random.seed(random_seed)
    new_neg = random.sample(neg, pos_len)
    samples = pos + new_neg
    random.seed(random_seed)
    samples = random.sample(samples, len(samples))
    samples = np.array(samples)
    return samples
def generate_f1(train_samples,average_circ_features,average_dis_features,NMF_cfeature,NMF_dfeature):
    hp_nmf_num=NMF_cfeature.shape[1]
    average_num=average_circ_features.shape[1]#低阶图编码的特征数量
    train_n = train_samples.shape[0]
    train_feature = np.zeros([train_n,2 * hp_nmf_num+2*average_num])
    train_label = np.zeros([train_n])
    for i in range(train_n):
        train_feature[i, 0:hp_nmf_num] = NMF_cfeature[train_samples[i, 0], :]
        train_feature[i,hp_nmf_num:hp_nmf_num+average_num] =average_dis_features[train_samples[i, 1], :]
        train_feature[i, hp_nmf_num+average_num:hp_nmf_num+2*average_num] = average_circ_features[train_samples[i, 0], :]
        train_feature[i, hp_nmf_num+2*average_num:2*hp_nmf_num+2*average_num] = NMF_dfeature[train_samples[i, 1], :]
        train_label[i] = train_samples[i, 2]
    return train_feature, train_label
# parameter
cun=[]
m_threshold = [0.5]
epochs = [200]
n_splits = 5
for random_seed in [0]:
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    for alfa in [0.001]:
        for beta in [0.5]:
            for HP_NMF_Features in [19]:  
                for lam1 in [0.01]:  
                    for lam2 in [0.1]:
                        fold = 0
                        result = np.zeros((1, 7), float)
                        association = np.loadtxt("../data/Dataset1/circRNA_disease_asso.txt", dtype=float, delimiter="\t")
                        samples = get_all_samples(association)
                        c_hyper_features = np.loadtxt("../data/Dataset1/hyper_circRNA.txt", dtype=float, delimiter="\t")
                        d_hyper_features = np.loadtxt("../data/Dataset1/hyper_disease.txt", dtype=float, delimiter="\t")
                        average_circ_features = c_hyper_features  # (c_multi_features+c_hyper_features)/2
                        average_dis_features = d_hyper_features  # (d_multi_features+d_hyper_features)/2
                        hp_association = HP(association, random_seed)
                        hp_association[hp_association < 0.5] = 0
                        DNMF_model = cmd_dnmf(association, hp_association, alfa,beta)  # np.dot(np.dot(association, association.T), association)
                        DNMF_model.pre_training()
                        CMD_asso = DNMF_model.training()
                        NMF_cfeature, NMF_dfeature = nmf(CMD_asso, HP_NMF_Features, lam1, lam2)
                        for s in itertools.product(m_threshold, epochs):
                            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
                            tprs = []
                            aucs = []
                            mean_fpr = np.linspace(0, 1, 100)
                            for train_index, val_index in kf.split(samples):
                                fold += 1
                                print(f"***************第：{fold} 折********************")
                                train_samples = samples[train_index, :]
                                val_samples = samples[val_index, :]
                                new_association = association.copy()
                                for i in val_samples:
                                    new_association[i[0], i[1]] = 0
                                train_feature, train_label = generate_f1(train_samples, average_circ_features,
                                                                         average_dis_features, NMF_cfeature, NMF_dfeature)
                                val_feature, val_label = generate_f1(val_samples, average_circ_features, average_dis_features,
                                                                     NMF_cfeature, NMF_dfeature)
                                model = BuildModel(train_feature, train_label, s[1])
                                loss, accuracy = model.evaluate(val_feature, val_label)
                                # print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")
                                # Calculate metrics
                                y_score = model.predict(val_feature)[:, 0]
                                fpr, tpr, thresholds = roc_curve(val_label, y_score)
                                tprs.append(interp(mean_fpr, fpr, tpr))
                                tprs[-1][0] = 0.0
                                roc_auc = auc(fpr, tpr)
                                aucs.append(roc_auc)
                                result += get_metrics(val_label, y_score)
                                final = result / n_splits
                            print("=======================多指标输出结果！！！===========================")
                            print(
                                f"**随机种子数为：{random_seed}，alfa参数{alfa}，beta参数{beta}，HP_NMF参数{HP_NMF_Features}，lam1参数{lam1}，lam2参数{lam2}**: auc,aupr,f1_score, accuracy, recall, specificity, precision",
                                final)
                            cun.append(final)
                        mean_tpr = np.mean(tprs, axis=0)
                        mean_tpr[-1] = 1.0
print(cun)