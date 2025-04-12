import numpy as np
GIP_circRNA=np.loadtxt(r"../data/Dataset1/circRNA_GIP.txt")
cos_circRNA=np.loadtxt(r"../data/Dataset1/cos_circRNA.txt")
func_circRNA=np.loadtxt(r"../data/Dataset1/circRNA_func.txt")
entro_circRNA=np.loadtxt(r"../data/Dataset1/entropy_circRNA.txt")
GIP_disease=np.loadtxt(r"../data/Dataset1/disease_GIP.txt")
cos_disease=np.loadtxt(r"../data/Dataset1/cos_disease.txt")
sem_disease=np.loadtxt(r"../data/Dataset1/disease_semantic.txt")
entro_disease=np.loadtxt(r"../data/Dataset1/entropy_disease.txt")
average_circRNA=(GIP_circRNA+entro_circRNA+func_circRNA+cos_circRNA)/4
average_disease=(GIP_disease+entro_disease+cos_disease+sem_disease)/4
np.savetxt('../data/Dataset1/average_circRNA.txt',average_circRNA, fmt="%6f",delimiter="\t")
np.savetxt('../data/Dataset1/average_disease.txt',average_disease, fmt="%6f",delimiter="\t")
