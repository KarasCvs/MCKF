import matplotlib.pyplot as plt
import json
import numpy as np


with open('./SLAM_data_reader/log_data/score.json', 'r') as score:
    score_dic = json.load(score)
mcekf_score = score_dic['score']
with open('./SLAM_data_reader/log_data/cov.json', 'r') as cov:
    cov_dic = json.load(cov)
mcekf_cov = cov_dic['cov_norm']


font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
plt.figure()
plt.plot(np.linspace(0, len(mcekf_score), len(mcekf_score)), mcekf_score, label="mcekf")
plt.legend(prop=font)
plt.grid(True)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('Count', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.title('Scan-matching Score', fontsize=20)
plt.figure()
plt.plot(np.linspace(0, len(mcekf_cov), len(mcekf_cov)), mcekf_cov, label="mcekf")
plt.legend(prop=font)
plt.grid(True)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('Sampaling Point', fontsize=15)
plt.ylabel('Mean of Covariance', fontsize=15)
plt.title('Mean of Covariance Matrix', fontsize=20)
plt.show()