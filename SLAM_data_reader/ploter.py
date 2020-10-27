import matplotlib.pyplot as plt
import json
import numpy as np


with open('./data/score.json', 'r') as score:
    score_dic = json.load(score)

with open('./data/cov.json', 'r') as cov:
    cov_dic = json.load(cov)
plt.figure()
plt.plot(np.linspace(0, len(score_dic['score']), len(score_dic['score'])), score_dic['score'])
plt.grid(True)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('Count', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.title('Scan-matching Score', fontsize=20)
plt.figure()
plt.plot(np.linspace(0, len(cov_dic['cov_norm']), len(cov_dic['cov_norm'])), cov_dic['cov_norm'])
plt.grid(True)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('Sampaling Point', fontsize=15)
plt.ylabel('Mean of Covariance', fontsize=15)
plt.title('Mean of Covariance Matrix', fontsize=20)
plt.show()