import re
import json
import numpy as np


m_count = []
score = []
score_dic = {}
with open('./SLAM_data_reader/logs/slam_gmapping-2-stdout.log', 'r') as file:
    lines = file.readlines()
    for line in lines:
        count_match = re.match(r'(m_count)(\s)(\d*)', line)
        if count_match:
            m_count.append(count_match.group(3))
        score_match = re.match(r'(Average Scan Matching Score=)(\d*\.\d*)', line)
        if score_match:
            score.append(score_match.group(2))
    score_dic['mcekf m_count'] = [float(i) for i in m_count]
    score_dic['mcekf score'] = [float(i) for i in score]
    with open('./log_data/score.json', 'w') as f:
        json_str = json.dumps(score_dic)
        f.write(json_str)


covariances = []
cov_norm = []
with open('./SLAM_data_reader/logs/mcekf_localization-1-stdout.log', 'r') as file:
    covariance = ''
    lines = file.readlines()
    for line in lines:
        line = line.replace('\n', '')
        if not len(line):
            if covariance != '':
                covariances.append(covariance)
            covariance = ''
        else:
            covariance += line

for i in range(len(covariances)):
    covariances[i] = covariances[i].replace('[', '')
    covariances[i] = covariances[i].replace(']', '')
    covariances[i] = re.split(r'\s+', covariances[i])
    for j in range(len(covariances[i])):
        if covariances[i][j] != '':
            covariances[i][j] = float(covariances[i][j])
        else:
            covariances[i].pop(j)
    covariances[i] = np.array(covariances[i]).reshape(15, 15)
    cov_norm.append(np.linalg.norm(covariances[i]))

with open('./SLAM_data_reader/log_data/cov.json', 'w') as f:
    json_str = json.dumps({'mcekf cov_norm': cov_norm})
    f.write(json_str)