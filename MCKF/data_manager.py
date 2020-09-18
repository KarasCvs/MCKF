import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np


class Manager():
    def __init__(self):
        self.path = './data/'
        self.time = time.strftime('%Y-%m-%d %H.%M.%S', time.localtime())
        self.file = os.path.join(self.path, self.time+'.json')
        self.temp_file = os.path.join(self.path, 'LastSimulation.json')

    def save_data(self, data):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        data['timestamp'] = self.time
        data = json.dumps(data)
        with open(self.file, 'w') as f:
            f.write(data)
        with open(self.temp_file, 'w') as f:
            f.write(data)

    def view_data(self, data):
        self.data = data
        self.parameters = data['parameters']
        self.states_dimension = data['shapes']['states dimension']
        self.obs_dimension = data['shapes']['obs dimension']
        self.N = int(data['parameters']['time']/data['parameters']['ts'])
        self.time_line = np.array(data['time line'])
        self.states = data['states']
        self.observations = data['observations']
        self.mse1 = data['mse1']
        self.mse = data['mse']

    def read_data(self, filename=None):
        if filename is None:
            filename = 'LastSimulation.json'
        with open(os.path.join(self.path, filename), 'r') as f:
            json_data = f.read()
            data = json.loads(json_data)
        self.view_data(data)
        return data

    def plot_all(self):
        self.plot_states()
        self.plot_mse1()
        self.plot_obs()
        self.plot_mse()

    def plot_states(self):
        plt.figure()
        for i in range(self.states_dimension):
            plt.subplot(100*self.states_dimension+11+i)
            for j in self.states:
                plt.plot(self.time_line, np.array(self.states[j])[i, :].reshape(self.N,), linewidth=1, linestyle="-", label=j)
            plt.grid(True)
            plt.legend(loc='upper left')
            plt.title(f"States {i}")

    def plot_obs(self):
        plt.figure()
        for i in self.observations:
            plt.plot(self.time_line, np.array(self.observations[i]).reshape(self.N,), linewidth=1, linestyle="-", label=i)
            plt.grid(True)
            plt.legend(loc='upper left')
            plt.title("Observation")

    def plot_mse1(self):
        plt.figure()
        for i in range(self.states_dimension):
            plt.subplot(100*self.states_dimension+11+i)
            for j in self.mse1:
                plt.plot(self.time_line, np.array(self.mse1[j])[i, :].reshape(self.N,), linewidth=1, linestyle="-", label=j)
            plt.grid(True)
            plt.legend(loc='upper left')
            plt.title(f"MSE of state{i}")

    def plot_mse(self):
        print(f'Sigma = {self.parameters["sigma"]}')
        for i in self.mse:
            print(f'{i}=\n{self.mse[i]}\n')

    def show(self):
        plt.show()

    def find(self, key, filename=None):
        key = key.lower()
        self.read_data(filename)
        for i in self.data:
            try:
                if type(self.data[i]) is dict:
                    for j in self.data[i]:
                        if j == key:
                            return self.data[i][j]
                else:
                    if i == key:
                        return self.data[i]
            except KeyError:
                print("Can't find key.\n")

    def locate(self, keywords):
        filenames = os.listdir(self.path)
        fitted_list = []
        for filename in filenames:
            for key in keywords:
                value = self.find(key, filename)
                if value is not keywords[key]:
                    break
                fitted_list.append(filename)
        if fitted_list:
            return fitted_list
        else:
            print("Can't find keys.\n")
