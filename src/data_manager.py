import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np


class Manager():
    def __init__(self):
        self.path = './data/'
        self.temp_file = os.path.join(self.path, 'LastSimulation.json')

    def noise_write(self, obs_dimension, length, r, additional_noise):
        noise = np.mat(r * np.random.randn(obs_dimension, length) + additional_noise * np.random.randn(obs_dimension, length))
        noise = {'obs_noise': noise.tolist()}
        noise_data = json.dumps(noise)
        with open('./data/noise.json', 'w') as f:
            f.write(noise_data)

    def noise_read(self):
        with open('./data/noise.json', 'r') as f:
            noise_json = f.read()
            noise = json.loads(noise_json)
        return np.mat(noise['obs_noise'])

    def save_data(self, data, name=None):
        time_ = time.strftime('%Y-%m-%d %H.%M.%S', time.localtime())
        if name is None:
            filename = os.path.join(self.path, time_+'.json')
        else:
            filename = os.path.join(self.path, name+'.json')
        self.view_data(data)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        data['timestamp'] = time_
        data = json.dumps(data)
        with open(filename, 'w') as f:
            f.write(data)
        with open(self.temp_file, 'w') as f:
            f.write(data)
        print(f"Data saved as '{filename}'")

    def view_data(self, data):
        self.data = data
        self.states_dimension = data['states dimension']
        self.obs_dimension = data['obs dimension']
        self.N = int(data['time']/data['ts'])
        self.time_line = np.array(data['time line'])
        self.states = data['states']
        self.observations = data['observations']
        self.mse1 = data['mse1']
        self.mse = data['mse']
        self.states = data['states']
        self.run_time = data['run time']

    def read_data(self, filename=None):
        if filename is None:
            filename = 'LastSimulation'
        with open(os.path.join(self.path, filename+'.json'), 'r') as f:
            json_data = f.read()
            data = json.loads(json_data)
        self.view_data(data)
        return data

    def plot_all(self):
        self.plot_states()
        self.plot_mse1()
        self.plot_obs()
        self.plot_mse()
        print(f"Filters time cost:\n{self.run_time}")

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
        for i in range(self.obs_dimension):
            plt.subplot(100*self.obs_dimension+11+i)
            for j in self.observations:
                plt.plot(self.time_line, np.array(self.observations[j])[i].reshape(self.N,), linewidth=1, linestyle="-", label=j)
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
        for i in self.mse:
            print(f'{i}=\n{self.mse[i]}\n')

    def plot(self, target, filename=None):
        data = np.array(self.find(target, filename))
        for i in range(data.shape[0]):
            plt.plot(self.time_line, data.reshape(data.shape[1]), label=target)
            plt.grid(True)
            plt.legend(loc='upper left')
            plt.title(target)

    def show(self):
        plt.show()

    # find in dict. 这个应该改成递归.
    # def find(self, key, filename=None):
    #     key = key.lower()
    #     for i in self.data:
    #         try:
    #             if i == key:
    #                 return self.data[i]
    #             if type(self.data[i]) is dict:
    #                 for j in self.data[i]:
    #                     if type(self.data[i][j]) is dict:
    #                         for n in self.data[i][j]:
    #                             if n == key:
    #                                 return self.data[i][j][n]
    #                     if j == key:
    #                         return self.data[i][j]
    #             else:
    #                 if i == key:
    #                     return self.data[i]
    #         except KeyError:
    #             print("Can't find key.\n")

    def find(self, key, filename=None, data=None):
        key = key.lower()
        if data is None:
            data = self.data
        for i in data:
            if i == key:
                return data[i]
            if type(data[i]) is dict:
                value = self.find(key, filename, data[i])
                if value is not None:
                    return value

    # locate in files.
    def locate(self, keywords):
        filenames = os.listdir(self.path)
        fitted_list = []
        for filename in filenames:
            fit = True
            for key in keywords:
                value = self.find(key, filename)
                fit &= (value == keywords[key])
            if fit:
                fitted_list.append(filename)
        if fitted_list:
            print(f"Find fitted files {fitted_list}")
            return fitted_list
        else:
            print("Can't find keys.\n")
