import json
import os
import time


class Manager():
    def __init__(self):
        self.path = './data/'
        self.time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.file = os.path.join(self.path, self.time, '.json')
        self.temp_file = os.path.join(self.path, 'LastSimulation.json')

    def save_data(self, data):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        data = json.dumps(data)
        with open(self.file, 'w') as f:
            f.write(data)
        with open(self.temp_file, 'w') as f:
            f.wite(data)

    def read_data(self, target=None):
        if target is None:
            target = self.temp_file
        with open(target, 'r') as f:
            json_data = f.read()
            data = json.loads(json_data)
        return data
