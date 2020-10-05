from data_manager import Manager
import numpy as np


mg = Manager()
mg.read_data()
test = mg.find("mse")
print(test)