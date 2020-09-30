from data_manager import Manager
import numpy as np

mg = Manager()
mg.read_data("test")
noise = mg.find("obs noise")
print(np.std(noise))
# mg.show()