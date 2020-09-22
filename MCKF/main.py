from simulation_run import sim_run
from data_manager import Manager

sigmas = [2]
for sigma in sigmas:
    sim_run(sigma)
# mg = Manager()
# keywords = {"description": "Non-Gaussian."}
# targets = mg.locate(keywords)
# for i in targets:
#     print(i)
#     print(mg.find("sigma", i), mg.find("MSE", i))
