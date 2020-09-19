from simulation_run import sim_run
from data_manager import Manager

# sigmas = [2, 3, 5, 10]
# for sigma in sigmas:
#     sim_run(sigma)
mg = Manager()
keywords = {"beta": 2}
targets = mg.locate(keywords)
for i in targets:
    print(i)
    print(mg.find("sigma", i), mg.find("MSE", i))
