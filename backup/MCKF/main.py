from SimulationSetup import sim_run
from data_manager import Manager

sigmas = [2]
repeat = 10
for sigma in sigmas:
    data = sim_run(sigma, repeat)
    mg = Manager()
    mg.view_data(data)
    mg.plot_all()
    mg.show()

# keywords = {"description": "Non-Gaussian."}
# targets = mg.locate(keywords)
# for i in targets:
#     print(i)
#     print(mg.find("sigma", i), mg.find("MSE", i))
