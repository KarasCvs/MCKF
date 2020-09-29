from SimulationSetup import sim_run
from data_manager import Manager

sigmas = [4]
repeat = 10
mg = Manager()
for sigma in sigmas:
    data = sim_run(repeat, sigma, ukf=1, ekf=1, mcekf=1)
    mg.view_data(data)
    # print(mg.states['mcekf states'])
    # mg.save_data(data)
    mg.plot_all()
    mg.show()

# keywords = {"description": "Non-Gaussian."}
# targets = mg.locate(keywords)
# for i in targets:
#     print(i)
#     print(mg.find("sigma", i), mg.find("MSE", i))
