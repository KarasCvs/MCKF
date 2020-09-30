from SimulationSetup import sim_run
from data_manager import Manager

sigmas = [2]
repeat = 30
mg = Manager()
for sigma in sigmas:
    data = sim_run(repeat, sigma, ukf=1, mcukf=1)
    mg.view_data(data)
    # print(mg.states['mcekf states'])
    # mg.save_data(data, "test")
    print(f'sigma = {sigma}')
    mg.plot_all()
# mg.show()

# keywords = {"description": "Non-Gaussian."}
# targets = mg.locate(keywords)
# for i in targets:
#     print(i)
#     print(mg.find("sigma", i), mg.find("MSE", i))
