from SimulationSetup import sim_run
from data_manager import Manager
import winsound

sigmas = [2, 4, 6, 8, 10]
repeat = 100
mg = Manager()
for sigma in sigmas:
    data = sim_run(repeat, sigma, ukf=1, ekf=1, mcekf=1, mcekf1=0)
    mg.view_data(data)
    # print(mg.states['mcekf states'])
    mg.save_data(data)
    print(f'sigma = {sigma}')
    mg.plot_all()
winsound.Beep(500, 1000)
mg.show()

# keywords = {"description": "Non-Gaussian."}
# targets = mg.locate(keywords)
# for i in targets:
#     print(i)
#     print(mg.find("sigma", i), mg.find("MSE", i))
