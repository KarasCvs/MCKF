from simulation import Simulation as Sim
from data_manager import Manager
import winsound

mg = Manager()

repeat = 10
sigmas = [2, 4, 6, 10]
sim = Sim(repeat)
sim.sys_run()
for sigma in sigmas:
    print(f'sigma = {sigma}')
    data = sim.filter_run(sigma, mcekf=1, imcekf=1, imcekf3=1, ekf=1)
    mg.view_data(data)
    # mg.save_data(data)
    mg.plot_mse()
    mg.plot_mse1()
    mg.plot_states()
winsound.Beep(300, 300)

# keywords = {"description": "Gaussian test"}
# targets = mg.locate(keywords)
# for i in targets:
#     print(i)
#     print(mg.find("sigma", i), mg.find("MSE", i))
mg.show()
