from simulation import Simulation as Sim
from data_manager import Manager
import winsound

mg = Manager()
repeat = 100
sigmas = [2, 4, 6, 10]
sim = Sim(repeat)
sim.sys_run()
for sigma in sigmas:
    data = sim.filter_run(sigma, ukf=1, ekf=1, mcekf=1)
    mg.view_data(data)
    mg.save_data(data)
    print(f'sigma = {sigma}')
    mg.plot_all()
winsound.Beep(300, 300)
# mg.show()

# keywords = {"description": "Non-Gaussian."}
# targets = mg.locate(keywords)
# for i in targets:
#     print(i)
#     print(mg.find("sigma", i), mg.find("MSE", i))
