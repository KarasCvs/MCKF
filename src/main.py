from simulation import Simulation as Sim
from data_manager import Manager
import winsound

mg = Manager()

repeat = 10
sigmas = [3, 10]
sim = Sim(repeat)
sim.sys_run()
for sigma in sigmas:
    print(f'sigma = {sigma}')
    data = sim.filter_run(sigma, mcekf=0, imcekf=1, ekf=1, imcekf2=1)
    # mg.save_data(data)
    mg.view_data(data)
    mg.plot_mse()
    mg.plot_mse1()
    mg.plot_states()
    mg.plot(mg.find('IMCEKF2, sigma_square_R'), 'Sigma of R')
    mg.plot(mg.find('IMCEKF2, sigma_square_Q'), 'Sigma of Q')
    # mg.plot(mg.find('IMCEKF, G(Q)'), 'IMCEKF, G(Q)')
    # mg.plot(mg.find('MCEKF_DS, L'), 'MCEKF, L')
winsound.Beep(300, 300)

# keywords = {"description": "Gaussian test"}
# targets = mg.locate(keywords)
# for i in targets:
#     print(i)
#     print(mg.find("sigma", i), mg.find("MSE", i))
mg.show()
