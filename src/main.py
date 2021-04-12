from simulation import Simulation as Sim
from data_manager import Manager
import winsound

mg = Manager()

repeat = 30
sigmas = [3, 5, 8, 10]
sim = Sim(repeat)
sim.sys_run()
for sigma in sigmas:
    print(f'sigma = {sigma}')
    data = sim.filter_run(sigma, mcekf=1, ekf=1, ukf=1)
    # mg.save_data(data)
    mg.view_data(data)
    mg.plot_mse()
    mg.plot_mse1()
    # mg.plot_states()
    # mg.plot(mg.find('IMCEKF, G(R)'), f'sigma={sigma}, G(R)')
    mg.plot(mg.find('IMCEKF, G(Q)'), f'sigma={sigma}, G(Q)')
    mg.plot(mg.find('IMCEKF, K'), f'sigma={sigma}, IMCEKF, K')
    # mg.plot(mg.find('EKF, K'), f'sigma={sigma}, EKF, K')
    # mg.plot(mg.find('IMCEKF, e'), 'IMCEKF, e')
    # mg.plot(mg.find('EKF, K'), 'EKF, K')
    mg.plot(mg.find('IMCEKF, L'), f'sigma={sigma}, L')
winsound.Beep(300, 300)

# keywords = {"description": "Gaussian test"}
# targets = mg.locate(keywords)
# for i in targets:
#     print(i)
#     print(mg.find("sigma", i), mg.find("MSE", i))
mg.show()
