from data_manager import Manager
import matplotlib.pyplot as plt


mg = Manager()
states = []
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}


# targets = mg.locate({"description": "Gaussian, 3 states"})
targets = mg.locate({"description": "Impulse, 3 states"})
mg.read_data(targets[0])
timeline = mg.find("time line")
ekf_mse = mg.find('EKF mse')
ukf_mse = mg.find('UKF mse')
# TA MSE
ekf_tamse = mg.find('EKF ta_mse')
ukf_tamse = mg.find('UKF ta_mse')
# for target in targets:
#     sigma = mg.find('sigma')
#     print(f'sigma = {sigma}')
ekf_tamse = mg.find('EKF ta_mse')
# MSE
# for i in range(3):
#     plt.figure(i)
#     plt.plot(timeline, ekf_mse[i], label="ekf", color='r', linewidth=1, linestyle='-')
#     plt.plot(timeline, ukf_mse[i], label="ukf", color='blue', linewidth=1, linestyle='-')
# for target in targets:
#     mg.read_data(target)
#     mg.plot_mse()
#     sigma = mg.find('sigma')
#     print(f'sigma = {sigma}')
#     state = mg.find("IMCEKF states")
#     mse = mg.find("IMCEKF mse")
#     for i in range(3):
#         plt.figure(i)
#         plt.plot(timeline, mse[i], label=f"sigma={sigma}", linewidth=1, linestyle='-.')
#         plt.legend(prop=font)
#         plt.grid(True)
#         plt.xlim(0, 15)
#         plt.xticks(fontsize=13)
#         plt.yticks(fontsize=13)
#         plt.xlabel('Time(s)', fontsize=15)
#         plt.ylabel('MSE', fontsize=15)
#         # plt.title(f'Mean-Square Error(MSE) of x{i+1} Under Gaussian Noise', fontsize=20)
#         plt.title(f'Mean-Square Error(MSE) of x{i+1} Under Gaussian Noise and Outlires', fontsize=20)
# Plot shape of impulse noise
# impulse = mg.find("add noise")
# plt.plot(timeline, impulse[0])
# plt.grid(True)
# plt.xlim(0, 15)
# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)
# plt.xlabel('Time(s)', fontsize=15)
# plt.ylabel('MSE', fontsize=15)
# plt.title('Impulse noise', fontsize=20)

# plt.show()
