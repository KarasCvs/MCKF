from data_manager import Manager


mg = Manager()
mg.read_data("2020-10-02 17.20.49")
# mg.plot("add noise")
mg.plot_all()
# target = mg.find("r")
# print(target)
mg.show()