from data_manager import Manager


mg = Manager()
mg.read_data("2020-10-06 12.24.25")
mg.plot("add noise")
# mg.read_data("2020-10-06 12.24.25")
# mg.plot("add noise")
# mg.plot_all()
# target = mg.find("r")
# print(target)
mg.show()