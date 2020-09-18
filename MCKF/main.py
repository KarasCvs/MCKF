from data_manager import Manager


mg = Manager()
# mg.read_data()
# mg.plot_all()
# mg.show()
target = {'sigma': 5}
a = mg.locate(target)
print(a)
