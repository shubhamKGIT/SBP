from files import Files, get_filename_with_ext
from pyrodata import Pyrodata

EXP = 21

# files
new_spectra_handle = Files(exp_number=EXP)
files = new_spectra_handle.files()

# pyrodata
pyro_handle = Pyrodata(exp_number=21, filenames= files)
spectra = pyro_handle.read_spectral_data()
print(spectra)
pyro_handle.plot_spectra()