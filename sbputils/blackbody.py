
import numpy as np

class PlanksBlackBody:
    "object to define the constants and the blackbody equation as abstract class"
    def __init__(self) -> None:
        self.constants()

    def constants(self) -> None:
        self.plank_constant: float = 6.626e-34    #unit: ML2T-1 or J.Hz-1
        self.speed_of_light: float = 2.99792458e8    # unit: MT-1
        self.boltzmann_constant: float = 1.380649e-23    # unit: M2LT-2K-1
        self.c1 = 2*np.pi*self.plank_constant*(self.speed_of_light**2)
        self.c2 = self.plank_constant*self.speed_of_light/ self.boltzmann_constant
    
    def brightness(self, wavelength_range: list = [400.0, 532.0, 640.0, 1000.0, 1500.0], in_nm: bool = True, temperature: float = 600.) -> list:
        radiation = []
        if in_nm:
            wavelength_range = list(map(lambda x: x/1e9, wavelength_range))
        else:
            pass
        for _, wavelength in enumerate(wavelength_range):
            rad = (self.c1/ (wavelength**5))*(1/ (np.exp(self.c2/ (wavelength* temperature)) - 1))
            radiation.append(rad)
        return radiation
    
    def wien_brightness(self, wavelength_range: list = [400, 532, 640, 1000, 1500], in_nm: bool = True, temperature: float = 600.) -> list:
        radiation = []
        if in_nm:
            wavelength_range = list(map(lambda x: x/1e9, wavelength_range))
        else:
            pass
        for _, wavelength in enumerate(wavelength_range):
            rad = (self.c1/ (wavelength**5))*(1/ (np.exp(self.c2/ (wavelength* temperature))))
            radiation.append(rad)
        return radiation

class Wien(PlanksBlackBody):
    def __init__(self) -> None:
        super.__init__()
    
    def brightness(self, wavelength_range: list = [400, 532, 640, 1000, 1500], temperature: float = 600.9) -> list:
        radiation = []
        for _, wavelength in enumerate(wavelength_range):
            rad = (self.c1/ (wavelength**5))*(np.exp(-self.c2/ (wavelength*temperature)))
            radiation.append(rad)
        return radiation
    


