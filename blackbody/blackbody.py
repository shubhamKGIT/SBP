
import os
import sys
import pathlib
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

class BlackBody:
    "object to define the constants and the blackbody equation as abstract class"
    def __init__(self, temperature) -> None:
        if temperature:
            self.T = temperature
        else:
            self.T = 300
        self.constants(self.T)

    def constants(self, temperature: float = 300) -> None:
        self.plank_constant: float = 6.626e-34    #unit: ML2T-1 or J.Hz-1
        self.speed_of_light: float = 2.99792458e8    # unit: MT-1
        self.boltzmann_constant: float = 1.380649e-23    # unit: M2LT-2K-1
        self.c1 = 2*self.plank_constant*(self.speed_of_light**2)
        self.c2 = self.plank_constant*self.speed_of_light/ self.boltzmann_constant
    
    def __repr__(self):
        return f"body has constants: {self.c1, self.c2}, and temperature {self.T}"
    
    def brightness(self, wavelengths: list = [400.0, 532.0, 640.0, 1000.0, 1500.0], in_nm: bool = True, temperature: float = 600.0):
        "assumes temperature is given in Kelvins (K) but checks for unit of "
        self.I_rad = np.zeros(len(wavelengths))
        self.I_rad_wien = np.zeros(len(wavelengths))
        if in_nm:
            wavelengths = list(map(lambda x: x/1e9, wavelengths))    # get it in meters
        else:
            wavelengths = list(map(lambda x: x/1e6, wavelengths))   # assuming passed as microns otherwise
        for i, wavelength in enumerate(wavelengths):
            rad = (self.c1/ (wavelength**5))*(1/(np.exp(self.c2/ (wavelength* temperature)) - 1))/1e6
            rad_wien = (self.c1/ (wavelength**5))*(1/np.exp(self.c2/ (wavelength* temperature)))/1e6
            self.I_rad[i] = rad
            self.I_rad_wien[i] = rad_wien
    
    def adjust_brightness(self, parameters: dict):
        "adjust for area of emission, solid anagle, efficiency of receiving system"
        try:
            solid_angle = parameters["solid_angle"]
            area = parameters["emitting_area"]
        except:
            raise Exception("paramters passed wrong, check the input dict")
        self.adjusted_rad  = np.pi*self.I_rad*solid_angle*area
        return self.adjusted_rad
    
    def get_brightness(self):
        if hasattr(self, 'I_rad'):
            return self.I_rad, self.I_rad_wien
        else:
            return None

    def blackbody_plot(self):
        pass

    
def test_blackbody():
    "intantiate class and test methods"
    b1 = BlackBody(1000)
    lambdas = np.arange(450, 680, 5)
    b1.brightness(wavelengths=lambdas, temperature=b1.T)
    B_w_s, B_w_s_wien = b1.get_brightness()
    print(lambdas, B_w_s)
    fig, ax = plt.subplots(1, 3)
    ax[0].plot(lambdas, B_w_s)
    ax[1].plot(lambdas, B_w_s_wien)
    print(b1)
    params = {"solid_angle": 0.1, "emitting_area": 0.5e-4}
    adjusted_emissions = b1.adjust_brightness(params)
    ax[2].plot(lambdas, adjusted_emissions)
    plt.show()


if __name__=="__main__":
    test_blackbody()
