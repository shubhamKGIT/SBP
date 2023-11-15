from sbputils.blackbody import PlanksBlackBody, Wien
import matplotlib.pyplot as plt


def simple_debug():
    items: list = [1., 2., 3.]
    for _, item in enumerate(items):
        print(item)

def plot_radiance(wavelengths: list, brightness_lists: list, labels: list, ylim: list, yscale: str) -> None:
    fig, ax = plt.subplots()
    for _, brightness in enumerate(brightness_lists):
        ax.plot(wavelengths, brightness)
    """plt.xlim(100, 10000)
    plt.ylim(1e-25, 1e10)
    plt.ylabel("log")"""
    ax.legend(labels)
    plt.yscale(yscale)
    plt.ylim(ylim)
    plt.show()

def scale_radiance(radiance: list) -> list:
    "scales radiance from W/m2-nm to W/cm2-1000nm"
    return list(map(lambda x: x/1e10, radiance))

WAVELENGTHS = [50, 100, 300, 1000, 3000, 6000, 10000]    # in nm
toy_star = PlanksBlackBody()
radiance_2000K = toy_star.brightness(wavelength_range=WAVELENGTHS, temperature=2000, in_nm=True)
radiance_1000K = toy_star.brightness(wavelength_range=WAVELENGTHS, temperature=1000, in_nm=True)
radiance_wien_2000K = toy_star.wien_brightness(wavelength_range= WAVELENGTHS, temperature=2000, in_nm=True)

YLIM = [1e1, 2e20]
LABELS = labels = ["plank @2000K", "wien @2000K", "plank @1000K"]
YSCALE = "log"
print(radiance_2000K, radiance_wien_2000K)
plot_radiance(WAVELENGTHS, [radiance_2000K, radiance_wien_2000K, radiance_1000K], LABELS, YLIM, YSCALE)

YSCALE = "linear"
plot_radiance(WAVELENGTHS, [scale_radiance(radiance_2000K)], ["scaled rad, W/cm2-1000nm"], [0, 1e2/2], YSCALE)