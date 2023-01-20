import matplotlib.pyplot as plt
import numpy as np

class Spectrum:
    def __init__(self,resolution):
        self.resolution=resolution
        self.output=None

    def draw(self):
        spectrum = np.zeros([self.resolution, self.resolution , 3], dtype=np.uint8)  # init the array
        spectrum[:, :, 1] = np.linspace(0, 255, self.resolution)
        spectrum = spectrum.swapaxes(0, 1)
        spectrum[:, :, 2] = np.linspace(self.resolution-1, 0, self.resolution)
        spectrum[:, :, 0] = np.linspace(0, self.resolution-1, self.resolution)
        self.output=spectrum
        return spectrum

    def show(self):
        plt.imshow(self.output, aspect= 'auto')
        plt.show()


self_Spectrum=Spectrum(260)
self_Spectrum.draw()
self_Spectrum.show()
