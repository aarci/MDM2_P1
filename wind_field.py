import matplotlib.pyplot as plt
import numpy as np
from perlin_numpy import (
    generate_perlin_noise_2d
)

def generate_wind_field(shape = (1000,1000)):

    '''
    Outputs X and Y components of the random 2D vector field corresponding to the wind
    '''

    np.random.seed()

    angles = (2 * np.pi) * generate_perlin_noise_2d(shape, (8,8))
    mags = 10 * (0.1 + generate_perlin_noise_2d(shape, (8,8)))

    X = mags * np.cos(angles)
    Y = mags * np.sin(angles)

    return X, Y

def generate_wind_field_uniform(angle, shape = (1000, 1000)):

    '''
    Outputs X and Y components of the uniform 2D vector field corresponding to the wind
    '''

    np.random.seed()

    angles = np.full(shape, angle)
    mags = np.full(shape, 2)

    X = mags * np.cos(angles)
    Y = mags * np.sin(angles)

    return X, Y

def plot_wind_field(X, Y):
    
    plt.quiver(X, Y, units = 'xy', minlength = 0)
    #plt.imshow(mags, cmap='gray', interpolation='lanczos')
    #plt.colorbar()
    #plt.savefig("wind.svg")
    plt.show()

def main():

    X, Y = generate_wind_field()
    plot_wind_field(X, Y)

if __name__ == "__main__":
    main()
