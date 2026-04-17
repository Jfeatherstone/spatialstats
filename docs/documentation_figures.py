"""
This script will create all of the figures for the documentation.

Conventions
-----------

- All files should be saved in png format.

- Make sure to close the figures after they are saved using ``plt.close()``;
this script is not meant to display anything.

- For a typical standalone plot the default size should be 4.5x4.5 inches at
default DPI. Feel free to change this as needed, but if you have no other
constraints, try to stick to this.

"""

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import spatialstats
import os, pathlib

from tqdm import tqdm

# The source file directory, used for saving the images
SOURCE_DIR = os.path.join((pathlib.Path(__file__).parent.resolve()), 'source/')

# For now this will just be in the source dir, but it might be changed by
# an argument.
SAVE_DIR = os.path.join(SOURCE_DIR, 'images')

# Since we have to save lots of figures
def savefig(fname):
    plt.savefig(os.path.join(SAVE_DIR, fname), bbox_inches='tight')


if __name__ == '__main__':

    # Create the save directory if it doesn't exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f'Created save file directory: {SAVE_DIR}')
        print()


    # TODO
