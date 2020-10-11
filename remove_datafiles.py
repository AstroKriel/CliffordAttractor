## load modules
import os
import numpy as np

## get list of files
filepath_program = os.path.dirname(os.path.realpath(__file__))
[os.remove(filepath_program+'/'+tmp_file) for tmp_file in os.listdir('.') if tmp_file.__contains__('.png')]
[os.remove(filepath_program+'/'+tmp_file) for tmp_file in os.listdir('.') if tmp_file.__contains__('.txt')]
