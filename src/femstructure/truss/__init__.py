import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current_dir))
from ..frame import Frame

          
class Truss(Frame):
    def __init__(self, input_path):
        super().__init__(input_path, job='truss')

