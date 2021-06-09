import sys
import os
from utils import image2midi

if not os.path.exists('../results/'):
    os.makedirs('../results/')

image2midi(sys.argv[1])