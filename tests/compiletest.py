#############################################################################################
# Compiles a circuit file and spits out the compiled output. Useful for compiler debugging. #
#############################################################################################

#
# File: compiletest.py
#

import os
import sys
sys.path.append(os.path.abspath("."))

from libcirc.compilecirc import *  # import error? try executing from root dir!

circfile = "circuits/toffoli.circ"

print(compileCircuit(fname=circfile))
