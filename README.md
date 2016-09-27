# CircuitSimulator

A quantum circuit simulator based on http://arxiv.org/abs/1601.07601. Features:

- A simple programming language for quantum circuits in the Clifford+T gate set
- An algorithm for sampling from the output distribution of a circuit
- An algorithm for calculating the probability of a particular output for a circuit

This application is written in python, a cross-platform programming language.
A C implementation also is available for the slow part of both algorithms.
The C implementation is used by default, but a python-only version is also available.

See the documentation for a complete description.

## Installation

1. Install python

  Python 3 is recommended, although the code should still work in python 2.
  Download from https://www.python.org/downloads/ or use your favorite package manager.

2. Install numpy

  Download numpy from http://www.scipy.org/scipylib/download.html or use pip. Scipy is not required.

3. (Optional) Install matplotlib

  The `circuits/hiddenshift.py` test script generates a plot similar to the one in http://arxiv.org/abs/1601.07601 using matplotlib. You can disable this but setting `plot = False` on line 30, or you can download and install matplotlib from http://matplotlib.org/users/installing.html.

4. Download the code

  Press the clone or download button above and press "Download ZIP" or open up a terminal and type
  ```
  $ git clone git@github.com:patrickrall/CircuitSimulator.git
  ```

5. Compile the code C implementation

  On Mac OS X and Linux simply:
  ..1. Opening a terminal
  ..2. Change into the root directory: `$ cd CircuitSimulator`
  ..3. Call make: `$ make`

  On Windows, use the python implementation by passing the `-py` option to `main.py` until we figure things out.

## Usage

There are two ways to use CircuitSimulator:

1. Use the `main.py` handle to compile a circuit and run an algorithm.

  Examples:
  ```
  python main.py circuits/toffoli.circ MMM
  python main.py circuits/HTstack.circ 0
  ```
  A full usage statement is available:
  ```
  python main.py -h
  ```
  Or just read the documentation.

2. Import the `libcirc/probability.py` file and use the `probability()` and `sampleQubits()` functions.

  This is useful for automatically generated circuits that need to be evaluated. The `circuits/hiddenshift.py` file does exactly this: it randomly generates a circuit implementing the hidden shift algorithm and evaluates it immediately.
  Read `circuits/hiddenshift.py` for guidance when writing your own code and/or consult the documentation.
