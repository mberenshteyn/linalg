# Linear Algebra Calculator

This project is a work-in-progress implementation of linear algebra math in Python, with the intent of creating a WolframAlpha-like interface for users to solve linear algebra problems and view step-by-step solutions. The scope of the math will initially be limited to that of UC Berkeley's Math 54 course, but it could be expanded later on. 

Though most of the mathematical functions exist thanks to packages like [NumPy](http://www.numpy.org/), I'm writing most of them myself so that I can display individual steps of the calculations in later phases. 

## Phase 1 (Current)

The project is currently in its first phase, revolving around the implementation of the math and using it from the command line. The features thus far are basic matrix math, various property methods, row reduction/back substitution, calculation of determinant and eigenvalues, and solution of linear systems.

Updates in the near future include finding row/column/null/left null spaces, rank, dimension, calculation of eigenvectors, and diagonalization. Longer-term additions include handling vector inner product/length/orthogonality, the orthogonal decomposition theorem, the Gram-Schmidt process, QR factorization, orthogonal diagonalization, spectral decomposition, and singular value decomposition.

## Phase 2

The second phase revolves around building the first version of the interpreter/calculator and providing step-by-step math, including text descriptions of what is happening in each step.

## Phase 3

The third and final phase will focus on creating the full user interface for the calculator and its release (possibly as a web application). 
