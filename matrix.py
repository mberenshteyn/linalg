from fractions import *
import numpy.polynomial.polynomial as np

class Matrix:

    """
    Constructors
    """

    def __init__(self, array, aug = None):
        self.rows = [[Fraction(val) if type(val) != np.Polynomial else val for val in row] for row in array]
        self.unchanged_rows = [[Fraction(val) if type(val) != np.Polynomial else val for val in row] for row in array] #Never change this variable!
        self.aug_matrix = aug
        self.cleanup()

    @classmethod
    def identity(cls, size):
        elements = [[0 for i in range(size)] for i in range(size)]
        for i in range(size):
            elements[i][i] = 1
        return Matrix(elements)

    @classmethod
    def col_vector(cls, array):
        elements = [[el] for el in array]
        return Matrix(elements)

    @classmethod
    def row_vector(cls, array):
        elements = [array]
        return Matrix(elements)

    @classmethod
    def copy(cls, mat):
        """
        Returns a new Matrix object with the same dimensions and values as the initialized Matrix instance.
        Does not retain augments.
        """
        return Matrix(mat.unchanged_rows)

    @classmethod
    def copy_current(cls, mat):
        """
        Returns a new Matrix object with the same dimensions and values as the current Matrix instance.
        Does not retain augments.
        """
        return Matrix(mat.rows)

    """
    Display Methods
    """

    def __str__(self):
        """
        Returns the matrix instance as represented by a list of lists (rows).
        """
        return str([[str(val) for val in row] for row in self.rows])

    def print_nice(self):
        """
        Prints the rows one-by-one for ease of reading in shell.
        """
        for i in range(self.num_row):
            formatted_row = []
            for j in range(self.num_col):
                value = self._get_val(i, j)
                if type(value) is not np.Polynomial:
                    formatted_row.append(str(value))
                else:
                    formatted_row.append(polynomial_format(value))
            print(str(formatted_row))

    def cleanup(self):
        """
        Iterates through every value in the matrix instance to round numbers to 12 digits.
        Also converts redundant floats (e.g. 1.0, -0.0) into ints.
        """
        if self.aug_matrix is not None:
            self.aug_matrix.cleanup()
        for i in range(self.num_row):
            for j in range(self.num_col):
                val_ij = self._get_val(i, j)
                if type(val_ij) is not np.Polynomial and val_ij == int(val_ij):
                    self._change_val(i, j, int(val_ij))

    """
    Getters
    """

    @property
    def num_row(self):
        """
        Returns the number of rows in the matrix instance.
        """
        return len(self.rows)

    @property
    def num_col(self):
        """
        Returns the number of columns in the matrix instance.
        """
        return len(self.rows[0])

    def _get_row(self, i):
        """
        Returns the ith row of the matrix instance as a row vector matrix.
        """
        return Matrix.row_vector(self.rows[i])

    def _get_col(self, j):
        """
        Returns the jth column of the matrix instance as a column vector matrix.
        """
        elements = [row[j] for row in self.rows]
        return Matrix.col_vector(elements)

    def _get_val(self, i, j):
        """
        Returns the value at the ith row and jth column of the matrix instance.
        """
        return self.rows[i][j]

    """
    Setters
    """

    def _change_val(self, i, j, new_val):
        """
        Changes the value at the ith row and jth column of the matrix instance.
        """
        if type(new_val) == np.Polynomial:
            self.rows[i][j] = new_val
        else:
            self.rows[i][j] = Fraction(new_val)

    def _init_val(self, i, j, val):
        """
        Like _change_val, but applies to the unchanged_rows list of lists as well.
        Primarily for purpose of producing matrices properly after adding, subtracting, or multiplying.
        """
        self._change_val(i, j, val)
        if type(val) == np.Polynomial:
            self.unchanged_rows[i][j] = val
        else:
            self.unchanged_rows[i][j] = Fraction(val)

    def _change_row(self, i, new_row):
        """
        Changes the value at the ith row with a given new row.
        Should only ever be called as part of the _swap_rows method.
        """
        assert new_row.num_row == 1
        for j in range(new_row.num_col):
            newVal = new_row._get_val(0, j)
            self._change_val(i, j, newVal)

    """
    Row Operations
    """

    def _scale_row(self, i, scal):
        """
        Scales the ith row of the matrix instance by scal.
        """
        if (self.aug_matrix is not None):
            self.aug_matrix._scale_row(i, scal)
        scaled_row_as_list = [scal * self._get_val(i, j) for j in range(self.num_col)]
        scaled_row = Matrix.row_vector(scaled_row_as_list)
        self._change_row(i, scaled_row)
        self.cleanup()

    def _swap_rows(self, i, j):
        """
        Interchanges the ith and jth rows of the matrix instance.
        Should only ever be called as part of another method (e.g. gauss_elim).
        """
        if (self.aug_matrix is not None):
            self.aug_matrix._swap_rows(i, j)
        row_at_i = self._get_row(i)
        row_at_j = self._get_row(j)
        self._change_row(i, row_at_j)
        self._change_row(j, row_at_i)

    def _sub_from_row(self, i, j, scal = 1):
        """
        Subtracts a scalar multiple of the jth row from the ith row of the matrix instance.
        """
        if (self.aug_matrix is not None):
            self.aug_matrix._sub_from_row(i, j, scal)
        new_row_as_list = [0 for _ in range(self.num_col)]
        for k in range(self.num_col):
            new_row_as_list[k] = self._get_val(i, k) - scal * self._get_val(j, k)
        new_row = Matrix.row_vector(new_row_as_list)
        self._change_row(i, new_row)

    """
    Arithmetic Operations
    """

    def __eq__(self, other):
        """
        Returns True if both matrices have the same dimensions and consist of the same values.
        """
        if self is other:
            return True
        elif type(other) != Matrix:
            return False
        elif self.num_row != other.num_row or self.num_col != other.num_col:
            return False
        else:
            for i in range(self.num_row):
                for j in range(self.num_col):
                    if self._get_val(i, j) != other._get_val(i, j):
                        return False
            return True

    def __add__(self, other):
        """
        Returns a new matrix produced by summing the values in two matrix instances.
        Requires that the two matrices have the same dimensions.
        """
        assert isinstance(other, Matrix), "You cannot add non-matrix objects"
        assert self.num_row == other.num_row and self.num_col == other.num_col, "Two matrices must be the same size in order for addition to work correctly"
        sum = Matrix([[0 for _ in range(self.num_col)] for _ in range(self.num_row)])
        for i in range(self.num_row):
            for j in range(self.num_col):
                sum._init_val(i, j, self._get_val(i, j) + other._get_val(i, j))
        return sum

    def __sub__(self, other):
        """
        Returns a new matrix produced by subtracting the values in two matrix instances.
        Requires that the two matrices have the same dimensions.
        """
        assert isinstance(other, Matrix), "You cannot subtract non-matrix objects"
        assert self.num_row == other.num_row and self.num_col == other.num_col, "Two matrices must be the same size in order for subtraction to work correctly"
        sub = Matrix([[0 for _ in range(self.num_col)] for _ in range(self.num_row)])
        for i in range(self.num_row):
            for j in range(self.num_col):
                sub._init_val(i, j, self._get_val(i, j) - other._get_val(i, j))
        return sub

    def __mul__(self, other):
        """
        Chooses the appropriate matrix multiplication method, based on whether the given value is another matrix or a number.
        If the value is of any other type, the operation raises a TypeError.
        """
        if isinstance(other, Matrix):
            return self._matmul(other)
        elif type(other) == int or type(other) == float or type(other) == np.Polynomial:
            return self._scmul(other)
        raise TypeError(f"Cannot multiply matrices by objects of type {type(other)}")

    def __rmul__(self, other):
        """
        An alternative to the __mul__ method, used particularly if a scalar or polynomial is the preceding value.
        Calls upon __mul__ for functionality.
        """
        return self * other

    def _matmul(self, other):
        """
        Returns a new matrix of dimensions (self.num_row) x (other.num_col) produced by multiplying the values in two matrix instances.
        Requires that self.num_col == other.num_row
        """
        assert self.num_col == other.num_row, "The number of columns in the first matrix must equal the number of rows in the second matrix for multiplication to work correctly"
        product = Matrix([[0 for _ in range(other.num_col)] for _ in range(self.num_row)])
        for i in range(product.num_row):
            for j in range(product.num_col):
                sum_corr_prod = sum([x * y for x, y in zip(self._get_row(i), other._get_col(j))])
                product._init_val(i, j, sum_corr_prod)
        product.cleanup()
        return product

    def _scmul(self, scal):
        """
        Returns a new matrix produced by multiplying every value in the matrix instance by scal.
        """
        product = Matrix([[0 for _ in range(self.num_col)] for _ in range(self.num_row)])
        for i in range(product.num_row):
            for j in range(product.num_col):
                product._init_val(i, j, scal * self._get_val(i, j))
        product.cleanup()
        return product

    """
    Misc Methods
    """

    def _augment(self, aug_mat):
        """
        Augments a matrix to the right of the existing matrix instance.
        """
        assert isinstance(aug_mat, Matrix) and aug_mat.num_row == self.num_row, \
                f"The augmented column must have {self.num_row} rows"
        assert self.aug_matrix == None, f"The matrix instance is already augmented"
        self.aug_matrix = aug_mat

    def _deaugment(self):
        """
        Clears the augmented matrix if it exists, otherwise does nothing.
        """
        self.aug_matrix = None
