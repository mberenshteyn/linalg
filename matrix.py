from __future__ import annotations
from fractions import *
from typing import List
import numpy.polynomial.polynomial as np

class Matrix:

    """
    Constructors
    """

    def __init__(self, array: List, aug: Matrix = None) -> Matrix:
        self.rows = [[Fraction(val) if type(val) != np.Polynomial else val for val in row] for row in array]
        self.unchanged_rows = [[Fraction(val) if type(val) != np.Polynomial else val for val in row] for row in array]
        self.aug_matrix = aug
        self.cleanup()

    @classmethod
    def identity(cls, size: int) -> Matrix:
        elements = [[0 for i in range(size)] for i in range(size)]
        for i in range(size):
            elements[i][i] = 1
        return Matrix(elements)

    @classmethod
    def diagonal(cls, array: List) -> Matrix:
        dim : int = len(array)
        rows = []
        for index, value in enumerate(array):
            row = [0 for _ in range(dim)]
            row[index] = value
            rows.append(row)
        return Matrix(rows)

    @classmethod
    def col_vector(cls, array: List) -> Matrix:
        elements = [[el] for el in array]
        return Matrix(elements)

    @classmethod
    def row_vector(cls, array: List) -> Matrix:
        elements = [array]
        return Matrix(elements)

    @classmethod
    def merge_cols(cls, array: List[Matrix]) -> Matrix:
        col_count = len(array)
        row_count = array[0].num_row
        base = [[0 for _ in range(col_count)] for _ in range(row_count)]
        base_matrix = Matrix(base)
        for index, vector in enumerate(array):
            base_matrix._change_col(index, vector, True)
        return base_matrix

    @classmethod
    def merge_rows(cls, array: List[Matrix]) -> Matrix:
        row_count = len(array)
        col_count = array[0].num_col
        base = [[0 for _ in range(col_count)] for _ in range(row_count)]
        base_matrix = Matrix(base)
        for index, vector in enumerate(array):
            base_matrix._change_row(index, vector, True)
        return base_matrix

    @classmethod
    def copy(cls, mat: Matrix) -> Matrix:
        """
        Returns a new Matrix object with the same dimensions and values as the initialized Matrix instance.
        Does not retain augments.
        """
        return Matrix(mat.unchanged_rows)

    @classmethod
    def copy_current(cls, mat: Matrix) -> Matrix:
        """
        Returns a new Matrix object with the same dimensions and values as the current Matrix instance.
        Does not retain augments.
        """
        return Matrix(mat.rows)

    """
    Display Methods
    """

    def __str__(self) -> str:
        """
        Returns the matrix instance as represented by a list of lists (rows).
        """
        return str([[str(val) for val in row] for row in self.rows])

    def print_nice(self) -> None:
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

    def cleanup(self) -> None:
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
    def num_row(self) -> int:
        """
        Returns the number of rows in the matrix instance.
        """
        return len(self.rows)

    @property
    def num_col(self) -> int:
        """
        Returns the number of columns in the matrix instance.
        """
        return len(self.rows[0])

    def _get_row(self, i: int) -> Matrix:
        """
        Returns the ith row of the matrix instance as a row vector matrix.
        """
        return Matrix.row_vector(self.rows[i])

    def _get_col(self, j: int) -> Matrix:
        """
        Returns the jth column of the matrix instance as a column vector matrix.
        """
        elements = [row[j] for row in self.rows]
        return Matrix.col_vector(elements)

    def _get_val(self, i: int, j: int) -> Matrix:
        """
        Returns the value at the ith row and jth column of the matrix instance.
        """
        return self.rows[i][j]

    """
    Setters
    """

    def _change_val(self, i: int, j: int, new_val) -> None:
        """
        Changes the value at the ith row and jth column of the matrix instance.
        """
        if type(new_val) == np.Polynomial:
            self.rows[i][j] = new_val
        else:
            self.rows[i][j] = Fraction(new_val)

    def _init_val(self, i: int, j: int, val) -> None:
        """
        Like _change_val, but applies to the unchanged_rows list of lists as well.
        Primarily for purpose of producing matrices properly after adding, subtracting, or multiplying.
        """
        self._change_val(i, j, val)
        if type(val) == np.Polynomial:
            self.unchanged_rows[i][j] = val
        else:
            self.unchanged_rows[i][j] = Fraction(val)

    def _change_row(self, i: int, new_row: Matrix, init_values: bool = False) -> None:
        """
        Changes the value at the ith row with a given new row.
        """
        assert new_row.num_row == 1
        for j in range(new_row.num_col):
            newVal = new_row._get_val(0, j)
            if init_values:
                self._init_val(i, j, newVal)
            else:
                self._change_val(i, j, newVal)

    def _change_col(self, j: int, new_col: Matrix, init_values: bool = False) -> None:
        """
        Changes the value at the jth column with a given new column.
        """
        assert new_col.num_col == 1
        for i in range(new_col.num_row):
            newVal = new_col._get_val(i, 0)
            if init_values:
                self._init_val(i, j, newVal)
            else:
                self._change_val(i, j, newVal)

    """
    Row Operations
    """

    def _scale_row(self, i: int, scal) -> None:
        """
        Scales the ith row of the matrix instance by scal.
        """
        if (self.aug_matrix is not None):
            self.aug_matrix._scale_row(i, scal)
        scaled_row_as_list = [scal * self._get_val(i, j) for j in range(self.num_col)]
        scaled_row = Matrix.row_vector(scaled_row_as_list)
        self._change_row(i, scaled_row)
        self.cleanup()

    def _swap_rows(self, i: int, j: int) -> None:
        """
        Interchanges the ith and jth rows of the matrix instance.
        """
        if (self.aug_matrix is not None):
            self.aug_matrix._swap_rows(i, j)
        row_at_i = self._get_row(i)
        row_at_j = self._get_row(j)
        self._change_row(i, row_at_j)
        self._change_row(j, row_at_i)

    def _sub_from_row(self, i: int, j: int, scal = 1) -> None:
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

    def __eq__(self, other: Matrix) -> bool:
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

    def __add__(self, other: Matrix) -> Matrix:
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

    def __sub__(self, other: Matrix) -> Matrix:
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

    def __mul__(self, other) -> Matrix:
        """
        Chooses the appropriate matrix multiplication method, based on whether the given value is another matrix or a number.
        If the value is of any other type, the operation raises a TypeError.
        """
        if isinstance(other, Matrix):
            return self._matmul(other)
        elif type(other) == int or type(other) == float or type(other) == np.Polynomial:
            return self._scmul(other)
        raise TypeError(f"Cannot multiply matrices by objects of type {type(other)}")

    def __rmul__(self, other) -> Matrix:
        """
        An alternative to the __mul__ method, used particularly if a scalar or polynomial is the preceding value.
        Calls upon __mul__ for functionality.
        """
        return self * other

    def _matmul(self, other: Matrix) -> Matrix:
        """
        Returns a new matrix of dimensions (self.num_row) x (other.num_col) produced by multiplying the values in two matrix instances.
        Requires that self.num_col == other.num_row
        """
        assert self.num_col == other.num_row, "The number of columns in the first matrix must equal the number of rows in the second matrix for multiplication to work correctly"
        product = Matrix([[0 for _ in range(other.num_col)] for _ in range(self.num_row)])
        for i in range(product.num_row):
            for j in range(product.num_col):
                sum_corr_prod = self._sum_corr_prod(other, i, j)
                product._init_val(i, j, sum_corr_prod)
        product.cleanup()
        return product

    def _scmul(self, scal) -> Matrix:
        """
        Returns a new matrix produced by multiplying every value in the matrix instance by scal.
        """
        product = Matrix([[0 for _ in range(self.num_col)] for _ in range(self.num_row)])
        for i in range(product.num_row):
            for j in range(product.num_col):
                product._init_val(i, j, scal * self._get_val(i, j))
        product.cleanup()
        return product

    def _sum_corr_prod(self, other: Matrix, i: int, j: int) -> int:
        row = self._get_row(i)
        col = other._get_col(j)
        sum = 0
        for z in range(row.num_col):
            sum += (row._get_val(0, z) * col._get_val(z, 0))
        return sum

    """
    Misc Methods
    """

    def _del_row(self, i: int) -> None:
        """
        Removes the row corresponding with index i from the matrix instance.
        """
        self.rows.pop(i)

    def _del_col(self, i: int) -> None:
        """
        Removes the column corresponding with index i from the matrix instance.
        """
        for row in self.rows:
            row.pop(i)

    def _augment(self, aug_mat: Matrix) -> None:
        """
        Augments a matrix to the right of the existing matrix instance.
        """
        assert isinstance(aug_mat, Matrix) and aug_mat.num_row == self.num_row, \
                f"The augmented column must have {self.num_row} rows"
        assert self.aug_matrix == None, f"The matrix instance is already augmented"
        self.aug_matrix = aug_mat

    def _deaugment(self) -> None:
        """
        Clears the augmented matrix if it exists, otherwise does nothing.
        """
        self.aug_matrix = None
