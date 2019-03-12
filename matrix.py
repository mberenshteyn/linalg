from fractions import *
import numpy.polynomial.polynomial as np

class SizeError(Exception):
    pass

class Matrix:
    def __init__(self, array, aug = None):
        self.rows = [[Fraction(val) if type(val) != np.Polynomial else val for val in row] for row in array]
        self.unchanged_rows = [[Fraction(val) if type(val) != np.Polynomial else val for val in row] for row in array] #Never change this variable!
        self.aug_matrix = aug
        self.cleanup()

    @classmethod
    def identity(self, size):
        elements = [[0 for i in range(size)] for i in range(size)]
        for i in range(size):
            elements[i][i] = 1
        return Matrix(elements)

    @classmethod
    def colVector(self, array):
        elements = [[el] for el in array]
        return Matrix(elements)

    @classmethod
    def rowVector(self, array):
        elements = [array]
        return Matrix(elements)

    def __eq__(self, other):
        """
        Defines equality of matrices as having the same dimensions and consisting of the same values.

        >>> A = Matrix([[1, 2], [3, 4]])
        >>> B = Matrix([[1.0, 2.0], [3.0, 4.0]])
        >>> C = Matrix([[3, 4], [1, 2]])
        >>> A == B
        True
        >>> A == C
        False
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

    @classmethod
    def copy(self, mat):
        """
        Returns a new Matrix object with the same dimensions and values as the current Matrix instance.
        """
        return Matrix(mat.unchanged_rows)

    @classmethod
    def copy_current(self, mat):
        return Matrix(mat.rows)

    """
    Display Methods
    """

    def __str__(self):
        """
        Returns the matrix instance as represented by a list of lists (rows).
        Mostly a debugging tool, hoping it doesn't stick around to the end.
        """
        return str([[str(val) for val in row] for row in self.rows])

    def print_nice(self):
        """
        Prints the rows one-by-one, which makes output in the shell slightly nicer.
        This is mostly a debugging tool, and I'm hoping it's not actually part of the final implementation.
        Note to self: this should probably be __str__.
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
        Should only be called as part of other functions (e.g. gauss_elim and back_sub)

        >>> A = Matrix([[1.0, 0.0, -1.0], [0.5, -0.0, 3.0], [1.5, 2.0, 2.5]])
        >>> A.cleanup()
        >>> print(A)
        [['1', '0', '-1'], ['1/2', '0', '3'], ['3/2', '2', '5/2']]
        """
        if self.augmented:
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

        >>> A = Matrix([[3, 5], [1, 3], [6, 7]])
        >>> A.num_row
        3
        """
        return len(self.rows)

    @property
    def num_col(self):
        """
        Returns the number of columns in the matrix instance.

        >>> A = Matrix([[1, 3, 4, 5], [2, 6, 7, 8]])
        >>> A.num_col
        4
        """
        return len(self.rows[0])

    def _get_row(self, i):
        """
        Returns the ith row of the matrix instance as a row vector matrix.
        """
        return Matrix.rowVector(self.rows[i])

    def _get_col(self, j):
        """
        Returns the jth column of the matrix instance as a column vector matrix.
        """
        elements = [row[j] for row in self.rows]
        return Matrix.colVector(elements)

    def _get_val(self, i, j):
        """
        Returns the value at the ith row and jth column of the matrix instance.

        >>> A = Matrix([[1, 2], [3, 4]])
        >>> A._get_val(0, 1)
        Fraction(2, 1)
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

        >>> A = Matrix([[1, 2], [3, 4]])
        >>> A._scale_row(1, 2)
        >>> print(A)
        [['1', '2'], ['6', '8']]
        """
        if (self.augmented):
            self.aug_matrix._scale_row(i, scal)
        scaled_row_as_list = [scal * self._get_val(i, j) for j in range(self.num_col)]
        scaled_row = Matrix.rowVector(scaled_row_as_list)
        self._change_row(i, scaled_row)
        self.cleanup()

    def _swap_rows(self, i, j):
        """
        Interchanges the ith and jth rows of the matrix instance.
        Should only ever be called as part of another method (e.g. gauss_elim).

        >>> A = Matrix([[1, 2], [3, 4]])
        >>> A._swap_rows(0, 1)
        >>> print(A)
        [['3', '4'], ['1', '2']]
        """
        if (self.augmented):
            self.aug_matrix._swap_rows(i, j)
        row_at_i = self._get_row(i)
        row_at_j = self._get_row(j)
        self._change_row(i, row_at_j)
        self._change_row(j, row_at_i)

    def _sub_from_row(self, i, j, scal = 1):
        """
        Subtracts a scalar multiple of the jth row from the ith row of the matrix instance.

        >>> A = Matrix([[1, 3], [4, 4]])
        >>> A._sub_from_row(1, 0)
        >>> print(A)
        [['1', '3'], ['3', '1']]
        """
        if (self.augmented):
            self.aug_matrix._sub_from_row(i, j, scal)
        new_row_as_list = [0 for _ in range(self.num_col)]
        for k in range(self.num_col):
            new_row_as_list[k] = self._get_val(i, k) - scal * self._get_val(j, k)
        new_row = Matrix.rowVector(new_row_as_list)
        self._change_row(i, new_row)

    """
    Arithmetic Operations
    """

    def __add__(self, other):
        """
        Returns a new matrix produced by summing the values in two matrix instances.
        Requires that the two matrices have the same dimensions.

        >>> A = Matrix([[1, 2], [3, 4]])
        >>> B = Matrix([[3, 3], [0, 5]])
        >>> print(A + B)
        [['4', '5'], ['3', '9']]
        """
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

        >>> A = Matrix([[1, 2], [3, 4]])
        >>> B = Matrix([[3, 3], [0, 5]])
        >>> print(A - B)
        [['-2', '-1'], ['3', '-1']]
        """
        assert self.num_row == other.num_row and self.num_col == other.num_col, "Two matrices must be the same size in order for subtraction to work correctly"
        sub = Matrix([[0 for _ in range(self.num_col)] for _ in range(self.num_row)])
        for i in range(self.num_row):
            for j in range(self.num_col):
                sub._init_val(i, j, self._get_val(i, j) - other._get_val(i, j))
        return sub

    def __mul__(self, value):
        """
        Chooses the appropriate matrix multiplication method, based on whether the given value is another matrix or a number.
        If the value is of any other type, the operation raises a TypeError.

        >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> B = Matrix([[1, 4], [2, 5], [3, 6]])
        >>> print(A * B)
        [['14', '32'], ['32', '77']]
        >>> print(A * 3)
        [['3', '6', '9'], ['12', '15', '18']]
        """
        if isinstance(value, Matrix):
            return self._matmul(value)
        elif type(value) == int or type(value) == float or type(value) == np.Polynomial:
            return self._scmul(value)
        raise TypeError(f"Cannot multiply matrices by objects of type {type(value)}")

    def _matmul(self, other):
        """
        Returns a new matrix of dimensions (self.num_row) x (other.num_col) produced by multiplying the values in two matrix instances.
        Requires that self.num_col == other.num_row.

        >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> B = Matrix([[1, 4], [2, 5], [3, 6]])
        >>> print(A._matmul(B))
        [['14', '32'], ['32', '77']]
        """
        def reduce(a, b):
            """
            Returns the sum of the products of corresponding elements in row vector a and column vector b.

            >>> a = Matrix.rowVector([1, 1])
            >>> b = Matrix.colVector([2, 2])
            >>> reduce(a, b)
            4
            >>> c = Matrix.rowVector([1, 2, 3])
            >>> d = Matrix.colVector([1, 5, 3])
            >>> reduce(c, d)
            20
            """
            assert a.num_col == b.num_row, "Both vectors should be of equal length"
            tot = 0
            for i in range(a.num_col):
                product = a._get_val(0, i) * b._get_val(i, 0)
                tot += product
            return tot
        assert self.num_col == other.num_row, "The number of columns in the first matrix must equal the number of rows in the second matrix for multiplication to work correctly"
        product = Matrix([[0 for _ in range(other.num_col)] for _ in range(self.num_row)])
        for i in range(product.num_row):
            for j in range(product.num_col):
                product._init_val(i, j, reduce(self._get_row(i), other._get_col(j)))
        product.cleanup()
        return product

    def _scmul(self, scal):
        """
        Returns a new matrix produced by multiplying every value in the matrix instance by scal.

        >>> A = Matrix([[1, 1], [1, 1]])
        >>> print(A._scmul(3))
        [['3', '3'], ['3', '3']]
        >>> print(A._scmul(0))
        [['0', '0'], ['0', '0']]
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

    def transpose(self):
        """
        Returns the tranpose of the matrix instance, where the columns of the original matrix are the rows of the new matrix.

        >>> A = Matrix([[1, 2], [3, 4]])
        >>> print(A.transpose())
        [['1', '3'], ['2', '4']]
        >>> B = Matrix([[0, 1, 2], [3, 4, 5]])
        >>> print(B.transpose())
        [['0', '3'], ['1', '4'], ['2', '5']]
        """
        new_array = [[0 for _ in range(self.num_row)] for _ in range(self.num_col)]
        for i in range(self.num_row):
            for j in range(self.num_col):
                new_array[j][i] = self._get_val(i, j)
        return Matrix(new_array)

    def _augment(self, aug_mat):
        """
        Augments a matrix to the right of the existing matrix instance.

        >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> v = Matrix([[4], [3], [2]])
        >>> A._augment(v)
        >>> print(A.aug_matrix)
        [['4'], ['3'], ['2']]
        """
        assert isinstance(aug_mat, Matrix) and aug_mat.num_row == self.num_row, f"The augmented column must have {self.num_row} rows"
        assert self.aug_matrix == None, f"The matrix instance is already augmented"
        self.aug_matrix = aug_mat

    def _deaugment(self):
        """
        Clears the augmented matrix if it exists, otherwise does nothing.

        >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> v = Matrix([[4], [3], [2]])
        >>> A._augment(v)
        >>> A._deaugment()
        >>> print(A.aug_matrix == None)
        True
        """
        self.aug_matrix = None

    @property
    def inverse(self):
        """
        Returns the inverse of the matrix instance, which is defined as the matrix which results in
        an identity matrix when either left-multiplied or right-muliplied by the matrix instance.

        >>> A = Matrix([[1, 4, 2], [2, 0, 1], [3, 5, 2]])
        >>> print(A.inverse)
        [['-5/11', '2/11', '4/11'], ['-1/11', '-4/11', '3/11'], ['10/11', '7/11', '-8/11']]
        >>> print(A * A.inverse)
        [['1', '0', '0'], ['0', '1', '0'], ['0', '0', '1']]
        >>> print(A.inverse * A)
        [['1', '0', '0'], ['0', '1', '0'], ['0', '0', '1']]
        """

        if not self.square:
            raise SizeError("Non-square matrices do not have an inverse")
        elif self.determinant == 0:
            return None

        mat_copy = Matrix.copy_current(self)
        mat_copy._augment(Matrix.identity(self.num_row))

        mat_copy.gauss_elim()
        mat_copy.back_sub()
        return mat_copy.aug_matrix

    """
    Check Methods
    """

    @property
    def augmented(self):
        """
        Returns True if the matrix instance is augmented and False otherwise.
        """
        return (self.aug_matrix is not None)

    @property
    def square(self):
        """
        Returns True if the matrix instance has an equal number of rows and columns
        (i.e. it is square) and False otherwise.

        >>> A = Matrix([[1]])
        >>> B = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> C = Matrix([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        >>> A.square
        True
        >>> B.square
        False
        >>> C.square
        True
        """
        return self.num_row == self.num_col

    @property
    def is_identity(self):
        """
        Returns True if the matrix instance is an identity matrix (a square matrix with
        1s along the main diagonal and 0s for all other values.) and False otherwise.

        >>> A = Matrix([[1, 0], [0, 1]])
        >>> B = Matrix([[1, 0, 0], [0, 1, 0]])
        >>> C = Matrix([[1, 0], [0, 4]])
        >>> A.is_identity
        True
        >>> B.is_identity
        False
        >>> C.is_identity
        False
        """
        if not self.square:
            return False
        for i in range(self.num_row):
            for j in range(self.num_col):
                if i == j and self._get_val(i, j) != 1:
                    return False
                if i != j and self._get_val(i, j) != 0:
                    return False
        return True

    @property
    def diagonal(self):
        """
        Returns True if the matrix instance has nonzero values only along its main diagonal.

        >>> A = Matrix([[1, 0, 0], [0, 3, 0], [0, 0, 4]])
        >>> B = Matrix([[2, 0, 0], [0, 4, 0]])
        >>> C = Matrix([[1, 1], [0, 2]])
        >>> A.diagonal
        True
        >>> B.diagonal
        True
        >>> C.diagonal
        False
        """
        for i in range(self.num_row):
            for j in range(self.num_col):
                if i != j and self._get_val(i, j) != 0:
                    return False
        return True

    @property
    def check_ref(self):
        """
        Returns True if the matrix instance is in row echelon form, and False otherwise.
        If the return value is True, the matrix instance meets the following conditions:
            1) All nonzero rows are above any rows of all zeros.
            2) Each leading entry of a row is in a column right of the leading entry in the above row.
            3) All entries in a column before a leading entry are zeros.

        >>> A = Matrix([[3, 1, 4], [0, 5, 2], [0, 0, 2]])
        >>> B = Matrix([[0, 1], [0, 0]])
        >>> C = Matrix([[0, 3, 3], [2, 1, 4]])
        >>> A.check_ref
        True
        >>> B.check_ref
        True
        >>> C.check_ref
        False
        """
        eval_matrix = Matrix(self.unchanged_rows)
        eval_matrix.gauss_elim()
        return self == eval_matrix

    @property
    def check_rref(self):
        """
        Returns True if the matrix instance is in reduced row echelon form, and False otherwise.
        If the return value is True, the matrix instance meets conditions of row echelon form as well as the following:
            4) The leading entry in each nonzero row is 1.
            5) Each leading 1 is the only entry in its column.

        >>> A = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> B = Matrix([[2, 0], [0, 2]])
        >>> A.check_rref
        True
        >>> B.check_rref
        False
        """
        eval_matrix = Matrix(self.unchanged_rows)
        eval_matrix.gauss_elim()
        eval_matrix.back_sub()
        return self == eval_matrix

    @property
    def upper_tri(self):
        """
        Returns True if all the values below and to the left of the main diagonal in the matrix instance are 0 and False otherwise.

        >>> A = Matrix([[1, 3, 5], [0, 3, 0], [0, 0, 2]])
        >>> B = Matrix([[0, 0], [0, 0]])
        >>> C = Matrix([[0, 2], [2, 1]])
        >>> A.upper_tri
        True
        >>> B.upper_tri
        True
        >>> C.upper_tri
        False
        """
        for i in range(self.num_row):
            for j in range(self.num_col):
                if j < i and self._get_val(i, j) != 0:
                    return False
        return True

    @property
    def lower_tri(self):
        """
        Returns True if all the values above and to the right of the main diagonal in the matrix instance are 0 and False otherwise.

        >>> A = Matrix([[1, 0, 0], [2, 3, 0], [4, 3, 3]])
        >>> B = Matrix([[0, 0], [0, 0]])
        >>> C = Matrix([[1, 2], [0, 1]])
        >>> A.lower_tri
        True
        >>> B.lower_tri
        True
        >>> C.lower_tri
        False
        """
        for i in range(self.num_row):
            for j in range(self.num_col):
                if j > i and self._get_val(i, j) != 0:
                    return False
        return True

    @property
    def symmetric(self):
        """
        Returns True if the matrix instance is symmetrical (it equals its tranpose) and False otherwise.

        >>> A = Matrix([[1, 2], [2, 1]])
        >>> B = Matrix([[1, 1], [3, 3]])
        >>> A.symmetric
        True
        >>> B.symmetric
        False
        """
        return self == self.transpose()

    """
    Row Reduction
    """

    def gauss_elim(self):
        """
        Takes the matrix instance and goes through the process of Gaussian elimination to put it in row echelon form.
        The original form of the matrix is preserved in the instance variable self.unchanged_rows.

        >>> A = Matrix([[1, 1], [-1, -1]])
        >>> A.gauss_elim()
        >>> print(A)
        [['1', '1'], ['0', '0']]
        >>> B = Matrix([[1, 2, 2], [1, 1, 1], [0, -1, -1]])
        >>> B.gauss_elim()
        >>> print(B)
        [['1', '2', '2'], ['0', '-1', '-1'], ['0', '0', '0']]
        """
        current_row = 0
        current_col = 0
        while (current_row < self.num_row) and (current_col < self.num_col):
            i_max = max(range(current_row, self.num_row), key = lambda row: abs(self._get_val(row, current_col)))
            if self._get_val(i_max, current_col) == 0:
                current_col += 1
            else:
                self._swap_rows(current_row, i_max)
                for i in range(current_row + 1, self.num_row):
                    f = self._get_val(i, current_col) / self._get_val(current_row, current_col)
                    self._sub_from_row(i, current_row, f)
                current_row += 1
                current_col += 1
        self.cleanup()

    def back_sub(self):
        """
        Takes the matrix instance in row echelon form and goes through the process of back substitution to put it in reduced row echelon form.
        The original (non-REF) form of the matrix is preserved in instance variable self.unchanged_rows.

        >>> A = Matrix([[2, 1, 3, 4], [0, 0, 1, 4], [4, 2, 6, 8], [6, 3, 14, 35]])
        >>> A.gauss_elim()
        >>> A.back_sub()
        >>> print(A)
        [['1', '1/2', '0', '0'], ['0', '0', '1', '0'], ['0', '0', '0', '1'], ['0', '0', '0', '0']]
        >>> B = Matrix([[1, 2, 2], [1, 1, 1], [0, -1, -1]])
        >>> B.gauss_elim()
        >>> B.back_sub()
        >>> print(B)
        [['1', '0', '0'], ['0', '1', '1'], ['0', '0', '0']]
        """
        assert self.check_ref, "The matrix instance must be in row echelon form before starting back substitution"
        for index in range(self.num_row - 1, 0, -1):
            pivot, pivot_index = 1, 0
            while pivot_index < self.num_col:
                if self._get_val(index, pivot_index) != 0:
                    pivot = self._get_val(index, pivot_index)
                    break
                pivot_index += 1
            if pivot_index != self.num_col:
                for scaled_row_index in range(index - 1, -1, -1):
                    scale = (self._get_val(scaled_row_index, pivot_index) / pivot)
                    self._sub_from_row(scaled_row_index, index, scale)
        for i in range(self.num_row):
            new_pivot_index = 0
            while new_pivot_index < self.num_col:
                if self._get_val(i, new_pivot_index) != 0:
                    break
                new_pivot_index += 1
            if new_pivot_index != self.num_col:
                self._scale_row(i, 1 / self._get_val(i, new_pivot_index))
        self.cleanup()
        self.rref = True

    """
    Fundamental Matrix Spaces
    """

    def _pivot_cols(self):
        """
        Returns a list of pivot columns (by index) in the matrix object.
        """
        copy = Matrix.copy(self)
        copy.gauss_elim()
        copy.back_sub()
        pivot_cols = []
        for i in range(copy.num_row):
            for j in range(copy.num_col):
                if copy._get_val(i, j) == 1:
                    pivot_cols.append(j)
                    break
        return pivot_cols

    def col_space(self):
        """
        Calculates the column space (AKA image) of a matrix A, defined as all linear combinations of the columns of the matrix.
        This is done by finding the pivot columns of the matrix.
        The output is formatted as a set of "vectors" (1-column matrices) which is the basis of the column space.
        """
        pivot_indices = self._pivot_cols()
        basis = []
        for index in pivot_indices:
            basis.append(self._get_col(index))
        return basis

    def row_space(self):
        """
        Calculates the row space of a matrix A, which is equivalent to the column space of its transpose.
        """
        return self.transpose().col_space()

    def null_space(self):
        """
        Calculates the null space (AKA kernel) of a matrix A, defined as the set of all vectors for which Av = 0.
        The output is formatted as a set of "vectors" (1-column matrices) which is the basis of the null space.
        """
        # Fact 1: the dimension of the null space = num_col - num_pivot_row
        # Fact 2: the elements of a vector in the null space are the negative of the associated column's coefficients
        #         and 1 for the associated free variable
        # Fact 3: free variables are associated with non-pivot columns (indices not in self.pivot_cols())


        pass

    def lnull_space(self):
        """
        Calculates the left null space of a matrix A, which is equivalent to the null space of its transpose.
        """
        return self.transpose().null_space()

    """
    Determinant
    """

    def _strip_row(self, index):
        self.rows.pop(index)

    def _strip_col(self, index):
        for row in self.rows:
            row.pop(index)

    def _count_zeros(self):
        count = 0
        for i in range(self.num_row):
            for j in range(self.num_col):
                if self._get_val(i, j) == 0:
                    count += 1
        return count

    def _optimal_axis_calc(self):
        optimal_row = (0, 0, "row")
        optimal_col = (0, 0, "col")
        for i in range(self.num_row):
            row_zeros = self._get_row(i)._count_zeros()
            col_zeros = self._get_col(i)._count_zeros()
            if row_zeros > optimal_row[1]:
                optimal_row = (i, row_zeros, "row")
            if col_zeros > optimal_col[1]:
                optimal_col = (i, row_zeros, "col")
        return max(optimal_row, optimal_col, key = lambda l: l[1])

    @property
    def determinant(self):
        if not self.square:
            raise SizeError("Non-square matrices do not have calculable determinants")
        elif self.num_row == 1 and self.num_col == 1:
            return self._get_val(0, 0)
        elif self.num_row == 2 and self.num_col == 2:
            return (self._get_val(0, 0) * self._get_val(1, 1)) - (self._get_val(0, 1) * self._get_val(1, 0))
        elif self.upper_tri or self.lower_tri:
            det = 1
            for i in range(self.num_row):
                det *= self._get_val(i, i)
            return det
        else:
            start_axis = self._optimal_axis_calc()
            axis_index, axis_type = start_axis[0], start_axis[2]
            matrix_copy = Matrix.copy_current(self)
            det = 0
            if axis_type == "row":
                matrix_copy._strip_row(axis_index)
                for j in range(self.num_col):
                    multiplier = self._get_val(axis_index, j)
                    if multiplier != 0:
                        matrix_copy_2 = Matrix.copy_current(matrix_copy)
                        matrix_copy_2._strip_col(j)
                        if (axis_index + j) % 2 == 1:
                            multiplier *= -1
                        det += multiplier * matrix_copy_2.determinant
            elif axis_type == "col":
                matrix_copy._strip_col(axis_index)
                for i in range(self.num_row):
                    multiplier = self._get_val(i, axis_index)
                    if multiplier != 0:
                        matrix_copy_2 = Matrix.copy_current(matrix_copy)
                        matrix_copy_2._strip_row(i)
                        if (i + axis_index) % 2 == 1:
                            multiplier *= -1
                        det += multiplier * matrix_copy_2.determinant
            return det

    """
    Eigenvalues and Eigenvectors
    """

    @property
    def eigenvalues(self):
        if not self.square:
            raise SizeError("Matrix must be square in order to calculate eigenvalues")
        subtract_matrix = Matrix.identity(self.num_row) * np.Polynomial([0, 1])
        manipulated_matrix = self - subtract_matrix
        char_eq = manipulated_matrix.determinant
        coefficients = [coef for coef in char_eq]
        return np.polyroots(coefficients)

    def eigenvectors(self):
        # Iterate through process with each eigenvalue
        # Find null space for A - lambda I
        # Combine to construct basis for eigenspace
        pass

"""
Solving Linear Systems
"""

variables = ['a', 'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def sign(num):
    """
    A function that returns the string sign of a number (+ if positive, - if negative).
    """
    if num < 0:
        return "-"
    elif num > 0:
        return "+"

def polynomial_format(poly):
    """
    A function to convert polynomials into a more readable string format.
    """
    format = ""
    exp = 0
    for coef in poly:
        if coef == 0:
            pass
        elif exp == 0:
            format += f"{coef} "
        else:
            if format != "":
                format += sign(coef) + " "
            else:
                if sign(coef) == "-":
                    format = "-"
            if exp == 1:
                format += f"{abs(coef)}λ "
            else:
                format += f"{abs(coef)}λ^{exp}"
        exp += 1
    return format.rstrip()

def format_as_linear_sys(A):
    """
    Prints an augmented matrix A as a linear system. Mostly for debugging/early versions, so this is a lazy function description.
    """
    for i in range(A.num_row):
        if [0 for j in range(A.num_col)] == A._get_row(i):
            continue
        row_string = ""
        for j in range(A.num_col):
            coef = A._get_val(i, j)
            if j < A.num_col - 1:
                if coef == 0:
                    continue
                sign = '+ '
                if coef < 0:
                    if row_string == '':
                        sign = '-'
                    else:
                        sign = '- '
                elif row_string == '':
                    sign = ''
                if abs(coef) == 1:
                    coef_expr = ''
                else:
                    coef_expr = str(abs(coef))
                row_string += (sign + coef_expr + variables[j] + " ")
            else:
                if row_string == '':
                    row_string += "0 "
                row_string += ("= " + str(coef))
        print(row_string)

def solve_linear_sys(A, v = 0):
    """
    Solves a linear system where A is the coefficient matrix and v is the solution vector. If no v is provided, finds the solution to the homogeneous solution (if it exists).
    """
    if v == 0:
        v = [0 for _ in range(A.num_row)]
    aug_matrix = Matrix.copy(A)
    aug_matrix._augment(v)
    print("Given a linear system")
    format_as_linear_sys(aug_matrix)
    aug_matrix.gauss_elim()
    aug_matrix.back_sub()
    print()
    print("We find")
    format_as_linear_sys(aug_matrix)
    check_consistency(aug_matrix)

def check_consistency(aug_matrix):
    """
    Runs through a solved augmented matrix aug_matrix and checks that the corresponding linear system is consistent.
    If inconsistency is found (e.g. 0 = 1), it prints (but in the future, should return) that the system is inconsistent.
    """
    inconsistent_row = False
    for i in range(aug_matrix.num_row):
        nonzero_coef = False
        for j in range(aug_matrix.num_col):
            if j < aug_matrix.num_col - 1:
                if aug_matrix._get_val(i, j) != 0:
                    nonzero_coef = True
            if j == aug_matrix.num_col - 1:
                if aug_matrix._get_val(i, j) == 0:
                    continue
                else:
                    if nonzero_coef:
                        continue
                    else:
                        inconsistent_row = True
                        break
    if inconsistent_row:
        print("This system is inconsistent and has no solution.")
