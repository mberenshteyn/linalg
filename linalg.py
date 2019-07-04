from matrix import Matrix
from matproperties import MatProperties
from typing import List, Tuple

from numpy import iscomplex
import numpy.polynomial.polynomial as np

class LinAlg:

    @staticmethod
    def transpose(A: Matrix) -> Matrix:
        """
        Returns the tranpose of matrix A, where the columns of the
        original matrix are the rows of the new matrix.
        """
        new_array = [[0 for _ in range(A.num_row)] for _ in range(A.num_col)]
        for i in range(A.num_row):
            for j in range(A.num_col):
                new_array[j][i] = A._get_val(i, j)
        return Matrix(new_array)

    @staticmethod
    def inverse(A: Matrix) -> Matrix:
        """
        Returns the inverse of matrix A, defined as the matrix which results in an identity
        matrix when either left-multiplied or right-muliplied by the matrix instance.
        """
        if not MatProperties.square(A):
            raise ValueError("Non-square matrices do not have an inverse")
        mat_copy = Matrix.copy_current(A)
        mat_copy._augment(Matrix.identity(A.num_row))
        LinAlg.gauss_elim(mat_copy)
        LinAlg.back_sub(mat_copy)
        return mat_copy.aug_matrix

    @staticmethod
    def _count_zeros(A: Matrix) -> int:
        """
        Returns the number of elements equal to 0 in matrix A.
        """
        count = 0
        for i in range(A.num_row):
            for j in range(A.num_col):
                if A._get_val(i, j) == 0:
                    count += 1
        return count

    @staticmethod
    def _optimal_axis_calc(A: Matrix) -> int:
        """
        Uses the number of zeros in each row and column of matrix A return
        the row or column index to optimize calculation of the determinant.
        """
        optimal_row = (0, 0, "row")
        optimal_col = (0, 0, "col")
        for i in range(A.num_row):
            row_zeros = LinAlg._count_zeros(A._get_row(i))
            col_zeros = LinAlg._count_zeros(A._get_col(i))
            if row_zeros > optimal_row[1]:
                optimal_row = (i, row_zeros, "row")
            if col_zeros > optimal_col[1]:
                optimal_col = (i, row_zeros, "col")
        return max(optimal_row, optimal_col, key = lambda l: l[1])

    @staticmethod
    def determinant(A: Matrix) -> int:
        """
        Returns the determinant of matrix A.
        """
        if not MatProperties.square(A):
            raise ValueError("Non-square matrices do not have calculable determinants")
        elif A.num_row == 1 and A.num_col == 1:
            return A._get_val(0, 0)
        elif A.num_row == 2 and A.num_col == 2:
            return (A._get_val(0, 0) * A._get_val(1, 1)) - (A._get_val(0, 1) * A._get_val(1, 0))
        elif MatProperties.upper_tri(A) or MatProperties.lower_tri(A):
            det = 1
            for i in range(A.num_row):
                det *= A._get_val(i, i)
            return det
        else:
            start_axis = LinAlg._optimal_axis_calc(A)
            axis_index, axis_type = start_axis[0], start_axis[2]
            matrix_copy = Matrix.copy_current(A)
            det = 0
            if axis_type == "row":
                matrix_copy._del_row(axis_index)
                for j in range(A.num_col):
                    multiplier = A._get_val(axis_index, j)
                    if multiplier != 0:
                        matrix_copy_2 = Matrix.copy_current(matrix_copy)
                        matrix_copy_2._del_col(j)
                        if (axis_index + j) % 2 == 1:
                            multiplier *= -1.
                        det += multiplier * LinAlg.determinant(matrix_copy_2)
            elif axis_type == "col":
                matrix_copy._del_col(axis_index)
                for i in range(A.num_row):
                    multiplier = A._get_val(i, axis_index)
                    if multiplier != 0:
                        matrix_copy_2 = Matrix.copy_current(matrix_copy)
                        matrix_copy_2._del_row(i)
                        if (i + axis_index) % 2 == 1:
                            multiplier *= -1.
                        det += multiplier * LinAlg.determinant(matrix_copy_2)
            return det

    """
    Row Reduction
    """

    @staticmethod
    def gauss_elim(A: Matrix) -> None:
        """
        Takes the matrix instance and goes through the process of Gaussian elimination to put it in row echelon form.
        The original form of the matrix is preserved in the instance variable unchanged_rows.
        """
        current_row = 0
        current_col = 0
        while (current_row < A.num_row) and (current_col < A.num_col):
            i_max = max(range(current_row, A.num_row), key = lambda row: abs(A._get_val(row, current_col)))
            if A._get_val(i_max, current_col) == 0:
                current_col += 1
            else:
                A._swap_rows(current_row, i_max)
                for i in range(current_row + 1, A.num_row):
                    factor = A._get_val(i, current_col) / A._get_val(current_row, current_col)
                    A._sub_from_row(i, current_row, factor)
                current_row += 1
                current_col += 1
        A.cleanup()

    @staticmethod
    def back_sub(A: Matrix) -> None:
        """
        Takes the matrix instance in row echelon form and goes through the process of back substitution to put it in reduced row echelon form.
        The original (non-REF) form of the matrix is preserved in instance variable unchanged_rows.
        """
        for index in range(A.num_row - 1, 0, -1):
            pivot, pivot_index = 1, 0
            while pivot_index < A.num_col:
                if A._get_val(index, pivot_index) != 0:
                    pivot = A._get_val(index, pivot_index)
                    break
                pivot_index += 1
            if pivot_index != A.num_col:
                for scaled_row_index in range(index - 1, -1, -1):
                    scale = (A._get_val(scaled_row_index, pivot_index) / pivot)
                    A._sub_from_row(scaled_row_index, index, scale)
        for i in range(A.num_row):
            new_pivot_index = 0
            while new_pivot_index < A.num_col:
                if A._get_val(i, new_pivot_index) != 0:
                    break
                new_pivot_index += 1
            if new_pivot_index != A.num_col:
                A._scale_row(i, 1 / A._get_val(i, new_pivot_index))
        A.cleanup()

    """
    Fundamental Matrix Spaces
    """

    @staticmethod
    def _pivot_cols(A: Matrix) -> List[int]:
        """
        Returns a list of pivot columns (by index) in the matrix object.
        """
        copy = Matrix.copy(A)
        LinAlg.gauss_elim(copy)
        LinAlg.back_sub(copy)
        pivot_cols = []
        for i in range(copy.num_row):
            for j in range(copy.num_col):
                if copy._get_val(i, j) == 1:
                    pivot_cols.append(j)
                    break
        return pivot_cols

    @staticmethod
    def col_space(A: Matrix) -> List[Matrix]:
        """
        Calculates the basis of the column space of a matrix A,
        defined as all linear combinations of the columns of the matrix.
        """
        pivot_indices = A._pivot_cols()
        basis = []
        for index in pivot_indices:
            basis.append(A._get_col(index))
        return basis

    @staticmethod
    def row_space(A: Matrix) -> List[Matrix]:
        """
        Calculates the basis of the row space of a matrix A,
        which is equivalent to the column space of its transpose.
        """
        return LinAlg.col_space(LinAlg.transpose(A))

    @staticmethod
    def null_space(A: Matrix) -> List[Matrix]:
        """
        Calculates the basis of the null space of a matrix A,
        defined as the set of all vectors for which Av = 0.
        """
        calc_matrix = LinAlg.transpose(Matrix.copy_current(A))
        calc_matrix._augment(Matrix.identity(calc_matrix.num_row))
        LinAlg.gauss_elim(calc_matrix)
        LinAlg.back_sub(calc_matrix)

        basis = []
        zero_row = Matrix.row_vector([0 for _ in range(calc_matrix.num_col)])
        for i in range(calc_matrix.num_row):
            current = calc_matrix._get_row(i)
            if current == zero_row:
                corr_row = calc_matrix.aug_matrix._get_row(i)
                basis.append(LinAlg.transpose(corr_row))
        return basis

    @staticmethod
    def lnull_space(A: Matrix) -> List[Matrix]:
        """
        Calculates the basis of the left null space of a matrix A,
        which is equivalent to the null space of its transpose.
        """
        return LinAlg.null_space(LinAlg.transpose(A))

    """
    Eigenvalues and Eigenvectors
    """

    @staticmethod
    def check_eigenvalue(A: Matrix, value) -> bool:
        if not MatProperties.square(A):
            raise ValueError("Matrix must be square in order to have eigenvalues")
        subtract_matrix = Matrix.identity(A.num_row) * value
        manipulated_matrix = A - subtract_matrix
        if LinAlg.determinant(manipulated_matrix) == 0:
            return True
        return False

    @staticmethod
    def eigenvalues(A: Matrix) -> List:
        if not MatProperties.square(A):
            raise ValueError("Matrix must be square in order to calculate eigenvalues")
        subtract_matrix = Matrix.identity(A.num_row) * np.Polynomial([0., 1.])
        manipulated_matrix = A - subtract_matrix
        char_eq = LinAlg.determinant(manipulated_matrix)
        coefficients = [coef for coef in char_eq]
        raw_roots = np.polyroots(coefficients)
        true_roots = []
        for root in raw_roots:
            rounded = round(root, 3)
            if (iscomplex(rounded)):
                true_roots.append(root)
                break
            else:
                rounded = rounded.real
                if LinAlg.check_eigenvalue(A, float(rounded)):
                    true_roots.append(rounded)
                else:
                    true_roots.append(root)
        return true_roots

    @staticmethod
    def eigenvectors(A: Matrix) -> List[Matrix]:
        if not MatProperties.square(A):
            raise ValueError("Matrix must be square in order to calculate eigenvectors")
        eigenbasis = []
        identity = Matrix.identity(A.num_row)
        for ev in LinAlg.eigenvalues(A):
            current = A - (identity * ev.item())
            eigenbasis.extend(LinAlg.null_space(current))
        return eigenbasis

    @staticmethod
    def diagonalize(A: Matrix) -> Tuple[Matrix]:
        if not MatProperties.square(A):
            raise ValueError("Matrix must be square in order to diagonalize")
        evalues = LinAlg.eigenvalues(A)
        evectors = LinAlg.eigenvectors(A)
        if len(evalues) != A.num_row or len(evectors) != A.num_row:
            raise ValueError(f"Matrix must have {A.num_row} eigenvalues \
            and eigenvectors to be diagonalized")
        pre_matrix = Matrix.merge_cols(evectors)
        post_matrix = LinAlg.inverse(pre_matrix)
        diag_matrix = Matrix.diagonal(evalues)
        return pre_matrix, diag_matrix, post_matrix
