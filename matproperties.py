from matrix import Matrix

class MatProperties:
        @staticmethod
        def augmented(A: Matrix) -> bool:
            """
            Returns True if the matrix instance is augmented and False otherwise.
            """
            return (A.aug_matrix is not None)

        @staticmethod
        def square(A: Matrix) -> bool:
            """
            Returns True if the matrix instance has an equal number of rows and columns
            (i.e. it is square) and False otherwise.
            """
            return A.num_row == A.num_col

        @staticmethod
        def is_identity(A: Matrix) -> bool:
            """
            Returns True if the matrix instance is an identity matrix (a square matrix with
            1s along the main diagonal and 0s for all other values.) and False otherwise.
            """
            if not MatProperties.square(A):
                return False
            for i in range(A.num_row):
                for j in range(A.num_col):
                    if i == j and A._get_val(i, j) != 1:
                        return False
                    if i != j and A._get_val(i, j) != 0:
                        return False
            return True

        @staticmethod
        def diagonal(A: Matrix) -> bool:
            """
            Returns True if the matrix instance has nonzero values only along its main diagonal.
            """
            for i in range(A.num_row):
                for j in range(A.num_col):
                    if i != j and A._get_val(i, j) != 0:
                        return False
            return True

        @staticmethod
        def upper_tri(A: Matrix) -> bool:
            """
            Returns True if all the values below and to the left of the main diagonal in the matrix instance are 0 and False otherwise.
            """
            for i in range(A.num_row):
                for j in range(A.num_col):
                    if j < i and A._get_val(i, j) != 0:
                        return False
            return True

        @staticmethod
        def lower_tri(A: Matrix) -> bool:
            """
            Returns True if all the values above and to the right of the main diagonal in the matrix instance are 0 and False otherwise.
            """
            for i in range(A.num_row):
                for j in range(A.num_col):
                    if j > i and A._get_val(i, j) != 0:
                        return False
            return True

        @staticmethod
        def symmetric(A: Matrix) -> bool:
            """
            Returns True if the matrix instance is symmetrical (it equals its tranpose) and False otherwise.
            """
            for i in range(A.num_row):
                for j in range(A.num_col):
                    if A._get_val(i, j) != A._get_val(j, i):
                        return False
            return True
