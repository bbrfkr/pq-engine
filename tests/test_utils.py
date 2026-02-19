import pytest
import numpy as np
from pq_engine.utils import (
    check_square,
    check_hermite,
    check_unitary,
    check_one_trace,
    check_density,
)
from pq_engine.exceptions import (
    NotSquareError,
    NotHermiteError,
    NotUnitaryError,
    NonOneTraceError,
)


class TestCheckSquare:
    def test_square_matrix_passes(self):
        """正方形行列は通過するべき"""
        matrix = np.array([[1, 2], [3, 4]], dtype=np.complex64)
        check_square(matrix)  # 例外が発生しないことを確認

    def test_non_square_matrix_raises_error(self):
        """非正方形行列は例外を発生させるべき"""
        matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.complex64)
        with pytest.raises(NotSquareError):
            check_square(matrix)

    def test_single_element_matrix_passes(self):
        """1x1行列は正方形として通過するべき"""
        matrix = np.array([[5]], dtype=np.complex64)
        check_square(matrix)

    def test_row_vector_raises_error(self):
        """行ベクトルは例外を発生させるべき"""
        matrix = np.array([1, 2, 3], dtype=np.complex64)
        with pytest.raises(IndexError):
            check_square(matrix)

    def test_column_vector_raises_error(self):
        """列ベクトルは例外を発生させるべき"""
        matrix = np.array([[1], [2], [3]], dtype=np.complex64)
        with pytest.raises(NotSquareError):
            check_square(matrix)


class TestCheckHermite:
    def test_hermite_matrix_passes(self):
        """エルミート行列は通過するべき"""
        matrix = np.array([[1, 2 - 1j], [2 + 1j, 3]], dtype=np.complex64)
        check_hermite(matrix)

    def test_non_hermite_matrix_raises_error(self):
        """非エルミート行列は例外を発生させるべき"""
        matrix = np.array([[1, 2 + 1j], [3 + 1j, 4]], dtype=np.complex64)
        with pytest.raises(NotHermiteError):
            check_hermite(matrix)

    def test_real_symmetric_matrix_passes(self):
        """実対称行列はエルミートとして通過するべき"""
        matrix = np.array([[1, 2], [2, 3]], dtype=np.complex64)
        check_hermite(matrix)

    def test_non_square_matrix_indirect_raises_error(self):
        """check_squareを介して非正方形行列は例外を発生させるべき"""
        matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.complex64)
        with pytest.raises(NotSquareError):
            check_hermite(matrix)


class TestCheckUnitary:
    def test_identity_matrix_passes(self):
        """単位行列はユニタリとして通過するべき"""
        matrix = np.identity(2, dtype=np.complex64)
        check_unitary(matrix)

    def test_hadamard_matrix_passes(self):
        """アダマール行列はユニタリとして通過するべき"""
        matrix = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
        check_unitary(matrix)

    def test_pauli_x_matrix_passes(self):
        """パウリX行列はユニタリとして通過するべき"""
        matrix = np.array([[0, 1], [1, 0]], dtype=np.complex64)
        check_unitary(matrix)

    def test_non_unitary_matrix_raises_error(self):
        """非ユニタリ行列は例外を発生させるべき"""
        matrix = np.array([[1, 2], [3, 4]], dtype=np.complex64)
        with pytest.raises(NotUnitaryError):
            check_unitary(matrix)

    def test_non_square_matrix_indirect_raises_error(self):
        """check_squareを介して非正方形行列は例外を発生させるべき"""
        matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.complex64)
        with pytest.raises(NotSquareError):
            check_unitary(matrix)

    def test_non_unitary_norm_matrix_raises_error(self):
        """ノルムが1でない行列は例外を発生させるべき"""
        matrix = np.array([[2, 0], [0, 2]], dtype=np.complex64)
        with pytest.raises(NotUnitaryError):
            check_unitary(matrix)


class TestCheckOneTrace:
    def test_matrix_with_one_trace_passes(self):
        """トレースが1の行列は通過するべき"""
        matrix = np.array([[1, 0], [0, 0]], dtype=np.complex64)
        check_one_trace(matrix)

    def test_matrix_trace_close_to_one_passes(self):
        """トレースが1に近い行列は通過するべき（atolの範囲内）"""
        matrix = np.array([[1.00001, 0], [0, -0.00001]], dtype=np.complex64)
        check_one_trace(matrix)  # atol=1e-5なので通過すべき

    def test_matrix_with_trace_zero_raises_error(self):
        """トレースが0の行列は例外を発生させるべき"""
        matrix = np.array([[0, 0], [0, 0]], dtype=np.complex64)
        with pytest.raises(NonOneTraceError):
            check_one_trace(matrix)

    def test_matrix_with_trace_two_raises_error_verbose(self):
        """トレースが2の行列は例外を発生させるべき（明示的）"""
        matrix = np.array([[2, 0], [0, 0]], dtype=np.complex64)
        with pytest.raises(NonOneTraceError):
            check_one_trace(matrix)

    def test_non_square_matrix_indirect_raises_error(self):
        """check_squareを介して非正方形行列は例外を発生させるべき"""
        matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.complex64)
        with pytest.raises(NotSquareError):
            check_one_trace(matrix)


class TestCheckDensity:
    def test_valid_density_matrix_passes(self):
        """有効な密度行列は通過するべき"""
        matrix = np.array([[0.5, 0], [0, 0.5]], dtype=np.complex64)
        check_density(matrix)

    def test_pure_state_density_matrix_passes(self):
        """純粋状態の密度行列は通過するべき"""
        psi = np.array([1 / np.sqrt(2), 1j / np.sqrt(2)], dtype=np.complex64)
        matrix = np.outer(psi, np.conj(psi))
        check_density(matrix)

    def test_non_hermite_density_matrix_raises_error(self):
        """非エルミートの密度行列は例外を発生させるべき"""
        matrix = np.array([[0.5, 1j], [0, 0.5]], dtype=np.complex64)
        with pytest.raises(NotHermiteError):
            check_density(matrix)

    def test_non_one_trace_density_matrix_raises_error(self):
        """トレースが1でない密度行列は例外を発生させるべき"""
        matrix = np.array([[2, 0], [0, 0]], dtype=np.complex64)
        with pytest.raises(NonOneTraceError):
            check_density(matrix)

    def test_valid_3x3_density_matrix_passes(self):
        """3x3の有効な密度行列は通過するべき"""
        matrix = np.array(
            [[0.4, 0.1, 0.05], [0.1, 0.4, 0.05], [0.05, 0.05, 0.2]], dtype=np.complex64
        )
        # エルミート性を保証
        matrix = (matrix + matrix.conj().T) / 2
        # トレースを1に正規化
        matrix = matrix / np.trace(matrix)
        check_density(matrix)
