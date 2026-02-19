import pytest
import numpy as np
from pq_engine.time_evolution import TimeEvolution
from pq_engine.state import State
from pq_engine.exceptions import NotUnitaryError, NotSquareError


class TestTimeEvolution:
    def test_init_with_identity_matrix(self):
        """ユニタリな単位行列で初期化できるべき"""
        matrix = np.identity(2, dtype=np.complex64)
        time_evolution = TimeEvolution(matrix)
        np.testing.assert_array_equal(time_evolution.matrix, matrix)

    def test_init_with_hadamard_matrix(self):
        """アダマール行列で初期化できるべき"""
        matrix = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
        time_evolution = TimeEvolution(matrix)
        np.testing.assert_array_equal(time_evolution.matrix, matrix)

    def test_init_with_pauli_x_matrix(self):
        """パウリX行列で初期化できるべき"""
        matrix = np.array([[0, 1], [1, 0]], dtype=np.complex64)
        time_evolution = TimeEvolution(matrix)
        np.testing.assert_array_equal(time_evolution.matrix, matrix)

    def test_init_with_phase_matrix(self):
        """位相行列で初期化できるべき"""
        matrix = np.array([[1, 0], [0, 1j]], dtype=np.complex64)
        time_evolution = TimeEvolution(matrix)
        np.testing.assert_array_equal(time_evolution.matrix, matrix)

    def test_init_with_non_unitary_matrix_raises_error(self):
        """非ユニタリ行列で初期化すると例外が発生するべき"""
        matrix = np.array([[1, 2], [3, 4]], dtype=np.complex64)
        with pytest.raises(NotUnitaryError):
            TimeEvolution(matrix)

    def test_init_with_non_square_matrix_raises_error(self):
        """非正方形行列で初期化すると例外が発生するべき"""
        matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.complex64)
        with pytest.raises(NotSquareError):
            TimeEvolution(matrix)

    def test_init_with_scaled_unitary_matrix_raises_error(self):
        """スケーリングされたユニタリ行列で初期化すると例外が発生するべき"""
        matrix = np.array([[2, 0], [0, 2]], dtype=np.complex64)
        with pytest.raises(NotUnitaryError):
            TimeEvolution(matrix)

    def test_time_evolve_with_identity_matrix_preserves_state(self):
        """単位行列での時間発展は状態を保持するべき"""
        # 初期状態を作成
        psi = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.complex64)
        state_matrix = np.outer(psi, np.conj(psi))
        state = State(state_matrix)

        # 単位行列で時間発展
        identity_matrix = np.identity(2, dtype=np.complex64)
        time_evolution = TimeEvolution(identity_matrix)
        time_evolution.time_evolve(state)

        # 状態が変化しないことを確認
        np.testing.assert_array_almost_equal(state.matrix, state_matrix, decimal=6)

    def test_time_evolve_with_complex_state(self):
        """複素数で表現される状態での時間発展"""
        # 状態ベクトル: |psi> = (1/2)|0> + (i*sqrt(3)/2)|1>
        psi = np.array([0.5, 1j * np.sqrt(3) / 2], dtype=np.complex64)
        state_matrix = np.outer(psi, np.conj(psi))
        state = State(state_matrix)

        # パウリX行列で時間発展
        pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex64)
        time_evolution = TimeEvolution(pauli_x)
        time_evolution.time_evolve(state)

        # 行列のトレースチェック
        expected_trace = np.trace(state_matrix)
        actual_trace = np.trace(state.matrix)
        np.testing.assert_almost_equal(actual_trace, expected_trace, decimal=6)

    def test_time_evolve_preserves_trace(self):
        """時間発展は状態のトレースを保存するべき"""
        # ランダムな有効な状態を作成
        psi = np.array([0.6 + 0.2j, 0.8 - 0.3j], dtype=np.complex64)
        psi = psi / np.linalg.norm(psi)  # 正規化
        state_matrix = np.outer(psi, np.conj(psi))
        original_trace = np.trace(state_matrix)

        state = State(state_matrix)

        # ランダムなユニタリ行列を作成
        random_matrix = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
        # グラム・シュミットでユニタリ化
        random_matrix, _ = np.linalg.qr(random_matrix)
        time_evolution = TimeEvolution(random_matrix)
        time_evolution.time_evolve(state)

        # トレースが保存されていることを確認
        new_trace = np.trace(state.matrix)
        np.testing.assert_almost_equal(new_trace, original_trace, decimal=6)

    def test_time_evolve_with_3x3_matrix(self):
        """3x3行列での時間発展"""
        # 3x3の純粋状態を作成
        psi = np.array(
            [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)], dtype=np.complex64
        )
        state_matrix = np.outer(psi, np.conj(psi))
        state = State(state_matrix)

        # 3x3のユニタリ行列を作成
        random_matrix = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
        random_matrix, _ = np.linalg.qr(random_matrix)
        time_evolution = TimeEvolution(random_matrix)
        time_evolution.time_evolve(state)

        # 行列がエルミートになることを確認
        assert np.allclose(state.matrix, np.conj(state.matrix).T), (
            "状態行列はエルミートでなければならない"
        )

    def test_time_evolution_has_deterministic_behavior(self):
        """同じ行列と状態で時間発展を何度行っても結果が同じべき"""
        # 初期状態
        psi = np.array([0.8, 0.6j], dtype=np.complex64)
        psi = psi / np.linalg.norm(psi)
        state_matrix = np.outer(psi, np.conj(psi))

        # 重複する状態オブジェクト
        state1 = State(state_matrix.copy())
        state2 = State(state_matrix.copy())

        # 固定のユニタリ行列
        unitary_matrix = np.array([[0, 1], [1, 0]], dtype=np.complex64)
        time_evolution = TimeEvolution(unitary_matrix)

        # 両方の状態に同じ時間発展を適用
        time_evolution.time_evolve(state1)
        time_evolution.time_evolve(state2)

        # 結果が同じであることを確認
        np.testing.assert_array_almost_equal(state1.matrix, state2.matrix, decimal=6)

    def test_time_evolution_with_hadamard_gate(self):
        """アダマールゲートによる時間発展"""
        # |0> 状態
        psi = np.array([1, 0], dtype=np.complex64)
        state_matrix = np.outer(psi, np.conj(psi))
        state = State(state_matrix)

        # アダマール行列
        hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
        time_evolution = TimeEvolution(hadamard)
        time_evolution.time_evolve(state)

        # 期待される結果: H|0⟩⟨0|H† = [[0.5, 0.5], [0.5, 0.5]]
        expected_matrix = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex64)
        np.testing.assert_array_almost_equal(state.matrix, expected_matrix, decimal=6)

    def test_time_evolution_chain_operation(self):
        """連続した時間発展操作の正しさを検証"""
        # 初期状態 |0>
        psi = np.array([1, 0], dtype=np.complex64)
        state_matrix = np.outer(psi, np.conj(psi))
        state = State(state_matrix.copy())

        # 2つの異なるユニタリ行列
        hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
        phase = np.array([[1, 0], [0, 1j]], dtype=np.complex64)

        # 連続して適用 (U2 * U1 * |psi><psi| * U1† * U2†)
        te_hadamard = TimeEvolution(hadamard)
        te_phase = TimeEvolution(phase)

        te_hadamard.time_evolve(state)
        te_phase.time_evolve(state)

        # 一括で計算した場合と結果が一致することを確認
        psi_original = np.array([1, 0], dtype=np.complex64)
        state_matrix_direct = np.outer(psi_original, np.conj(psi_original))

        # U2 * U1
        combined_U = np.dot(phase, hadamard)
        # U1† * U2†
        combined_U_dag = np.dot(np.conj(hadamard).T, np.conj(phase).T)

        # 直接計算: U2 * U1 * |psi><psi| * U1† * U2†
        expected_matrix = np.dot(
            combined_U, np.dot(state_matrix_direct, combined_U_dag)
        )

        np.testing.assert_array_almost_equal(state.matrix, expected_matrix, decimal=6)

    def test_time_evolution_apply_order_matters(self):
        """適用順序が結果に影響することを検証"""
        # 初期状態 |+> = (|0> + |1>)/√2
        psi = np.array([1, 1], dtype=np.complex64) / np.sqrt(2)
        state_matrix = np.outer(psi, np.conj(psi))

        # アダマール行列と位相行列
        hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
        phase = np.array([[1, 0], [0, 1j]], dtype=np.complex64)

        # 順序1: H ->Phase
        state1 = State(state_matrix.copy())
        te_hadamard1 = TimeEvolution(hadamard)
        te_phase1 = TimeEvolution(phase)
        te_hadamard1.time_evolve(state1)
        te_phase1.time_evolve(state1)

        # 順序2: Phase -> H
        state2 = State(state_matrix.copy())
        te_phase2 = TimeEvolution(phase)
        te_hadamard2 = TimeEvolution(hadamard)
        te_phase2.time_evolve(state2)
        te_hadamard2.time_evolve(state2)

        # 順序が異なるので結果が異なるはず
        # もし結果が同じなら、行列の差が0になるので、assertでエラーになるはず
        difference = np.abs(state1.matrix - state2.matrix)
        assert not np.allclose(difference, 0, atol=1e-10), (
            "適用順序が同じだとテストとして意味がない"
        )

    def test_time_evolution_preserves_unitarity_preservation(self):
        """時間発展がユニタリ性を保存するかのテスト"""
        # 初期状態
        psi = np.array([0.9 + 0.1j, -0.2 + 0.4j], dtype=np.complex64)
        psi = psi / np.linalg.norm(psi)
        state_matrix = np.outer(psi, np.conj(psi))
        state = State(state_matrix.copy())

        # ランダムなユニタリ行列
        random_matrix = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
        random_matrix, _ = np.linalg.qr(random_matrix)
        time_evolution = TimeEvolution(random_matrix)
        time_evolution.time_evolve(state)

        # 変換後の状態もユニタリ（エルミート且つトレースが1）であることを確認
        assert np.allclose(state.matrix, np.conj(state.matrix).T), (
            "エルミート性を満たすべき"
        )
        assert np.allclose(np.trace(state.matrix), 1.0, atol=1e-5), (
            "トレースが1であるべき"
        )

    def test_compose_creates_tensor_product(self):
        """合成系の行列がテンソル積になっていることを確認"""
        hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
        pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex64)

        te_h = TimeEvolution(hadamard)
        te_p = TimeEvolution(pauli_x)

        composed = te_h.compose(te_p)

        expected = np.kron(hadamard, pauli_x)
        np.testing.assert_array_almost_equal(composed.matrix, expected, decimal=6)

    def test_compose_produces_unitary_matrix(self):
        """合成系の行列がユニタリであることを確認"""
        matrix1 = np.array([[1, 0], [0, 1j]], dtype=np.complex64)
        matrix2 = np.array([[0, 1], [1, 0]], dtype=np.complex64)

        te1 = TimeEvolution(matrix1)
        te2 = TimeEvolution(matrix2)

        composed = te1.compose(te2)

        U = composed.matrix
        U_dag = np.conj(U).T
        I = np.identity(4, dtype=np.complex64)

        np.testing.assert_array_almost_equal(np.dot(U, U_dag), I, decimal=6)
        np.testing.assert_array_almost_equal(np.dot(U_dag, U), I, decimal=6)

    def test_compose_with_different_dimensions(self):
        """異なる次元の時間発展を合成して正しい次元になることを確認"""
        hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
        unitary_3x3 = np.diag([1, 1j, -1]).astype(np.complex64)

        te_2x2 = TimeEvolution(hadamard)
        te_3x3 = TimeEvolution(unitary_3x3)

        composed = te_2x2.compose(te_3x3)

        assert composed.matrix.shape == (6, 6)

    def test_compose_on_composite_state(self):
        """合成系の時間発展が複合系に正しく作用することを確認"""
        hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
        pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex64)

        psi1 = np.array([1, 0], dtype=np.complex64)
        psi2 = np.array([0, 1], dtype=np.complex64)

        rho1 = np.outer(psi1, np.conj(psi1))
        rho2 = np.outer(psi2, np.conj(psi2))

        state = State(np.kron(rho1, rho2))

        te_h = TimeEvolution(hadamard)
        te_p = TimeEvolution(pauli_x)
        composed = te_h.compose(te_p)

        original_trace = np.trace(state.matrix)
        composed.time_evolve(state)

        assert np.trace(state.matrix) == pytest.approx(original_trace, abs=1e-5)
        assert np.allclose(state.matrix, np.conj(state.matrix).T)
