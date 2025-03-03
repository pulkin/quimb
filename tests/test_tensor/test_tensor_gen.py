from itertools import product
import functools

import pytest
import numpy as np
from numpy.testing import assert_allclose

import quimb as qu
import quimb.tensor as qtn
import quimb.gen.operators as qo


class TestSpinHam:

    @pytest.mark.parametrize("cyclic", [False, True])
    def test_var_terms(self, cyclic):
        n = 8
        Hd = qu.ham_mbl(n, dh=0.77, seed=42, cyclic=cyclic)
        Ht = qtn.MPO_ham_mbl(n, dh=0.77, seed=42, cyclic=cyclic).to_dense()
        assert_allclose(Hd, Ht)

    @pytest.mark.parametrize("var_two", ['none', 'some', 'only'])
    @pytest.mark.parametrize("var_one", ['some', 'only', 'onnly-some',
                                         'def-only', 'none'])
    def test_specials(self, var_one, var_two):
        K1 = qu.rand_herm(2**1)

        n = 10
        HB = qtn.SpinHam(S=1 / 2)

        if var_two == 'some':
            HB += 1, K1, K1
            HB[4, 5] += 1, K1, K1
            HB[7, 8] += 1, K1, K1
        elif var_two == 'only':
            for i in range(n - 1):
                HB[i, i + 1] += 1, K1, K1
        else:
            HB += 1, K1, K1

        if var_one == 'some':
            HB += 1, K1
            HB[2] += 1, K1
            HB[3] += 1, K1
        elif var_one == 'only':
            for i in range(n - 1):
                HB[i] += 1, K1
        elif var_one == 'only-some':
            HB[1] += 1, K1
        elif var_one == 'def-only':
            HB += 1, K1

        HB.build_nni(n)
        H_mpo = HB.build_mpo(n)
        H_sps = HB.build_sparse(n)

        assert_allclose(H_mpo.to_dense(), H_sps.A)

    def test_no_default_term(self):
        N = 10
        builder = qtn.SpinHam(1 / 2)

        for i in range(N - 1):
            builder[i, i + 1] += 1.0, 'Z', 'Z'

        H = builder.build_mpo(N)

        dmrg = qtn.DMRG2(H)
        dmrg.solve(verbosity=1)

        assert dmrg.energy == pytest.approx(-2.25)


def symbolic_mpo_contraction(mpo_tensors, s: str = " + ", p: str = "*"):
    """
    Performs a symbolic contraction of the MPO.

    Designed to check the correctness of MPO construction by
    performing out all aux bond contractions of symbolic tensors.

    Parameters
    ----------
    mpo
        Symbolic tensors of MPO.
    s : str
        The sum label.
    p : str
        The (direct) product label.

    Returns
    -------
    str
        A string with the symbolic contraction.
    """

    def string_product(s1: str, s2: str, s: str, p: str):
        """Scalar product of two strings."""
        s1 = s1.split(s)
        s2 = s2.split(s)
        return s.join(i + p + j for i, j in product(s1, s2) if len(i) * len(j) > 0)

    result = None
    for i in mpo_tensors:
        assert i.dtype.kind in 'SU'

        if i.ndim == 0:
            i = np.array([[i]])

        if i.ndim == 1:
            if result is None:
                i = i[np.newaxis, :]
            else:
                i = i[:, np.newaxis]

        assert i.ndim == 2

        if result is None:
            result = i

        else:
            assert i.shape[0] == result.shape[1]
            new = []
            for row in result:
                new.append([])
                for column in i.T:
                    new[-1].append(s.join(
                        _t for _t in (string_product(_i, _j, s, p) for _i, _j in zip(row, column)) if len(_t) > 0))
            result = np.array(new)
    return result.squeeze()


@functools.lru_cache(maxsize=2)
def get_1p_basis():
    """Convention-dependent 1p basis"""
    op_a_, op_a = map(qu.fermion_operator, "+-")
    op_n = op_a_.dot(op_a)
    vals, vecs = np.linalg.eigh(op_n)
    id_empty, id_occ = np.argsort(vals)
    vec_empty = vecs[:, id_empty]
    vec_occ = vecs[:, id_occ]
    return vec_empty, vec_occ


def MPS_np_state(n, i):
    """
    Assembles a fermionic n-particle state.

    Parameters
    ----------
    n : int
        The size of the space.
    i
        An ordered list of occupied states.

    Returns
    -------
    MPS_product_state
        A product state where the specific degrees of freedom are occupied.
    """
    if isinstance(i, int):
        i = (i,)
    i = np.array(i)
    if len(set(i)) != len(i):
        raise ValueError(f"Some of the input states are not unique: {i}")
    v, _ = get_1p_basis()
    op_one, op_a_, op_z = map(qu.fermion_operator, "i+z")
    result = v[np.newaxis, :] * np.ones(n)[:, np.newaxis]
    for _i in i:
        result = np.einsum("nij,nj->ni", [op_one] * _i + [op_a_] + [op_z] * (n-_i-1), result)
    return qtn.MPS_product_state(result)


class TestTBHam:

    def test_symbolic_empty(self):
        assert symbolic_mpo_contraction(qtn.hubbard_ham_mpo_tensor(1, 1, symbolic=True)) == ""

    def test_symbolic_121(self):
        assert symbolic_mpo_contraction(qtn.hubbard_ham_mpo_tensor(1, [.1, 1], symbolic=True)) == "{e,n}"

    def test_symbolic_115(self):
        assert set(str(symbolic_mpo_contraction(qtn.hubbard_ham_mpo_tensor(1, [np.eye(5)], symbolic=True))).split(" + ")) == {
            "{e_0,n}*1*1*1*1", "1*{e_1,n}*1*1*1", "1*1*{e_2,n}*1*1", "1*1*1*{e_3,n}*1", "1*1*1*1*{e_4,n}"}

    def test_symbolic_115_ndiag(self):
        assert set(str(symbolic_mpo_contraction(qtn.hubbard_ham_mpo_tensor(1, [np.eye(5) + np.eye(5, k=2) + np.eye(5, k=-2)],
                                                                           symbolic=True))).split(" + ")) == {
            "{e_0,n}*1*1*1*1", "1*{e_1,n}*1*1*1", "1*1*{e_2,n}*1*1", "1*1*1*{e_3,n}*1", "1*1*1*1*{e_4,n}",
            "a^*sz*{t_0_2,(sz,a)}*1*1", "1*a^*sz*{t_1_3,(sz,a)}*1", "1*1*a^*sz*{t_2_4,(sz,a)}",
            "a*sz*{t_0_2^,(a^,sz)}*1*1", "1*a*sz*{t_1_3^,(a^,sz)}*1", "1*1*a*sz*{t_2_4^,(a^,sz)}"
        }

    def test_symbolic_221(self):
        assert set(str(symbolic_mpo_contraction(qtn.hubbard_ham_mpo_tensor(2, 1, symbolic=True))).split(" + ")) == {
            "a^*{t,(sz,a)}", "a*{t^,(a^,sz)}"}

    def test_symbolic_212(self):
        assert set(str(symbolic_mpo_contraction(qtn.hubbard_ham_mpo_tensor(2, [np.eye(2)], symbolic=True))).split(" + ")) == {
            "{e_0,n}*1*1*1", "1*{e_1,n}*1*1", "1*1*{e_0,n}*1", "1*1*1*{e_1,n}"}

    def test_symbolic_222(self):
        assert set(str(symbolic_mpo_contraction(qtn.hubbard_ham_mpo_tensor(2, [np.eye(2), np.eye(2)], symbolic=True))).split(
            " + ")) == {
            "{e_0,n}*1*1*1", "1*{e_1,n}*1*1", "1*1*{e_0,n}*1", "1*1*1*{e_1,n}", "a^*sz*{t_1_[0,0],(sz,a)}*1",
            "1*a^*sz*{t_1_[1,1],(sz,a)}", "a*sz*{t_1_[0,0]^,(a^,sz)}*1", "1*a*sz*{t_1_[1,1]^,(a^,sz)}"}

    @pytest.mark.parametrize("n", [2, 3])
    @pytest.mark.parametrize("blocks", [1 + .1j, (-2, 1+.1j), [np.diag(np.arange(3)),
                                                               np.exp(1.j * np.arange(9)).reshape(3, 3)]])
    def test_main(self, n, blocks):
        uc_size = qo._hubbard_canonic_2s_form(blocks).shape[1]
        N = n * uc_size

        mpo = qtn.MPO_ham_hubbard(n, blocks)

        ham_1p = np.empty((N, N), dtype=np.complex128)
        for i in range(N):
            for j in range(N):
                bra = MPS_np_state(N, i)
                ket = MPS_np_state(N, j)
                bra.align_(mpo, ket)
                result = bra & mpo & ket
                ham_1p[i, j] = result ^ all

        assert_allclose(ham_1p, qo.ham_tb(n, blocks))


class TestHubbardHam:

    def test_symbolic_empty(self):
        assert symbolic_mpo_contraction(qtn.hubbard_ham_mpo_tensor(1, 1, coulomb_blocks=1, symbolic=True)) == ""

    def test_symbolic_empty_strange(self):
        assert symbolic_mpo_contraction(qtn.hubbard_ham_mpo_tensor(1, [1, 0],
                                                                   coulomb_blocks=[-1, 0], symbolic=True)) == ""

    def test_symbolic_121(self):
        assert symbolic_mpo_contraction(qtn.hubbard_ham_mpo_tensor(1, [1, 0],
                                                                   coulomb_blocks=[1, 0], symbolic=True)) == "{e,n}"

    def test_symbolic_115_pure(self):
        assert set(str(symbolic_mpo_contraction(
            qtn.hubbard_ham_mpo_tensor(1, [np.zeros((5, 5))], coulomb_blocks=[np.eye(5, k=2) + np.eye(5, k=-2)], symbolic=True)
        )).split(" + ")) == {
            "n*1*{u_0_2,n}*1*1", "1*n*1*{u_1_3,n}*1", "1*1*n*1*{u_2_4,n}",
        }

    def test_symbolic_421_mixed(self):
        assert set(str(symbolic_mpo_contraction(
            qtn.hubbard_ham_mpo_tensor(4, 1, coulomb_blocks=[1, 1, 1], symbolic=True)
        )).split(" + ")) == {
            "a^*{t,(sz,a)}*1*1", "1*a^*{t,(sz,a)}*1", "1*1*a^*{t,(sz,a)}",
            "a*{t^,(a^,sz)}*1*1", "1*a*{t^,(a^,sz)}*1", "1*1*a*{t^,(a^,sz)}",
            "{e,n}*1*1*1", "1*{e,n}*1*1", "1*1*{e,n}*1", "1*1*1*{e,n}",
            "n*{u_1,n}*1*1", "1*n*{u_1,n}*1", "1*1*n*{u_1,n}",
            "n*1*{u_2,n}*1", "1*n*1*{u_2,n}",
        }

    @pytest.mark.parametrize("n", [2, 3])
    @pytest.mark.parametrize("blocks", [1, (0, -2, 3), [[(0, 1, 2), (1, 0, 3), (2, 3, 0)],
                                                        np.exp(np.arange(9)).reshape(3, 3)]])
    def test_main(self, n, blocks):
        uc_size = qo._hubbard_canonic_2s_form(blocks).shape[1]
        N = n * uc_size

        mpo = qtn.MPO_ham_hubbard(n, coulomb_blocks=blocks)

        ham_1p = np.zeros((N, N), dtype=np.complex128)
        for i in range(N):
            for j in range(i + 1, N):
                bra = MPS_np_state(N, (i, j))
                ket = bra.H
                bra.align_(mpo, ket)
                result = bra & mpo & ket
                ham_1p[i, j] = ham_1p[j, i] = result ^ all

        assert_allclose(ham_1p, qo.ham_tb(n, blocks))
