"""Generate specific tensor network states and operators.
"""
from numbers import Integral
from itertools import permutations

import numpy as np

from ..core import make_immutable, ikron
from ..linalg.base_linalg import norm_fro_dense
from ..gen.operators import spin_operator, fermion_operator, eye, _gen_mbl_random_factors, _hubbard_canonic_2s_form
from ..gen.rand import randn, choice, random_seed_fn, rand_phase
from .tensor_core import Tensor, _asarray
from .tensor_1d import MatrixProductState, MatrixProductOperator
from .tensor_tebd import NNI


@random_seed_fn
def rand_tensor(shape, inds, tags=None, dtype=float, left_inds=None):
    """Generate a random tensor with specified shape and inds.

    Parameters
    ----------
    shape : sequence of int
        Size of each dimension.
    inds : sequence of str
        Names of each dimension.
    tags : sequence of str
        Labels to tag this tensor with.
    dtype : {'float64', 'complex128', 'float32', 'complex64'}, optional
        The underlying data type.
    left_inds : sequence of str, optional
        Which, if any, indices to group as 'left' indices of an effective
        matrix. This can be useful, for example, when automatically applying
        unitary constraints to impose a certain flow on a tensor network but at
        the atomistic (Tensor) level.

    Returns
    -------
    Tensor
    """
    data = randn(shape, dtype=dtype)
    return Tensor(data=data, inds=inds, tags=tags, left_inds=left_inds)


@random_seed_fn
def rand_phased(shape, inds, tags=None, dtype=complex):
    """Generate a random tensor with specified shape and inds, and randomly
    'phased' (distributed on the unit circle) data, such that
    ``T.H @ T == T.norm()**2 == T.size``.

    Parameters
    ----------
    shape : sequence of int
        Size of each dimension.
    inds : sequence of str
        Names of each dimension.
    tags : sequence of str
        Labels to tag this tensor with.
    dtype : {'complex128', 'complex64'}, optional
        The underlying data type - can only be complex.

    Returns
    -------
    Tensor
    """
    data = rand_phase(shape, dtype=dtype)
    return Tensor(data=data, inds=inds, tags=tags)


# --------------------------------------------------------------------------- #
#                                    MPSs                                     #
# --------------------------------------------------------------------------- #

@random_seed_fn
def MPS_rand_state(n, bond_dim, phys_dim=2, normalize=True, cyclic=False,
                   dtype=float, trans_invar=False, **mps_opts):
    """Generate a random matrix product state.

    Parameters
    ----------
    n : int
        The number of sites.
    bond_dim : int
        The bond dimension.
    phys_dim : int, optional
        The physical (site) dimensions, defaults to 2.
    normalize : bool, optional
        Whether to normalize the state.
    cyclic : bool, optional
        Generate a MPS with periodic boundary conditions or not, default is
        open boundary conditions.
    dtype : {float, complex} or numpy dtype, optional
        Data type of the tensor network.
    trans_invar : bool (optional)
        Whether to generate a translationally invariant state,
        requires cyclic=True.
    mps_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductState`.
    """
    if trans_invar and not cyclic:
        raise ValueError("State cannot be translationally invariant with open "
                         "boundary conditions.")

    cyc_dim = (bond_dim,) if cyclic else ()

    def gen_shapes():
        yield (*cyc_dim, bond_dim, phys_dim)
        for _ in range(n - 2):
            yield (bond_dim, bond_dim, phys_dim)
        yield (bond_dim, *cyc_dim, phys_dim)

    def gen_data(shape):
        return randn(shape, dtype=dtype)

    def scale(x):
        return x / norm_fro_dense(x)**(1 / (x.ndim - 1))

    if trans_invar:
        array = scale(gen_data(next(gen_shapes())))
        arrays = (array for _ in range(n))
    else:
        arrays = map(scale, map(gen_data, gen_shapes()))

    rmps = MatrixProductState(arrays, **mps_opts)

    if normalize == 'left':
        rmps.left_canonize(normalize=True)
    elif normalize == 'right':
        rmps.left_canonize(normalize=True)
    elif normalize:
        rmps /= (rmps.H @ rmps)**0.5

    return rmps


def MPS_product_state(arrays, cyclic=False, **mps_opts):
    """Generate a product state in MatrixProductState form, i,e,
    with bond dimension 1, from single site vectors described by ``arrays``.
    """
    cyc_dim = (1,) if cyclic else ()

    def gen_array_shapes():
        if len(arrays) == 1:
            yield (-1,)
        else:
            yield (*cyc_dim, 1, -1)
            for _ in range(len(arrays) - 2):
                yield (1, 1, -1)
            yield (*cyc_dim, 1, -1)

    mps_arrays = (_asarray(array).reshape(*shape)
                  for array, shape in zip(arrays, gen_array_shapes()))

    return MatrixProductState(mps_arrays, shape='lrp', **mps_opts)


def MPS_computational_state(binary, dtype=float, cyclic=False, **mps_opts):
    """A computational basis state in Matrix Product State form.

    Parameters
    ----------
    binary : str or sequence of int
        String specifying the state, e.g. ``'00101010111'`` or ``[0, 0, 1]``.
    cyclic : bool, optional
        Generate a MPS with periodic boundary conditions or not, default open
        boundary conditions.
    mps_opts
        Supplied to MatrixProductState constructor.
    """

    array_map = {
        '0': np.array([1., 0.], dtype=dtype),
        '1': np.array([0., 1.], dtype=dtype),
    }

    def gen_arrays():
        for s in binary:
            yield array_map[str(s)]

    return MPS_product_state(tuple(gen_arrays()), cyclic=cyclic, **mps_opts)


def MPS_neel_state(n, down_first=False, dtype=float, **mps_opts):
    """Generate the neel state in Matrix Product State form.

    Parameters
    ----------
    n : int
        The number of spins.
    down_first : bool, optional
        Whether to start with '1' (down) or '0' (up) first.
    mps_opts
        Supplied to MatrixProductState constructor.
    """
    binary_str = "01" * (n // 2) + (n % 2 == 1) * "0"
    if down_first:
        binary_str = "1" + binary_str[:-1]
    return MPS_computational_state(binary_str, dtype=dtype, **mps_opts)


@random_seed_fn
def MPS_rand_computational_state(n, dtype=float, **mps_opts):
    """Generate a random computation basis state, like '01101001010'.

    Parameters
    ----------
    n : int
        The number of qubits.
    seed : int, optional
        The seed to use.
    dtype : {float, complex} or numpy dtype, optional
        Data type of the tensor network.
    mps_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductState`.
    """
    cstr = (choice(('0', '1')) for _ in range(n))
    return MPS_computational_state(cstr, dtype=dtype, **mps_opts)


def MPS_zero_state(n, bond_dim=1, phys_dim=2, cyclic=False,
                   dtype=float, **mps_opts):
    """The all-zeros MPS state, of given bond-dimension.

    Parameters
    ----------
    n : int
        The number of sites.
    bond_dim : int, optional
        The bond dimension, defaults to 1.
    phys_dim : int, optional
        The physical (site) dimensions, defaults to 2.
    cyclic : bool, optional
        Generate a MPS with periodic boundary conditions or not, default is
        open boundary conditions.
    dtype : {float, complex} or numpy dtype, optional
        Data type of the tensor network.
    mps_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductState`.
    """
    cyc_dim = (bond_dim,) if cyclic else ()

    def gen_arrays():
        yield np.zeros((*cyc_dim, bond_dim, phys_dim), dtype=dtype)
        for _ in range(n - 2):
            yield np.zeros((bond_dim, bond_dim, phys_dim), dtype=dtype)
        yield np.zeros((bond_dim, *cyc_dim, phys_dim), dtype=dtype)

    return MatrixProductState(gen_arrays(), **mps_opts)


def MPS_sampler(n, dtype=complex, squeeze=True, **mps_opts):
    """A product state for sampling tensor network traces. Seen as a vector it
    has the required property that ``psi.H @ psi == d`` always for hilbert
    space size ``d``.
    """
    arrays = [rand_phase(2, dtype=dtype) for _ in range(n)]
    psi = MPS_product_state(arrays, **mps_opts)
    if squeeze:
        psi.squeeze_()
    return psi


# --------------------------------------------------------------------------- #
#                                    MPOs                                     #
# --------------------------------------------------------------------------- #

def MPO_identity(n, phys_dim=2, dtype=float, cyclic=False, **mpo_opts):
    """Generate an identity MPO of size ``n``.

    Parameters
    ----------
    n : int
        The number of sites.
    phys_dim : int, optional
        The physical (site) dimensions, defaults to 2.
    dtype : {float, complex} or numpy dtype, optional
        Data type of the tensor network.
    cyclic : bool, optional
        Generate a MPO with periodic boundary conditions or not, default is
        open boundary conditions.
    mpo_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductOperator`.
    """
    II = np.identity(phys_dim, dtype=dtype)
    cyc_dim = (1,) if cyclic else ()

    def gen_arrays():
        yield II.reshape(*cyc_dim, 1, phys_dim, phys_dim)
        for _ in range(n - 2):
            yield II.reshape(1, 1, phys_dim, phys_dim)
        yield II.reshape(1, *cyc_dim, phys_dim, phys_dim)

    return MatrixProductOperator(gen_arrays(), **mpo_opts)


def MPO_identity_like(mpo, **mpo_opts):
    """Return an identity matrix operator with the same physical index and
    inds/tags as ``mpo``.
    """
    return MPO_identity(n=mpo.nsites, phys_dim=mpo.phys_dim(), dtype=mpo.dtype,
                        site_tag_id=mpo.site_tag_id, cyclic=mpo.cyclic,
                        upper_ind_id=mpo.upper_ind_id,
                        lower_ind_id=mpo.lower_ind_id, **mpo_opts)


def MPO_zeros(n, phys_dim=2, dtype=float, cyclic=False, **mpo_opts):
    """Generate a zeros MPO of size ``n``.

    Parameters
    ----------
    n : int
        The number of sites.
    phys_dim : int, optional
        The physical (site) dimensions, defaults to 2.
    dtype : {float, complex} or numpy dtype, optional
        Data type of the tensor network.
    cyclic : bool, optional
        Generate a MPO with periodic boundary conditions or not, default is
        open boundary conditions.
    mpo_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductOperator`.
    """
    cyc_dim = (1,) if cyclic else ()

    def gen_arrays():
        yield np.zeros((*cyc_dim, 1, phys_dim, phys_dim), dtype=dtype)
        for _ in range(n - 2):
            yield np.zeros((1, 1, phys_dim, phys_dim), dtype=dtype)
        yield np.zeros((1, *cyc_dim, phys_dim, phys_dim), dtype=dtype)

    return MatrixProductOperator(gen_arrays(), **mpo_opts)


def MPO_zeros_like(mpo, **mpo_opts):
    """Return a zeros matrix operator with the same physical index and
    inds/tags as ``mpo``.
    """
    return MPO_zeros(n=mpo.nsites, phys_dim=mpo.phys_dim(),
                     dtype=mpo.dtype, site_tag_id=mpo.site_tag_id,
                     upper_ind_id=mpo.upper_ind_id, cyclic=mpo.cyclic,
                     lower_ind_id=mpo.lower_ind_id, **mpo_opts)


@random_seed_fn
def MPO_rand(n, bond_dim, phys_dim=2, normalize=True, cyclic=False,
             herm=False, dtype=float, **mpo_opts):
    """Generate a random matrix product state.

    Parameters
    ----------
    n : int
        The number of sites.
    bond_dim : int
        The bond dimension.
    phys_dim : int, optional
        The physical (site) dimensions, defaults to 2.
    normalize : bool, optional
        Whether to normalize the operator such that ``trace(A.H @ A) == 1``.
    cyclic : bool, optional
        Generate a MPO with periodic boundary conditions or not, default is
        open boundary conditions.
    dtype : {float, complex} or numpy dtype, optional
        Data type of the tensor network.
    herm : bool, optional
        Whether to make the matrix hermitian (or symmetric if real) or not.
    mpo_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductOperator`.
    """
    cyc_shp = (bond_dim,) if cyclic else ()

    shapes = [(*cyc_shp, bond_dim, phys_dim, phys_dim),
              *((bond_dim, bond_dim, phys_dim, phys_dim),) * (n - 2),
              (bond_dim, *cyc_shp, phys_dim, phys_dim)]

    def gen_data(shape):
        data = randn(shape, dtype=dtype)
        if not herm:
            return data

        trans = (0, 2, 1) if len(shape) == 3 else (0, 1, 3, 2)
        return data + data.transpose(*trans).conj()

    arrays = map(lambda x: x / norm_fro_dense(x)**(1 / (x.ndim - 1)),
                 map(gen_data, shapes))

    rmpo = MatrixProductOperator(arrays, **mpo_opts)

    if normalize:
        rmpo /= (rmpo.H @ rmpo)**0.5

    return rmpo


@random_seed_fn
def MPO_rand_herm(n, bond_dim, phys_dim=2, normalize=True,
                  dtype=float, **mpo_opts):
    """Generate a random hermitian matrix product operator.
    See :class:`~quimb.tensor.tensor_gen.MPO_rand`.
    """
    return MPO_rand(n, bond_dim, phys_dim=phys_dim, normalize=normalize,
                    dtype=dtype, herm=True, **mpo_opts)


# ---------------------------- MPO hamiltonians ----------------------------- #

def maybe_make_real(X):
    """Check if ``X`` is real, if so, convert to contiguous array.
    """
    if np.allclose(X.imag, np.zeros_like(X)):
        return np.ascontiguousarray(X.real)
    return X


def spin_ham_mpo_tensor(one_site_terms, two_site_terms, S=1 / 2,
                        left_two_site_terms=None, which=None, cyclic=False):
    """Generate tensor(s) for a spin hamiltonian MPO.

    Parameters
    ----------
    one_site_terms : sequence of (scalar, operator)
        The terms that act on a single site, each ``operator`` can be a string
        suitable to be sent to :func:`spin_operator` or an actual 2d-array.
    two_site_terms : sequence of (scalar, operator operator)
        The terms that act on two neighbouring sites, each ``operator`` can be
        a string suitable to be sent to :func:`spin_operator` or an actual
        2d-array.
    S : fraction, optional
        What size spin to use, defaults to spin-1/2.
    left_two_site_terms : sequence of (scalar, operator operator), optional
        If the interaction to the left of this site has different spin terms
        then the equivalent list of terms for that site.
    which : {None, 'L', 'R', 'A'}, optional
        If ``None``, generate the middle tensor, if 'L' a left-end tensor, if
        'R' a right-end tensor and if 'A' all three.
    cyclic : bool, optional
        Whether to use periodic boundary conditions - default False.

    Returns
    -------
    numpy.ndarray[, numpy.ndarray, numpy.ndarray]
        The middle, left, right or all three MPO tensors.
    """
    # assume same interaction type everywhere
    if left_two_site_terms is None:
        left_two_site_terms = two_site_terms

    # local dimension
    D = int(2 * S + 1)
    # bond dimension to right
    B = len(two_site_terms) + 2
    # bond dimension to left
    BL = len(left_two_site_terms) + 2

    H = np.zeros((BL, B, D, D), dtype=complex)

    # add one-body terms
    for factor, s in one_site_terms:
        if isinstance(s, str):
            s = spin_operator(s, S=S)
        H[-1, 0, :, :] += factor * s

    # add two-body terms
    for i, (factor, s1, _) in enumerate(two_site_terms):
        if isinstance(s1, str):
            s1 = spin_operator(s1, S=S)
        H[-1, 1 + i, :, :] = factor * s1

    for i, (_, _, s2) in enumerate(left_two_site_terms):
        if isinstance(s2, str):
            s2 = spin_operator(s2, S=S)
        H[i + 1, 0, :, :] = s2

    H[0, 0, :, :] = eye(D)
    H[-1, -1, :, :] = eye(D)

    H = maybe_make_real(H)
    make_immutable(H)

    if which in {None, 'M'}:
        return H

    if cyclic:
        # need special conditions for first MPO matrix
        HL = np.zeros_like(H)
        HL[0, :, :, :] = H[-1, :, :, :]
        HL[1:-1, -1, :, :] = H[1:-1, 0, :, :]
        HR = H
    else:
        HL = H[-1, :, :, :]
        HR = H[:, 0, :, :]

    if which == 'L':
        return HL
    elif which == 'R':
        return HR
    elif which == 'A':
        return HL, H, HR


class _TermAdder:
    """Simple class to allow ``SpinHam`` syntax like
    ``builder[i, j] += (1/2, 'Z', 'X')``. This object is temporarily created
    by the getitem call, accumulates the new term, then has its the new
    combined list of terms extracted in the setitem call.
    """

    def __init__(self, terms, nsite):
        self.terms = terms
        self.nsite = nsite

    def __iadd__(self, new):
        if len(new) != self.nsite + 1:
            raise ValueError(
                "New terms should be of the form")

        if self.terms is None:
            self.terms = [new]
        else:
            self.terms += [new]
        return self


class SpinHam:
    """Class for easily building custom spin hamiltonians in MPO or NNI form.
    Currently limited to nearest neighbour interactions (and single site
    terms). It is possible to set 'default' translationally invariant terms,
    but also terms acting on specific sites only (which take precedence).
    It is also possible to build a sparse matrix version of the hamiltonian
    (obviously for small sizes only).

    Parameters
    ----------
    S : float, optional
        The type of spin, defaults to 1/2.
    cyclic : bool, optional
        Whether to use periodic boundary conditions - default is False.

    Examples
    --------
    Initialize the spin hamiltonian builder:

        >>> builder = SpinHam(S=3 / 2)

    Add some two-site terms:

        >>> builder += 0.5, '+', '-'
        >>> builder += 0.5, '-', '+'
        >>> builder += 1.0, 'Z', 'Z'

    Add a single site term:

        >>> builder -= 0.3, 'Z'

    Build a MPO version of the hamiltonian for use with DMRG:

        >>> mpo_ham = builder.build_mpo(100)
        >>> mpo_ham
        <MatrixProductOperator(tensors=100, structure='I{}', nsites=100)>

    Build a NNI version of the hamiltonian for use with TEBD:

        >>> builder.build_nni(100)
        <NNI(n=100, cyclic=False)>

    You can also set terms for specific sites (this overides any of the
    'default', translationally invariant terms set as above):

        >>> builder[10, 11] += 0.75, '+', '-'
        >>> builder[10, 11] += 0.75, '-', '+'
        >>> builder[10, 11] += 1.5, 'Z', 'Z'

    Or specific one-site terms (which again overides any default
    single site terms set above):

        >>> builder[10] += 3.7, 'Z'
        >>> builder[11] += 0.0, 'I' # '0' term turns off field
    """

    def __init__(self, S=1 / 2, cyclic=False):
        self.S = S
        self.one_site_terms = []
        self.two_site_terms = []
        self.cyclic = cyclic

        # Holders for any non-translationally invariant terms
        self.var_one_site_terms = {}
        self.var_two_site_terms = {}

    def add_term(self, factor, *operators):
        """Add another term to the expression to be built.

        Parameters
        ----------
        factor : scalar
            Scalar factor to multiply this term by.
        *operators : str or array
            The operators to use. Can specify one or two for single or two site
            terms respectively. Can use strings, which are supplied to
            :func:`~quimb.spin_operator`, or actual arrays as long as they have
            the correct dimension.
        """
        if factor == 0.0:
            # no need to add zero terms
            return

        if len(operators) == 1:
            self.one_site_terms.append((factor, *operators))
        elif len(operators) == 2:
            self.two_site_terms.append((factor, *operators))
        else:
            raise NotImplementedError("3-body+ terms are not supported yet.")

    def sub_term(self, factor, *operators):
        """Subtract a term - simple alias that flips sign of ``factor``.
        """
        self.add_term(-factor, *operators)

    def __iadd__(self, term):
        self.add_term(*term)
        return self

    def __isub__(self, term):
        self.sub_term(*term)
        return self

    def __getitem__(self, sites):
        """Part of the machinery that allows terms to be added to specific
        sites like::

            >>> builder[i] += 1/2, 'X'
            >>> builder[45, 46] += 1/2, 'Z', 'Z'

        """
        if isinstance(sites, Integral):
            return _TermAdder(self.var_one_site_terms.get(sites, None), 1)

        i, j = sorted(sites)
        if j - i != 1:
            raise NotImplementedError("Can only add nearest neighbour terms.")

        return _TermAdder(self.var_two_site_terms.get(sites, None), 2)

    def __setitem__(self, sites, value):
        """Part of the machinery that allows terms to be added to specific
        sites like::

            >>> builder[i] += 1/2, 'X'
            >>> builder[45, 46] += 1/2, 'Z', 'Z'

        Could also be called directly with a list of terms like::

            >>> builder[13, 14] = [(1, 'Z', 'Z'), (0.5, 'X', 'Y')]

        Which would overide any terms set so far.
        """
        if isinstance(value, _TermAdder):
            terms = value.terms
        else:
            terms = value

        if isinstance(sites, Integral):
            self.var_one_site_terms[sites] = terms
        else:
            i, j = sorted(sites)
            if j - i != 1:
                raise ValueError("Can only add nearest neighbour terms.")
            self.var_two_site_terms[sites] = terms

    def build_mpo(self, n, upper_ind_id='k{}', lower_ind_id='b{}',
                  site_tag_id='I{}', tags=None, bond_name=""):
        """Build an MPO instance of this spin hamiltonian of size ``n``. See
        also ``MatrixProductOperator``.
        """
        # cache the default term
        t_defs = {}

        def get_default_term(which):
            try:
                return t_defs[which]
            except KeyError:
                t_defs['L'], t_defs[None], t_defs['R'] = spin_ham_mpo_tensor(
                    self.one_site_terms, self.two_site_terms,
                    S=self.S, which='A', cyclic=self.cyclic)
                return t_defs[which]

        def gen_tensors():
            for i in range(n):
                which = {0: 'L', n - 1: 'R'}.get(i, None)

                ij_L = (i - 1, i)
                ij_R = (i, i + 1)

                # check for site/bond specific terms
                var_one = i in self.var_one_site_terms
                var_two = (
                    (ij_L in self.var_two_site_terms) or
                    (ij_R in self.var_two_site_terms)
                )

                if not (var_one or var_two):
                    yield get_default_term(which)
                else:
                    t1s = self.var_one_site_terms.get(i, self.one_site_terms)
                    t2s = self.var_two_site_terms.get(ij_R,
                                                      self.two_site_terms)
                    t2s_L = self.var_two_site_terms.get(ij_L,
                                                        self.two_site_terms)

                    yield spin_ham_mpo_tensor(t1s, t2s, S=self.S,
                                              left_two_site_terms=t2s_L,
                                              which=which, cyclic=self.cyclic)

        return MatrixProductOperator(arrays=gen_tensors(), bond_name=bond_name,
                                     upper_ind_id=upper_ind_id,
                                     lower_ind_id=lower_ind_id,
                                     site_tag_id=site_tag_id, tags=tags)

    def build_sparse(self, n, **ikron_opts):
        """Build a sparse matrix representation of this Hamiltonian.

        Parameters
        ----------
        n : int, optional
            The number of spins to build the matrix for.
        ikron_opts
            Supplied to :func:`~quimb.core.ikron`.

        Returns
        -------
        H : matrix
        """
        ikron_opts.setdefault('sparse', True)

        D = int(2 * self.S + 1)
        dims = [D] * n

        terms = []
        for i in range(n):

            t1s = self.var_one_site_terms.get(i, self.one_site_terms)
            for factor, s in t1s:
                if isinstance(s, str):
                    s = spin_operator(s, S=self.S, sparse=True)
                terms.append(
                    ikron(factor * s, dims, i, **ikron_opts)
                )

            if (i + 1 == n) and (not self.cyclic):
                break

            t2s = self.var_two_site_terms.get((i, i + 1), self.two_site_terms)
            for factor, s1, s2 in t2s:
                if isinstance(s1, str):
                    s1 = spin_operator(s1, S=self.S, sparse=True)
                if isinstance(s2, str):
                    s2 = spin_operator(s2, S=self.S, sparse=True)
                terms.append(
                    ikron([factor * s1, s2], dims, [i, i + 1], **ikron_opts)
                )

        return sum(terms)

    def _get_spin_op(self, factor, *ss):
        if len(ss) == 1:
            s, = ss
            if isinstance(s, str):
                s = spin_operator(s, S=self.S)
            return factor * s

        if len(ss) == 2:
            s1, s2 = ss
            if isinstance(s1, str):
                s1 = spin_operator(s1, S=self.S)
            if isinstance(s2, str):
                s2 = spin_operator(s2, S=self.S)
            return factor * (s1 & s2)

    def _sum_spin_ops(self, terms):
        H = sum(self._get_spin_op(*term) for term in terms)
        H = maybe_make_real(H)
        make_immutable(H)
        return H

    def build_nni(self, n=None, **nni_opts):
        """Build a nearest neighbour interactor instance of this spin
        hamiltonian of size ``n``. See also
        :class:`~quimb.tensor.tensor_tebd.NNI`.

        Parameters
        ----------
        n : int, optional
            The number of spins, if the hamiltonian only has two-site terms
            this is optional.

        Returns
        -------
        NNI
        """
        H1s, H2s = {}, {}

        # add default two site term
        if self.two_site_terms:
            H2s[None] = self._sum_spin_ops(self.two_site_terms)

        # add specific two site terms
        if self.var_two_site_terms:
            for sites, terms in self.var_two_site_terms.items():
                H2s[sites] = self._sum_spin_ops(terms)

        # add default one site term
        if self.one_site_terms:
            H1s[None] = self._sum_spin_ops(self.one_site_terms)

        # add specific one site terms
        if self.var_one_site_terms:
            for site, terms in self.var_one_site_terms.items():
                H1s[site] = self._sum_spin_ops(terms)

        return NNI(H2=H2s, H1=H1s, n=n, cyclic=self.cyclic, **nni_opts)


def _ham_ising(j=1.0, bx=0.0, *, S=1 / 2, cyclic=False):
    H = SpinHam(S=1 / 2, cyclic=cyclic)
    H += j, 'Z', 'Z'
    H -= bx, 'X'
    return H


def MPO_ham_ising(n, j=1.0, bx=0.0, *, S=1 / 2, cyclic=False, **mpo_opts):
    """Ising Hamiltonian in MPO form.

    Parameters
    ----------
    n : int
        The number of sites.
    j : float, optional
        The ZZ interaction strength.
    bx : float, optional
        The X-magnetic field strength.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Generate a MPO with periodic boundary conditions or not, default is
        open boundary conditions.
    mpo_opts or nni_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductOperator`.

    Returns
    -------
    MatrixProductOperator
    """
    H = _ham_ising(j=j, bx=bx, S=S, cyclic=cyclic)
    return H.build_mpo(n, **mpo_opts)


def NNI_ham_ising(n=None, j=1.0, bx=0.0, *, S=1 / 2, cyclic=False, **nni_opts):
    """Ising Hamiltonian in NNI form.

    Parameters
    ----------
    n : int
        The number of sites.
    j : float, optional
        The ZZ interaction strength.
    bx : float, optional
        The X-magnetic field strength.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Generate a NNI with periodic boundary conditions or not, default is
        open boundary conditions.
    mpo_opts or nni_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.NNI`.

    Returns
    -------
    NNI
    """
    H = _ham_ising(j=j, bx=bx, S=S, cyclic=cyclic)
    return H.build_nni(n=n, **nni_opts)


def _ham_XY(j=1.0, bz=0.0, *, S=1 / 2, cyclic=False):
    H = SpinHam(S=S, cyclic=cyclic)

    try:
        jx, jy = j
    except (TypeError, ValueError):
        jx = jy = j

    if jx == jy:
        # easy way to enforce realness
        H += jx / 2, '+', '-'
        H += jx / 2, '-', '+'
    else:
        H += jx, 'X', 'X'
        H += jy, 'Y', 'Y'

    H -= bz, 'Z'

    return H


def MPO_ham_XY(n, j=1.0, bz=0.0, *, S=1 / 2, cyclic=False, **mpo_opts):
    """XY-Hamiltonian in MPO form.

    Parameters
    ----------
    n : int
        The number of sites.
    j : float or (float, float), optional
        The XX and YY interaction strength.
    bz : float, optional
        The Z-magnetic field strength.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Generate a MPO with periodic boundary conditions or not, default is
        open boundary conditions.
    mpo_opts or nni_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductOperator`.

    Returns
    -------
    MatrixProductOperator
    """
    H = _ham_XY(j=j, bz=bz, S=S, cyclic=cyclic)
    return H.build_mpo(n, **mpo_opts)


def NNI_ham_XY(n=None, j=1.0, bz=0.0, *, S=1 / 2, cyclic=False, **nni_opts):
    """XY-Hamiltonian in NNI form.

    Parameters
    ----------
    n : int
        The number of sites.
    j : float or (float, float), optional
        The XX and YY interaction strength.
    bz : float, optional
        The Z-magnetic field strength.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Generate a NNI with periodic boundary conditions or not, default is
        open boundary conditions.
    nni_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.NNI`.

    Returns
    -------
    NNI
    """
    H = _ham_XY(j=j, bz=bz, S=S, cyclic=cyclic)
    return H.build_nni(n=n, **nni_opts)


def _ham_heis(j=1.0, bz=0.0, *, S=1 / 2, cyclic=False):
    H = SpinHam(S=S, cyclic=cyclic)

    try:
        jx, jy, jz = j
    except (TypeError, ValueError):
        jx = jy = jz = j

    if jx == jy:
        # easy way to enforce realness
        H += jx / 2, '+', '-'
        H += jx / 2, '-', '+'
    else:
        H += jx, 'X', 'X'
        H += jy, 'Y', 'Y'
    H += jz, 'Z', 'Z'

    H -= bz, 'Z'

    return H


def MPO_ham_heis(n, j=1.0, bz=0.0, *, S=1 / 2, cyclic=False, **mpo_opts):
    """Heisenberg Hamiltonian in MPO form.

    Parameters
    ----------
    n : int
        The number of sites.
    j : float or (float, float, float), optional
        The XX, YY and ZZ interaction strength.
    bz : float, optional
        The Z-magnetic field strength.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Generate a MPO with periodic boundary conditions or not, default is
        open boundary conditions.
    mpo_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductOperator`.

    Returns
    -------
    MatrixProductOperator
    """
    H = _ham_heis(j=j, bz=bz, S=S, cyclic=cyclic)
    return H.build_mpo(n, **mpo_opts)


def NNI_ham_heis(n=None, j=1.0, bz=0.0, *, S=1 / 2, cyclic=False, **nni_opts):
    """Heisenberg Hamiltonian in NNI form.

    Parameters
    ----------
    n : int
        The number of sites.
    j : float or (float, float, float), optional
        The XX, YY and ZZ interaction strength.
    bz : float, optional
        The Z-magnetic field strength.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Generate a NNI with periodic boundary conditions or not, default is
        open boundary conditions.
    nni_opts
        Supplied to :class:`~quimb.tensor.tensor_gen.NNI`.

    Returns
    -------
    NNI
    """
    H = _ham_heis(j=j, bz=bz, S=S, cyclic=cyclic)
    return H.build_nni(n=n, **nni_opts)


def MPO_ham_XXZ(n, delta, jxy=1.0, *, S=1 / 2, cyclic=False, **mpo_opts):
    """XXZ-Hamiltonian in MPO form.

    Parameters
    ----------
    n : int
        The number of sites.
    delta : float
        The Z-interaction strength.
    jxy : float, optional
        The X- and Y- interaction strength, defaults to 1.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Generate a MPO with periodic boundary conditions or not, default is
        open boundary conditions.
    mpo_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductOperator`.

    Returns
    -------
    MatrixProductOperator
    """
    return MPO_ham_heis(n, j=(jxy, jxy, delta), S=S, cyclic=cyclic, **mpo_opts)


def NNI_ham_XXZ(n=None, delta=None, jxy=1.0, *,
                S=1 / 2, cyclic=False, **nni_opts):
    """XXZ-Hamiltonian in NNI form.

    Parameters
    ----------
    n : int
        The number of sites.
    delta : float
        The Z-interaction strength.
    jxy : float, optional
        The X- and Y- interaction strength, defaults to 1.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Generate a NNI with periodic boundary conditions or not, default is
        open boundary conditions.
    nni_opts
        Supplied to :class:`~quimb.tensor.tensor_gen.NNI`.

    Returns
    -------
    NNI
    """
    if delta is None:
        raise ValueError("You need to specify ``delta``.")
    return NNI_ham_heis(n, j=(jxy, jxy, delta), S=S, cyclic=cyclic, **nni_opts)


def _ham_bilinear_biquadratic(theta, *, S=1 / 2, cyclic=False):
    H = SpinHam(S=S, cyclic=cyclic)

    H += np.cos(theta), 'X', 'X'
    H += np.cos(theta), 'Y', 'Y'
    H += np.cos(theta), 'Z', 'Z'

    XX = spin_operator('X', S=S) @ spin_operator('X', S=S)
    YY = spin_operator('Y', S=S) @ spin_operator('Y', S=S)
    ZZ = spin_operator('Z', S=S) @ spin_operator('Z', S=S)

    H += np.sin(theta), XX, XX
    H += np.sin(theta), XX, YY
    H += np.sin(theta), XX, ZZ
    H += np.sin(theta), YY, XX
    H += np.sin(theta), YY, YY
    H += np.sin(theta), YY, ZZ
    H += np.sin(theta), ZZ, XX
    H += np.sin(theta), ZZ, YY
    H += np.sin(theta), ZZ, ZZ

    return H


def MPO_ham_bilinear_biquadratic(n=None, theta=0, *, S=1 / 2, cyclic=False,
                                 compress=True, **mpo_opts):
    """ Hamiltonian of one-dimensional bilinear biquadratic chain in MPO form,
    see PhysRevB.93.184428.

    Parameters
    ----------
    n : int
        The number of sites.
    theta : float or (float, float), optional
        The parameter for linear and non-linear term of interaction strength,
        defaults to 0.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Generate a NNI with periodic boundary conditions or not, default is
        open boundary conditions.
    mpo_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductOperator`.

    Returns
    -------
    MatrixProductOperator
    """
    H = _ham_bilinear_biquadratic(theta, S=S, cyclic=cyclic)
    H_mpo = H.build_mpo(n, **mpo_opts)
    if compress is True:
        H_mpo.compress(cutoff=1e-12, cutoff_mode='rel' if cyclic else 'sum2')
    return H_mpo


def NNI_ham_bilinear_biquadratic(n=None, theta=0, *, S=1 / 2,
                                 cyclic=False, **nni_opts):
    """ Hamiltonian of one-dimensional bilinear biquadratic chain in NNI form,
    see PhysRevB.93.184428.

    Parameters
    ----------
    n : int
        The number of sites.
    theta : float or (float, float), optional
        The parameter for linear and non-linear term of interaction strength,
        defaults to 0.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Generate a NNI with periodic boundary conditions or not, default is
        open boundary conditions.
    nni_opts
        Supplied to :class:`~quimb.tensor.tensor_gen.NNI`.

    Returns
    -------
    NNI
    """
    H = _ham_bilinear_biquadratic(theta, S=S, cyclic=cyclic)
    return H.build_nni(n=n, **nni_opts)


def _ham_mbl(n, dh, j=1.0, seed=None, S=1 / 2, *, cyclic=False,
             dh_dist='s', dh_dim=1, beta=None):
    # start with the heisenberg builder
    H = _ham_heis(j, S=S, cyclic=cyclic)

    dhds, rs = _gen_mbl_random_factors(n, dh, dh_dim, dh_dist, seed, beta)

    # generate noise, potentially in all directions, each with own strength
    for i in range(n):
        dh_r_xyzs = zip(dhds, rs[:, i], 'XYZ')
        for dh, r, xyz in dh_r_xyzs:
            H[i] += dh * r, xyz

    return H


def MPO_ham_mbl(n, dh, j=1.0, seed=None, S=1 / 2, *, cyclic=False,
                dh_dist='s', dh_dim=1, beta=None, **mpo_opts):
    """The many-body-localized spin hamiltonian in MPO form.

    Parameters
    ----------
    n : int
        Number of spins.
    dh : float
        Random noise strength.
    j : float, or (float, float, float), optional
        Interaction strength(s) e.g. 1 or (1., 1., 0.5).
    seed : int, optional
        Random number to seed the noise with.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Whether to use periodic boundary conditions - default is False.
    dh_dist : {'s', 'g', 'qp'}, optional
        Whether to use sqaure, guassian or quasiperiodic noise.
    beta : float, optional
        Frequency of the quasirandom noise, only if ``dh_dist='qr'``.
    mpo_opts
        Supplied to :class:`MatrixProductOperator`.

    Returns
    -------
    MatrixProductOperator
    """
    H = _ham_mbl(n, dh=dh, j=j, seed=seed, S=S, cyclic=cyclic,
                 dh_dist=dh_dist, dh_dim=dh_dim, beta=beta)
    return H.build_mpo(n, **mpo_opts)


def NNI_ham_mbl(n, dh, j=1.0, seed=None, S=1 / 2, *, cyclic=False,
                dh_dist='s', dh_dim=1, beta=None, **nni_opts):
    """The many-body-localized spin hamiltonian in NNI form.

    Parameters
    ----------
    n : int
        Number of spins.
    dh : float
        Random noise strength.
    j : float, or (float, float, float), optional
        Interaction strength(s) e.g. 1 or (1., 1., 0.5).
    seed : int, optional
        Random number to seed the noise with.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Whether to use periodic boundary conditions - default is False.
    dh_dist : {'s', 'g', 'qp'}, optional
        Whether to use sqaure, guassian or quasiperiodic noise.
    beta : float, optional
        Frequency of the quasirandom noise, only if ``dh_dist='qr'``.
    nni_opts
        Supplied to :class:`NNI`.

    Returns
    -------
    NNI
    """
    H = _ham_mbl(n, dh=dh, j=j, seed=seed, S=S, cyclic=cyclic,
                 dh_dist=dh_dist, dh_dim=dh_dim, beta=beta)
    return H.build_nni(n, **nni_opts)


class FSAGenerator(object):

    def __init__(self, dims: int):
        """
        A finite signalling agent (FSA) MPO generator inspired by
        PRA 78 012356 (2008). Constructs sparse MPOs from diagrams.

        Parameters
        ----------
        dims : int
            The number of aux bonds in MPO (= 2 * dimensionality of the problem).
        """
        self.dims = dims
        self.signals = []
        for i in range(dims):
            self.signals.append(set())
        self.transitions = {}

    def __call__(self, *args):
        """Sets the signal to a non-zero value."""
        if len(args) != self.dims + 1:
            raise ValueError(f"Wrong number of arguments supplied: expected {self.dims + 1}, got {len(args)}")

        for i in range(self.dims):
            self.signals[i].add(args[i])

        self.transitions[args[:-1]] = args[-1]

    def __setitem__(self, key, item):
        return self(*key, item)

    def __iter__(self):
        for k in sorted(self.transitions):
            yield k, self.transitions[k]

    def __str__(self):
        return ", ".join(f"{k} = {v}" for k, v in self)

    def __repr__(self):
        return f"FSAGenerator(dims={self.dims})"

    def get_lookup(self, order=sorted):
        """
        Converts index labels into integers using
        the order specified.

        Parameters
        ----------
        order : Callable
            The ordering function.

        Returns
        -------
            Lookup dicts for every dimension.
        """
        return tuple(dict(zip(order(i), range(len(i)))) for i in self.signals)

    def el_shape(self):
        """
        Returns the common element shape.
        """

        def get_shape(e):
            if isinstance(e, np.ndarray):
                return e.shape
            elif isinstance(e, (list, tuple)):
                return np.array(e).shape
            elif isinstance(e, (str, int, float, complex)):
                return tuple()
            else:
                raise ValueError(f"Unknown type to measure the shape: {e}")

        result = set(map(get_shape, self.transitions.values()))
        if len(result) == 1:
            return result.pop()
        elif len(result) == 0:
            raise ValueError("No transitions declared: no shape can be measured")
        else:
            raise ValueError(f"Several different shapes found: {result}")

    def as_tensor(self, lookup=None):
        """
        Assembles an MPO tensor.

        Parameters
        ----------
        lookup : Iterable
            Lookup tables for every dimension.

        Returns
        -------
        A numpy tensor with the MPO.
        """
        dtype = np.array(tuple(self.transitions.values())).dtype
        result = np.zeros(tuple(len(i) for i in self.signals) + self.el_shape(), dtype=dtype)
        if lookup is None:
            lookup = self.get_lookup()
        for k, v in self.transitions.items():
            index = tuple(lookup[i][kk] for i, kk in enumerate(k))
            result[index] = v
        return result

    def same_dims(self, dim1, dim2):
        """
        Declares auxiliary indexes of an MPO to correspond
        to the same dimension by truncating dead-end states.
        This is an immediate operation.
        """
        new = self.signals[dim1] & self.signals[dim2]
        self.signals[dim1] = set(new)
        self.signals[dim2] = set(new)
        self.remove_invalid_transitions()

    def remove_invalid_transitions(self):
        """
        Finds and cleans-up dead-end states and transitions.
        """
        transaction = set()
        for k in self.transitions.keys():
            for i, signal in enumerate(k):
                if signal not in self.signals[i]:
                    transaction.add(k)
                    break
        for k in transaction:
            del self.transitions[k]

    def remove_signals(self, dim: int, *signals):
        """
        Removes specific signals from the dimension.

        Parameters
        ----------
        dim : int
            The dimension to process.
        *signals
            Signals to remove.
        """
        signals = set(signals)
        self.signals[dim] -= signals

        for k in self.transitions.keys():
            if k[dim] in signals:
                del self.transitions[k]

    def constrain_signals(self, dim, *signals):
        """
        Constrains the specific dimension to a
        subset of signals.

        Parameters
        ----------
        dim : int
            The dimension to process.
        *signals
            Signals to keep.
        """
        self.remove_signals(dim, *(set(self.signals[dim]) - set(signals)))

    def copy(self):
        """
        Returns a deep copy of self.
        """
        result = self.__class__(self.dims)
        result.signals = list(set(i) for i in self.signals)
        result.transitions = dict(self.transitions)
        return result


def hubbard_ham_mpo_tensor(n: int, blocks=None, coulomb_blocks=None, symbolic: bool = False) -> np.ndarray:
    """
    Generate the multi-band Hubbard Hamiltonian MPO tensors.

    Parameters
    ----------
    n : int
        The number of fermions.
    blocks
        Neighbour hopping amplitude blocks (the t_{ij} a^+_i a_j, i < j + H.c. term)
          * complex: nearest-neighbor hopping amplitude;
          * 1D array: on-cite energy and hopping amplitudes (from nearest to furthermost);
          * 3D array: tight-binding matrix blocks;
    coulomb_blocks
        Neighbor Coulomb blocks (the u_{ij} n_i n_j, i < j term). Same rules as for the
        `blocks` argument are applied.
    symbolic : bool
        Symbolic tensors for debugging the MPO construction.

    Yields
    ------
    MPO tensors.
    """
    # n_unit - unit cell size
    n_unit_b, n_unit_cb = None, None

    if blocks is not None:
        blocks = _hubbard_canonic_2s_form(blocks)
        n_unit_b = blocks.shape[1]

    if coulomb_blocks is not None:
        coulomb_blocks = _hubbard_canonic_2s_form(coulomb_blocks, require_real=True)
        n_unit_cb = coulomb_blocks.shape[1]

    if n_unit_b is None and n_unit_cb is None:
        raise ValueError("Neither 'blocks' nor 'coulomb_blocks' specified: the Hamiltonian is empty")
    elif n_unit_b is None:
        n_unit = n_unit_cb
        blocks = np.zeros((0, n_unit, n_unit))
    elif n_unit_cb is None:
        n_unit = n_unit_b
        coulomb_blocks = np.zeros((0, n_unit, n_unit))
    elif n_unit_b != n_unit_cb:
        raise ValueError(f"The shapes of 'blocks' and 'coulomb_blocks' are incompatible: {blocks.shape} vs {coulomb_blocks.shape}")
    else:
        n_unit = n_unit_b

    # Move the diagonal coulomb to the 1p part
    if len(coulomb_blocks) > 0:
        if len(blocks) == 0:
            blocks = np.zeros((1, n_unit, n_unit), dtype=coulomb_blocks.dtype)
        a = np.arange(n_unit)
        blocks[0, a, a] += np.diag(coulomb_blocks[0])
        coulomb_blocks[0, a, a] = 0

    # TODO: this part is here mostly for debugging; move it somewhere or create an abstraction
    if symbolic:
        op_one, op_a_, op_a, op_sz, op_n = "1", "a^", "a", "sz", "n"
        s = blocks.shape
        if s[1] == s[2] == 1:
            if s[0] > 2:
                fmt = "t_{0:d}"
            else:
                fmt = "t"
            fmt_diag = "e"
        elif s[0] == 1:
            fmt = "t_{1:d}_{2:d}"
            fmt_diag = "e_{:d}"
        else:
            fmt = "t_{0:d}_[{1:d},{2:d}]"
            fmt_diag = "e_{:d}"

        blocks_num = blocks
        blocks = np.array(
            tuple(fmt.format(*_i) if blocks_num[_i] != 0 else "" for _i in np.ndindex(*s))
        )
        blocks.shape = s
        blocks[0, range(n_unit), range(n_unit)] = tuple(fmt_diag.format(_i) if blocks_num[0, _i, _i] != 0 else "" for _i in range(n_unit))

        s = coulomb_blocks.shape
        if s[1] == s[2] == 1:
            if s[0] > 1:
                fmt = "u_{0:d}"
            else:
                fmt = "u"
        elif s[0] == 1:
            fmt = "u_{1:d}_{2:d}"
        else:
            fmt = "u_{0:d}_[{1:d},{2:d}]"

        coulomb_blocks_num = coulomb_blocks
        coulomb_blocks = np.array(
            tuple(fmt.format(*_i) if coulomb_blocks_num[_i] != 0 else "" for _i in np.ndindex(*s))
        )
        coulomb_blocks.shape = s

        scalar_product = lambda x, y: f"{{{x},{y}}}"
        dot_product = lambda x, y: f"({x},{y})"
        conj = lambda x: np.where(x == "", "", np.core.defchararray.add(x, "^"))
        is_nonzero = lambda x: x != ""

    else:
        op_one, op_a_, op_a, op_sz = map(fermion_operator, "i+-z")
        op_n = op_a_.dot(op_a)

        scalar_product = lambda x, y: x * y
        dot_product = np.dot
        conj = np.conj
        is_nonzero = lambda x: x != 0

    blocks_rview = blocks.swapaxes(0, 1).reshape((n_unit, len(blocks) * n_unit))
    coulomb_blocks_rview = coulomb_blocks.swapaxes(0, 1).reshape((n_unit, len(coulomb_blocks) * n_unit))

    # The index 1, 2 stands for the order of the operator appearing in the MPO
    op_a1 = op_a  # dot_product(op_one, op_a)
    op_a1_ = op_a_  # dot_product(op_a_, op_one)
    op_a2 = dot_product(op_sz, op_a)
    op_a2_ = dot_product(op_a_, op_sz)

    g = FSAGenerator(2)
    # TODO The MPO can be optimized by allowing different MPOs at different unit cell positions
    # Rules for constructing single-particle terms:
    # AMP    1     2          i-1   i    i+1  i+2        i+d-1  i+d i+d+1 i+d+2         n_unit
    # t_d * one x one x ... x one x a_ x sz x sz x ... x  sz  x a2 x one x one x ... x one

    # Unit cell cycle (size n_unit)
    # ------------------------
    # s0 -> s1 -> s2 -> ... -> s(n_unit-1) -> s0 "one"
    for i in range(n_unit):
        g[f"s{i}", f"s{(i + 1) % n_unit}"] = op_one

    # On-cite potentials
    # ------------------
    # Term: e_i * n_i
    # s(i) -> e "e_i * n"
    for i in range(n_unit):
        t = blocks[0, i, i]
        if is_nonzero(t):
            g[f"s{i}", "e"] = scalar_product(t, op_n)

    def s2(_op_1, _op_2, _op_3, _blocks, _prefix):
        """Two-site terms"""
        # s(i) -> path(i,1) "_op_1"
        # path(i,j) -> path(i,j+1) "_op_2"
        # path(i,j) -> e "_blocks_{i,j} * _op_3"
        for i in range(n_unit):
            g[f"s{i}", f"path_{_prefix}_{i},1"] = _op_1
            for j in range(1, _blocks.shape[1] - i - 1):
                g[f"path_{_prefix}_{i},{j}", f"path_{_prefix}_{i},{j + 1}"] = _op_2
            for j in range(1, _blocks.shape[1] - i):
                t = _blocks[i, i + j]
                if is_nonzero(t):
                    g[f"path_{_prefix}_{i},{j}", "e"] = scalar_product(t, _op_3)

    # Hoppings
    # --------
    # Term: t_{ij} a^+_i a_j, i < j
    s2(op_a1_, op_sz, op_a2, blocks_rview, "t")

    # Hoppings H.c.
    # -------------
    # Term: t_{ij}^* a^+_j a_i, i < j
    s2(op_a1, op_sz, op_a2_, conj(blocks_rview), "thc")

    # Coulomb terms
    # -------------
    # Term: u_{ij} n_i n_j, i < j
    s2(op_n, op_one, op_n, coulomb_blocks_rview, "u")

    # Tail
    # ----
    # e -> e "one"
    g["e", "e"] = op_one

    # MPO = 1D topology
    g.same_dims(0, 1)
    lookup = g.get_lookup()
    H = g.as_tensor(lookup)

    n_cites = n * n_unit

    if n_cites == 1:
        yield H[lookup[0]["s0"], lookup[1]["e"]]

    else:
        yield H[lookup[0]["s0"]]
        for i in range(n_cites - 2):
            yield H
        yield H[:, lookup[1]["e"]]


def MPO_ham_hubbard(n: int, blocks=None, coulomb_blocks=None, **kwargs):
    """
    Assembles the Hubbard Hamiltonian MPO.

    Parameters
    ----------
    n : int
        The number of fermions.
    blocks
        Neighbour hopping amplitude blocks.
          * complex: nearest-neighbor hopping amplitude;
          * 1D array: on-cite energy and hopping amplitudes (from nearest to furthermost);
          * 3D array: tight-binding matrix blocks;
    coulomb_blocks
        Neighbor Coulomb blocks (the u_{ij} n_i n_j, i < j term). Same rules as for the
        `blocks` argument are applied.
    **kwargs
        Keyword arguments to the `MatrixProductOperator` constructor.

    Returns
    -------
    MatrixProductOperator
        The Hubbard Hamiltonian MPO.
    """
    return MatrixProductOperator(arrays=hubbard_ham_mpo_tensor(n, blocks=blocks, coulomb_blocks=coulomb_blocks), **kwargs)


def MPO_ham_hubbard_canonic(n: int, u: float, mu:float, **kwargs):
    """
    Assembles the Hubbard Hamiltonian MPO (canonic version).
    The Hamiltoninan is parametrized by a single parameter U
    (hopping is assumed to be -1).

    Parameters
    ----------
    n : int
        The number of fermions.
    u : float
        The Hubbard U.
    mu : float
        The chemical potential.
    **kwargs
        Keyword arguments to the `MatrixProductOperator` constructor.

    Returns
    -------
    MatrixProductOperator
        The Hubbard Hamiltonian MPO.
    """
    return MPO_ham_hubbard(n, blocks=[-mu * np.eye(2), -np.eye(2)], coulomb_blocks=[[(0, u), (u, 0)]], **kwargs)


def MPO_fermion_number(n: int, i, **kwargs):
    """
    MPO describing single-particle density matrix elements.

    Parameters
    ----------
    n : int
        The length of MPO.
    i
        The indexes of the density matrix element desired.
        Diagonal is assumed if a single integer is specified.
    **kwargs
        Keyword arguments to the `MatrixProductOperator` constructor.

    Returns
    -------
    MatrixProductOperator
        The operator of the density matrix element.
    """
    if isinstance(i, int):
        j = i
    else:
        i, j = i

    op_one, op_a_, op_a, op_z = map(fermion_operator, "i+-z")

    if i < j:
        pos1, pos2 = op_a_.dot(op_one), op_z.dot(op_a)
    elif i > j:
        pos1, pos2 = op_one.dot(op_a), op_a_.dot(op_z)
        i, j = j, i
    else:
        pos1 = pos2 = op_a_.dot(op_a)

    op_one = op_one[np.newaxis, np.newaxis, ...]
    pos1 = pos1[np.newaxis, np.newaxis, ...]
    pos2 = pos2[np.newaxis, np.newaxis, ...]
    op_z = op_z[np.newaxis, np.newaxis, ...]

    return MatrixProductOperator(
        arrays=(
            op_one if _i < i else
            pos1 if _i == i else
            op_z if _i < j else
            pos2 if _i == j else
            op_one
            for _i in range(n)
        ),
        **kwargs
    )


def get_fermion_rdm1(state: MatrixProductState):
    """
    Retrieves the reduced single-particle density matrix from the `state`.

    Parameters
    ----------
    state : MatrixProductState
        The state to retrieve the density matrix from.

    Returns
    -------
    array
        A square array with the density matrix.
    """
    n = state.nsites
    ket = state
    bra = state.H
    result = np.zeros((n, n), dtype=state.dtype)
    for i in range(n):
        for j in range(i, n):
            mpo_n = MPO_fermion_number(n, (i, j))
            bra.align_(mpo_n, ket)
            result[i, j] = (bra & mpo_n & ket) ^ all
            result[j, i] = np.conj(result[i, j])
    return result


def MPO_fermion_total_number(n: int, **kwargs):
    """
    MPO describing the total particle number operator.

    Parameters
    ----------
    n : int
        The length of MPO.
    **kwargs
        Keyword arguments to the `MatrixProductOperator` constructor.

    Returns
    -------
    MatrixProductOperator
        The total particle number operator.
    """
    op_one, op_a_, op_a = map(fermion_operator, "i+-")
    op_n = op_a_.dot(op_a)
    if n == 1:
        return MatrixProductOperator(arrays=[op_n], **kwargs)
    g = FSAGenerator(2)
    g["s", "s"] = op_one
    g["s", "e"] = op_n
    g["e", "e"] = op_one
    g.same_dims(0, 1)
    lookup = g.get_lookup()
    N = g.as_tensor(lookup)
    return MatrixProductOperator(arrays=[N[lookup[0]["s"]]] + [N] * (n-2) + [N[:, lookup[1]["e"]]], **kwargs)


def get_fermion_n(state: MatrixProductState):
    """
    Retrieves the total number of particles.

    Parameters
    ----------
    state : MatrixProductState
        The state to calculate particles in.

    Returns
    -------
    float
        The particle number.
    """
    n = state.nsites
    ket = state
    bra = state.H
    mpo_N = MPO_fermion_total_number(n)
    bra.align_(mpo_N, ket)
    return np.real((bra & mpo_N & ket) ^ all)
