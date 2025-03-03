Built-in & Random States & Operators
====================================

States
------

- :func:`~quimb.gen.states.basis_vec`
- :func:`~quimb.gen.states.up`
- :func:`~quimb.gen.states.down`
- :func:`~quimb.gen.states.plus`
- :func:`~quimb.gen.states.minus`
- :func:`~quimb.gen.states.yplus`
- :func:`~quimb.gen.states.yminus`
- :func:`~quimb.gen.states.bloch_state`
- :func:`~quimb.gen.states.bell_state`
- :func:`~quimb.gen.states.singlet`
- :func:`~quimb.gen.states.thermal_state`
- :func:`~quimb.gen.states.computational_state`
- :func:`~quimb.gen.states.neel_state`
- :func:`~quimb.gen.states.singlet_pairs`
- :func:`~quimb.gen.states.werner_state`
- :func:`~quimb.gen.states.ghz_state`
- :func:`~quimb.gen.states.w_state`
- :func:`~quimb.gen.states.levi_civita`
- :func:`~quimb.gen.states.perm_state`
- :func:`~quimb.gen.states.graph_state_1d`


Operators
---------

**Gate operators**:

- :func:`~quimb.gen.operators.pauli`
- :func:`~quimb.gen.operators.hadamard`
- :func:`~quimb.gen.operators.phase_gate`
- :func:`~quimb.gen.operators.T_gate`
- :func:`~quimb.gen.operators.S_gate`
- :func:`~quimb.gen.operators.U_gate`
- :func:`~quimb.gen.operators.rotation`
- :func:`~quimb.gen.operators.Rx`
- :func:`~quimb.gen.operators.Ry`
- :func:`~quimb.gen.operators.Rz`
- :func:`~quimb.gen.operators.phase_gate`
- :func:`~quimb.gen.operators.swap`
- :func:`~quimb.gen.operators.iswap`
- :func:`~quimb.gen.operators.controlled`
- :func:`~quimb.gen.operators.CNOT`
- :func:`~quimb.gen.operators.cX`
- :func:`~quimb.gen.operators.cY`
- :func:`~quimb.gen.operators.cZ`

**Hamiltonians and related operators**:

- :func:`~quimb.gen.operators.spin_operator`
- :func:`~quimb.gen.operators.ham_heis`
- :func:`~quimb.gen.operators.ham_heis_2D`
- :func:`~quimb.gen.operators.ham_ising`
- :func:`~quimb.gen.operators.ham_XY`
- :func:`~quimb.gen.operators.ham_XXZ`
- :func:`~quimb.gen.operators.ham_j1j2`
- :func:`~quimb.gen.operators.ham_mbl`
- :func:`~quimb.gen.operators.zspin_projector`
- :func:`~quimb.gen.operators.create`
- :func:`~quimb.gen.operators.destroy`
- :func:`~quimb.gen.operators.num`
- :func:`~quimb.gen.operators.ham_hubbard_hardcore`

Most of these are cached (and immutable), so can be called repeatedly without creating any new objects:

.. code-block:: py3

    >>> pauli('Z') is pauli('Z')
    True


Random States & Operators
-------------------------

**Random pure states**:

- :func:`~quimb.gen.rand.rand_ket`
- :func:`~quimb.gen.rand.rand_haar_state`
- :func:`~quimb.gen.rand.gen_rand_haar_states`
- :func:`~quimb.gen.rand.rand_product_state`
- :func:`~quimb.gen.rand.rand_matrix_product_state`
- :func:`~quimb.gen.rand.rand_mera`

**Random operators**:

- :func:`~quimb.gen.rand.rand_matrix`
- :func:`~quimb.gen.rand.rand_herm`
- :func:`~quimb.gen.rand.rand_pos`
- :func:`~quimb.gen.rand.rand_rho`
- :func:`~quimb.gen.rand.rand_uni`
- :func:`~quimb.gen.rand.rand_mix`
- :func:`~quimb.gen.rand.rand_seperable`
- :func:`~quimb.gen.rand.rand_iso`

All of these functions accept a ``seed`` argument for replicability:

.. code-block:: py3

    >>> rand_rho(2, seed=42)
    qarray([[ 0.196764+7.758223e-19j, -0.08442 +2.133635e-01j],
            [-0.08442 -2.133635e-01j,  0.803236-2.691589e-18j]])


    >>> rand_rho(2, seed=42)
    qarray([[ 0.196764+7.758223e-19j, -0.08442 +2.133635e-01j],
            [-0.08442 -2.133635e-01j,  0.803236-2.691589e-18j]])


For some applications, generating random numbers with ``numpy`` alone can be a bottleneck.
``quimb`` will instead peform fast, multi-threaded random number generation with `randomgen <https://github.com/bashtage/randomgen>`_ if it is installed, which can potentially offer an order of magnitude better performance. While the random number sequences can be still replicated using the ``seed`` argument, they also depend (deterministically) on the number of threads used, so may vary across machines unless this is set (e.g. with ``'OMP_NUM_THREADS'``). Use of ``randomgen`` can be explicitly turned off with the environment variable ``QUIMB_USE_RANDOMGEN='false'``.

The following gives a quick idea of the speed-ups possible. First random, complex, normally distributed array generation with a naive ``numpy`` method:

.. code-block:: py3

    >>> import numpy as np
    >>> %timeit np.random.randn(2**22) + 1j * np.random.randn(2**22)
    297 ms ± 2.09 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


And generation with ``quimb``:

.. code-block:: py3

    >>> import quimb as qu
    >>> %timeit qu.randn(2**22, dtype=complex)
    32.1 ms ± 1.39 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
