import jax.numpy as jnp
from spectraxgk.operators import StreamingOperator, LenardBernstein, ElectrostaticDrive


def test_streaming_couple_real_symmetry():
    Nn, Nm = 4, 2
    op = StreamingOperator(Nn=Nn, Nm=Nm, kpar=0.5, vth=1.0)
    X = jnp.zeros((Nn, Nm))
    X = X.at[2, 1].set(1.0)
    Y = op.couple_real(X)
    # Should only populate neighbors n=1 and n=3 in same m=1
    assert jnp.count_nonzero(Y) == 2
    assert Y[1, 1] != 0.0 and Y[3, 1] != 0.0


def test_lenard_bernstein_diagonal_damping():
    Nn, Nm = 3, 2
    lb = LenardBernstein(Nn=Nn, Nm=Nm, nu=0.5)
    C = jnp.zeros((Nn, Nm), dtype=jnp.complex64)
    C = C.at[1, 0].set(1.0 + 0.0j)
    dC = lb(C)
    # With alpha=1, beta=2 default => lambda_{1,0}=1
    assert jnp.isclose(dC[1, 0].real, -0.5)
    assert jnp.isclose(dC[1, 0].imag, 0.0)


def test_electrostatic_drive_targets_n1_m0():
    Nn, Nm = 4, 3
    drv = ElectrostaticDrive(Nn=Nn, Nm=Nm, kpar=2.0, coef=3.0)
    C = jnp.zeros((Nn, Nm), dtype=jnp.complex64)
    C = C.at[0, 0].set(2.0 + 4.0j)
    dC = drv(C)
    # E = C00/k = (2+4j)/2 = (1+2j); only (1,0) gets coef*E
    assert jnp.isclose(dC[1, 0].real, 3.0 * 1.0)
    assert jnp.isclose(dC[1, 0].imag, 3.0 * 2.0)
    assert jnp.count_nonzero(dC) == 1
