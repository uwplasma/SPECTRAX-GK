
# Inputs (TOML)

An example:
```toml
[sim]
mode = "dg"              # "fourier" or "dg"
backend = "diffrax"      # "eig" or "diffrax"
tmax = 10.0              # in units of 1/ω_p (converted to seconds)
nt = 200
nonlinear = true

[grid]
L_lambdaD = 64           # length in multiples of Debye length (λ_D of a chosen species)
Nx = 32
debye_species = "e_plus" # optional selector for λ_D & ω_p

[hermite]
N = 24

[bc]
kind = "periodic"

[[species]]
name = "e_plus"
q = -1.0
n0 = 0.5*1e19
mass_base = "electron"     # or "proton", scaled by mass_multiple
mass_multiple = 1.0
temperature_eV = 1.0
drift_c = +0.1
````

## Units & Conversions

* `tmax`: multiples of $1/\omega_p$ → converted in `run` to seconds using the chosen debye species
* `L_lambdaD`: multiples of $\lambda_D$ for that species → converted to meters
* `temperature_eV`: converted to $v_{\text{th}}$ from $T$ via $v_{\text{th}}=\sqrt{2k_BT/m}$
* `drift_c`: $u_0 = (\text{drift\_c})\, c$
* Densities `n0` read in SI (m⁻³)

You can also write safe arithmetic like `0.5*1e19` or `2*pi` in numeric fields; the loader sanitizes and evaluates them.
