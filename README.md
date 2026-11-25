# Beyond Bias Regimes: Numerical Experiments

This repository reproduces the numerical experiments of the paper

> Yaming GUO, Ying SUN, and Hui XIONG. *Beyond Bias Regimes: A Continuous Law Governing
> Convergence and Noise Tolerance in Single-Neuron Learning.*

## Overview

The paper studies population gradient descent (GD) on the squared loss for a single affine
neuron $f_\theta(x) = \sigma_\alpha(\langle w, x \rangle + b)$ with parameters
$\theta = (w, b) \in \mathbb{R}^d \times \mathbb{R}$ and activation
$\sigma_\alpha(z) = \max(z, \alpha z)$, $\alpha \in [0, 1)$, which is ReLU for $\alpha = 0$ and
Leaky ReLU for $0 < \alpha < 1$. The inputs are isotropic Gaussian, $X \sim \mathcal{N}(0, I_d)$,
and the target $\theta^\star = (w^\star, b^\star)$ has $\|w^\star\| = 1$ and $b^\star \le 0$. In
the realizable setting, with population loss
$F_\alpha(\theta) = \frac{1}{2}\mathbb{E}[(f_\theta(X) - f_{\theta^\star}(X))^2]$, the central
quantity is the *target-local signal strength*
$\Lambda_\alpha(b^\star) := \lambda_{\min}(\nabla^2 F_\alpha(\theta^\star))$. The experiments
measure it against the normalized correction

$$\zeta(\theta) = \frac{\langle \theta - \theta^\star, \nabla F_\alpha(\theta) \rangle}{\Lambda_\alpha(b^\star)\|\theta - \theta^\star\|^2}, \qquad \theta \neq \theta^\star.$$

The **Local-to-Regional Signal Identity** states that, for every $0 < C \le 4/5$,
$\zeta \ge 1$ on the certified region
$\mathcal{A}_{b^\star} = \{\theta = (w, b) : b \ge b^\star\|w\|,\ F_\alpha(\theta) \le C F_\alpha(0)\}$,
with infimum $1$ over $\mathcal{A}_{b^\star} \setminus \{\theta^\star\}$. From this identity, the
paper derives iteration bounds for GD (the **Regional Gradient Descent** and **Region-Free
Gradient Descent** theorems), a closed-form noise budget $B_\alpha(b^\star; \varepsilon)$ for
reaching accuracy $\varepsilon$ under agnostic labels $Y$ (the **Closed-Form Noise Budget for
Agnostic GD** theorem), and the gains from leakage at matched bias in terms of the signal gain
$\varrho = \Lambda_\alpha(b^\star)/\Lambda_0(b^\star)$ (the **Guarantee Comparison at Matched
Bias** corollary). In the agnostic setting, $\theta^\star$ is a stationary minimizer of
$L_\alpha(\theta) = \frac{1}{2}\mathbb{E}[(f_\theta(X) - Y)^2]$ with risk
$\mathrm{OPT} = L_\alpha(\theta^\star)$. The experiments test how sharp these guarantees are and
how well they predict the actual behavior of population GD.

## Experiments

By the rotational invariance of $X$, every pair $(\theta, \theta^\star)$ reduces to three
coordinates $(w_\parallel, w_\perp, b)$ with $w_\perp \ge 0$, whatever the dimension $d$. All
expectations are computed by deterministic Gaussian quadrature, with cancellation-free wedge
integrals near the target ([docs/theory.md](docs/theory.md)), so no data are sampled. Every
reported extremum and counterexample is recomputed with an independent adaptive SciPy
integrator and with 40-digit `mpmath` integration, and GD trajectories and trap labels are
rechecked with the adaptive integrator ([docs/protocol.md](docs/protocol.md)). All checks
recorded in the `audit.json` files pass. Unless stated otherwise, the grid is
$b^\star \in \{0, -0.5, -1, -2, -3, -4\}$ and $\alpha \in \{0, 0.01, 0.1, 0.3, 0.5, 0.9\}$, with
$C = 4/5$.

| Section | Experiment | Objective | Figure |
|---|---|---|---|
| 8.1 | Regional lower bound | Check $\zeta \ge 1$ throughout $\mathcal{A}_{b^\star}$ and whether the bound becomes tight as $\theta \to \theta^\star$. | `regional_bound.pdf` |
| 8.1 | Necessity of the region conditions | Test whether $\zeta \ge 1$ fails when either condition defining $\mathcal{A}_{b^\star}$ is dropped. | `region_conditions.pdf` |
| 8.1 | Critical loss threshold | Bound $\bar{C}_\alpha(b^\star) = \inf\{F_\alpha(\theta)/F_\alpha(0) : b \ge b^\star\lVert w \rVert,\ \zeta(\theta) < 1\}$ from above and compare it with $4/5$. | `critical_threshold.pdf` |
| 8.2 | Convergence rate of GD | Test whether $\Lambda_\alpha(b^\star)$ predicts the rate of plain GD and oracle projected GD. | `convergence_rate.pdf` |
| 8.3 | Agnostic noise tolerance | Compare $B_\alpha(b^\star; \varepsilon)$ with the least $\mathrm{OPT}$ of labels that trap oracle projected GD. | `noise_tolerance.pdf` |
| 8.4 | Gains from leakage at matched bias | Compare the gains in iteration cost and trap level with the scaling in $\varrho$ that the guarantees predict. | `matched_bias.pdf` |

Where a figure shows only some leak coefficients, the full grid is in `figures/supplementary/`.
The settings of every experiment are in `configs/`.

## Main Findings

- **Regional lower bound (8.1).** None of the $34{,}335{,}252$ evaluated parameters, of which
  $21{,}557{,}898$ lie in $\mathcal{A}_{b^\star}$, has $\zeta < 1$. They lie on spheres
  $\|\theta - \theta^\star\| = \delta$ with $\delta/\|\theta^\star\| \in [10^{-10}, 1.3]$. Near
  $\theta^\star$, the minimum of $\zeta - 1$ over each sphere approaches zero at the first-order
  rate of the ray along $v_{\min}$, the eigenvector of the smallest Hessian eigenvalue, and the
  smallest value, $1.5 \times 10^{-18}$, occurs on the innermost sphere. The bound is therefore
  tight only near the target.
- **Region conditions and the threshold $4/5$ (8.1).** Without the bias condition, $\zeta < 1$
  at all $101$ sampled displacements $\delta \in [10^{-8}, 0.5]$ along the reverse ray
  $\theta^\star - \delta v_{\min}$, for every pair of the grid. With the bias condition kept, the
  minimum of $\zeta - 1$ over each level set $F_\alpha(\theta) = cF_\alpha(0)$ is positive for
  every $c \le 4/5$ but turns negative at some level $c \ge 0.94$, close to the loss of the zero
  predictor. On a finer grid of $136$ pairs $(b^\star, \alpha)$, where a point counts as a
  counterexample only when $\zeta \le 1 - 10^{-9}$ is confirmed at 40 digits, every counterexample
  that satisfies $b \ge b^\star\|w\|$ has a loss ratio above $4/5$. The smallest such ratio,
  $0.8169$ at $b^\star = -6$ and $\alpha = 5 \times 10^{-5}$, bounds $\bar{C}_\alpha(b^\star)$
  from above, so $4/5$ is within $0.02$ of the largest uniform threshold.
- **Convergence rate (8.2).** In all $36$ pairs, with $357$ starts per pair and step size
  $\eta = 1$, the median iteration cost per decade equals half the bound of the Regional Gradient
  Descent theorem to within $0.14\%$ for oracle projected GD and $0.64\%$ for the converged runs
  of plain GD, because near $\theta^\star$ the error aligns with $v_{\min}$. Plain GD converges
  from every start for $\alpha > 0$, whereas for ReLU the share of runs that fail to converge
  grows as $b^\star$ decreases, from $84/357$ at $b^\star = 0$ to $357/357$ at $b^\star = -4$.
  Hence $\Lambda_\alpha(b^\star)$ predicts the rate of GD far outside the ball covered by the
  Region-Free Gradient Descent theorem.
- **Agnostic noise tolerance (8.3).** Let $U_\alpha(b^\star; \varepsilon)$ be the least
  $\mathrm{OPT}$ of labels that make some point a trap while keeping $\theta^\star$ a minimizer of
  $L_\alpha$. At $\varepsilon = 10^{-2}F_\alpha(0)$, the cheapest trap found, an upper bound on
  $U_\alpha$, lies above $B_\alpha$ in all $150$ pairs with $b^\star \in \{0, -0.25, \dots, -6\}$,
  and the optimality of $\theta^\star$ under each set of trap labels is audited. For
  $\alpha = 0.9$, the ratio $U_\alpha/B_\alpha$ lies between $1.35$ and $17.3$. For ReLU, the
  ratio is large only where both noise levels are negligible, with
  $U_0/F_0(0) \le 6.5 \times 10^{-5}$ for $b^\star \le -3$.
- **Leakage at matched bias (8.4).** Leakage lowers the iteration cost of projected GD in
  proportion to $\varrho = \Lambda_\alpha/\Lambda_0$, with log–log slopes from $0.9999$ to
  $1.0006$ for $\varrho$ up to $1.04 \times 10^7$. At the common accuracy
  $\varepsilon = 10^{-2}F_0(0)$, write the scaled trap gain as
  $(1 - \alpha)^2 U_\alpha/U_0 = \varrho^\beta$. It stays in the predicted band
  $[\varrho, \varrho^2]$, that is, $1 \le \beta \le 2$, apart from an excess over $\varrho^2$ of
  at most $0.073\%$ and, for $\alpha \le 0.5$, a shortfall below $\varrho$ at intermediate bias.
  The shortfall is largest at $\beta = 0.75$, for $\alpha = 0.01$ and $b^\star = -4.25$, so it is
  at most about $\varrho^{1/4}$.

These experiments give numerical evidence over finite grids; the universal statements are the
theorems of the paper. Every number quoted in Section 8, together with the ReLU noise-budget
thresholds quoted in Sections 6 and 7, is listed in [results/numbers.md](results/numbers.md).

## Reproducing the Results

The code requires Python 3.12 or later and [uv](https://docs.astral.sh/uv/), and drawing the
figures also requires a LaTeX installation. The saved results were produced on Linux x86-64
with Python 3.12.3, NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.11.2, and mpmath 1.4.1.

```bash
uv sync --frozen
uv run regional-bound --config configs/regional_bound.toml            # Sec. 8.1 (~14 min)
uv run region-conditions --config configs/region_conditions.toml      # Sec. 8.1 (~3 min)
uv run critical-threshold --config configs/critical_threshold.toml    # Sec. 8.1 (~20 s)
uv run convergence-rate --config configs/convergence_rate.toml        # Secs. 8.2, 8.4 (~119 min)
uv run noise-tolerance --config configs/noise_tolerance.toml          # Sec. 8.3 (~66 min)
uv run matched-bias --config configs/matched_bias.toml                # Sec. 8.4 (~89 min)
uv run figures                                                        # figures/
uv run report                                                         # results/numbers.md
```

Each experiment writes its result tables (`*.csv`, `*.csv.gz`), together with `audit.json`,
`config.toml`, and `metadata.json`, to `results/<experiment>/`. The run times in the comments are
the wall-clock times of the saved runs on a single machine, about 4.9 h in total. The last two
commands read only the saved results, so they regenerate the figures and
[results/numbers.md](results/numbers.md) without rerunning any experiment.

## Repository Structure

```text
src/beyond_bias_regimes/  Numerical routines, one entry point per experiment, figures, and report
configs/                  Settings of the experiments and the figures
docs/                     Derivations (theory.md) and experimental protocol (protocol.md)
results/                  Saved results, audits, and numbers.md
figures/                  Figures of Section 8, with full grids in supplementary/
```

## Citation

```bibtex
@article{guo2026beyond,
  title   = {Beyond Bias Regimes: A Continuous Law Governing Convergence and Noise Tolerance in Single-Neuron Learning},
  author  = {Guo, Yaming and Sun, Ying and Xiong, Hui},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

Released under the MIT License.
