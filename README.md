# Beyond Bias Regimes: numerical experiments

This repository contains the code, configurations and saved results of the numerical
experiments in Section 8 of the paper

> Yaming GUO, Ying SUN and Hui XIONG. *Beyond Bias Regimes: A Continuous Law Governing
> Convergence and Noise Tolerance in Single-Neuron Learning.*
>
> The Hong Kong University of Science and Technology (Guangzhou). Corresponding authors: Ying SUN (sunyinggilly@gmail.com) and Hui XIONG (xionghui@ust.hk).

The paper studies population gradient descent (GD) on the squared loss for a single affine
neuron $f_\theta(x)=\sigma_\alpha(\langle w,x\rangle+b)$ with ReLU ($\alpha=0$) or Leaky ReLU
($0<\alpha<1$) activation, isotropic Gaussian inputs, and a target with bias $b^\star\le0$. Its
central quantity is the target-local signal strength $\Lambda_\alpha(b^\star)$, the smallest
eigenvalue of the Hessian of the loss at the target. The paper proves that
$\Lambda_\alpha(b^\star)$ equals the worst normalized correction toward the target over a
certified region $\mathcal A_{b^\star}=\{b\ge b^\star\|w\|,\ F_\alpha(\theta)\le CF_\alpha(0)\}$
with $C\le4/5$: the ratio

$$\zeta(\theta)=\frac{\langle\theta-\theta^\star,\nabla F_\alpha(\theta)\rangle}
{\Lambda_\alpha(b^\star)\|\theta-\theta^\star\|^2}$$

satisfies $\zeta\ge1$ on $\mathcal A_{b^\star}$, with infimum $1$. It derives from this identity
an iteration bound for
GD, a noise budget for agnostic labels, and the gain from leakage at matched bias. The
experiments test these results and ask how closely they describe the actual behaviour of GD.

## Experiments

All quantities are population expectations. By rotational invariance, every parameter reduces
to three coordinates, and the expectations are computed by deterministic Gaussian integration,
so no data are sampled. Unless stated otherwise, the grid is
$b^\star\in\{0,-0.5,-1,-2,-3,-4\}$ and $\alpha\in\{0,0.01,0.1,0.3,0.5,0.9\}$.

| Section | Experiment | Question | Figure |
|---|---|---|---|
| 8.1 | Regional lower bound | Does $\zeta\ge1$ hold throughout $\mathcal A_{b^\star}$, and is the bound attained? | `regional_bound.pdf` |
| 8.1 | Necessity of the region conditions | Is each condition defining $\mathcal A_{b^\star}$ needed? | `region_conditions.pdf` |
| 8.1 | Critical loss threshold | How much room does the hypothesis $C\le4/5$ leave? | `critical_threshold.pdf` |
| 8.2 | Convergence rate of gradient descent | Does $\Lambda_\alpha(b^\star)$ predict the speed of GD, inside and outside the region? | `convergence_rate.pdf` |
| 8.3 | Agnostic noise tolerance | How conservative is the noise budget compared with the least noise that stops GD? | `noise_tolerance.pdf` |
| 8.4 | Gains from leakage at matched bias | Does leakage improve the iteration cost and the noise tolerance by the factors that the guarantees predict? | `matched_bias.pdf` |

Where a figure in the paper shows only some leak coefficients, the full grid is in
`figures/supplementary/`.

## Main findings

- **Regional lower bound.** No evaluated point of $\mathcal A_{b^\star}$ has $\zeta<1$, and the
  smallest observed $\zeta$ decreases to $1$ as $\theta$ approaches $\theta^\star$, so the bound
  is tight only near the target.
- **Region conditions.** Without the bias condition the bound fails at every sampled distance
  from the target; without the loss condition it holds on every level set up to $4/5$ but fails
  near the zero predictor. Every counterexample that satisfies the bias condition has a loss
  ratio above $4/5$, and the smallest, $0.8169$, shows that $4/5$ is within $0.02$ of the largest
  uniform threshold.
- **Convergence rate.** In every pair, oracle projected GD and GD take half the iterations per
  decade that the Regional Gradient Descent theorem certifies at step size $1$, because near
  the target the error aligns with the weakest direction. Hence $\Lambda_\alpha(b^\star)$
  quantifies the speed of GD even far outside the ball covered by the Region-Free Gradient
  Descent theorem.
- **Noise tolerance.** The cheapest labels that trap projected GD lie above the noise budget in
  all 150 pairs. For $\alpha=0.9$ the budget is within a factor $1.35$ to $17.3$ of them; for
  ReLU the gap is large only where both noise levels are negligible.
- **Leakage at matched bias.** Leakage lowers the iteration cost of projected GD in proportion
  to the signal gain $\varrho=\Lambda_\alpha/\Lambda_0$, as it lowers the iteration bound, and it
  raises the cheapest trap within the band $[\varrho,\varrho^2]$ of the budget gain, except for a
  shortfall of at most about $\varrho^{1/4}$ at intermediate bias.

The experiments give evidence over finite grids; the universal statements are the theorems of
the paper. Every number quoted in Section 8, and the bias thresholds of the ReLU noise budget
quoted in Sections 6 and 7, are listed in `results/numbers.md`.

## Reproducing the results

The code requires Python 3.12 or later and [uv](https://docs.astral.sh/uv/). The figures are
typeset by LaTeX, so drawing them also requires a TeX installation with `amsmath` and
`amssymb`. Each command below writes its results, an audit of its numerical checks, and its
configuration and software versions to `results/`.

```bash
uv sync --frozen
uv run regional-bound --config configs/regional_bound.toml            # Section 8.1
uv run region-conditions --config configs/region_conditions.toml      # Section 8.1
uv run critical-threshold --config configs/critical_threshold.toml    # Section 8.1
uv run convergence-rate --config configs/convergence_rate.toml        # Sections 8.2 and 8.4
uv run noise-tolerance --config configs/noise_tolerance.toml          # Section 8.3
uv run matched-bias --config configs/matched_bias.toml                # Section 8.4
uv run figures                                                        # figures/
uv run report                                                         # results/numbers.md
```

The last two commands read only the saved results.

## Repository structure

```text
configs/                  settings of each experiment and of the figures
docs/
  theory.md               derivations behind the computations
  protocol.md             design, recorded quantities and verification of each experiment
src/beyond_bias_regimes/
  gaussian.py             population moments and the gap zeta - 1 without cancellation
  gradient.py             population gradient in closed form at any parameter
  precise.py              40-digit integrator used for verification
  rays.py                 weakest-direction rays
  search.py               search over directions with the scale handled analytically
  regional_bound.py       Section 8.1, regional lower bound
  region_conditions.py    Section 8.1, necessity of the region conditions
  critical_threshold.py   Section 8.1, critical loss threshold
  convergence_rate.py     Section 8.2, GD and oracle projected GD
  budget.py               noise budget
  labels.py               cheapest trap labels
  noise_tolerance.py      Section 8.3, cheapest traps against the noise budget
  matched_bias.py         Section 8.4, cheapest traps at matched bias
  records.py              result files, metadata and independent rechecks
  figures.py              figures of Section 8
  report.py               numbers quoted in Section 8
results/                  saved results
figures/                  figures of Section 8, and full grids in supplementary/
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

The code is released under the MIT License; see `LICENSE`.
