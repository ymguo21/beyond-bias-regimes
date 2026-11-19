# Experimental protocol

This note describes the design of each experiment of Section 8 of the paper, the quantities
it records, and how they are verified. The formulas are derived in `theory.md`, and every
numerical setting is in `configs/`. All quantities are population expectations computed by
deterministic Gaussian integration; no data are sampled, so the results carry no statistical
error. Unless stated otherwise, the grid is $b^\star\in\{0,-0.5,-1,-2,-3,-4\}$ and
$\alpha\in\{0,0.01,0.1,0.3,0.5,0.9\}$, with $C=4/5$.

| Section | Experiment | Configuration | Results |
|---|---|---|---|
| 8.1 | Regional lower bound | `regional_bound.toml` | `results/regional_bound/` |
| 8.1 | Necessity of the region conditions | `region_conditions.toml` | `results/region_conditions/` |
| 8.1 | Critical loss threshold | `critical_threshold.toml` | `results/critical_threshold/` |
| 8.2 | Convergence rate of gradient descent | `convergence_rate.toml` | `results/convergence_rate/` |
| 8.3 | Agnostic noise tolerance | `noise_tolerance.toml` | `results/noise_tolerance/` |
| 8.4 | Gains from leakage at matched bias | `matched_bias.toml` | `results/matched_bias/` |

## Numerical methods shared by all experiments

**Evaluation of $\zeta-1$.** Near the target ($\|\theta-\theta^\star\|^2<0.01\|\theta^\star\|^2$)
the gap is computed from the scaled residuals of `theory.md` §4, whose relative accuracy is
about $10^{-12}$ at every distance; elsewhere it is computed from the moment polynomials of
§3. Points near the target are defined by an exact displacement $(\delta,v)$ rather than by
rounded coordinates, since rounding $\theta$ would dominate gaps below $10^{-16}$.

**Independent integrators.** Reported extrema and counterexamples are recomputed by two
integrators that share no closed forms with the experiments: an adaptive SciPy integration of
the full gradient, whose resolution of $\zeta-1$ is about $10^{-13}$, and an mpmath tanh-sinh
integration at 40 digits plus the digits lost to cancellation near the target, which decides
the sign of smaller gaps. The tolerances of these comparisons are stated in the
configuration files, together with the observed agreement.

**Population gradient.** The full gradient (`theory.md` §9) is available in closed form at
every parameter, through the univariate and bivariate normal distributions; within relative
distance $0.1$ of the target, the interaction between the two gates is integrated over the
thin wedge where they disagree, without cancellation.

**Rounding allowance.** A point that lies on a constraint boundary by construction, such as a
projected iterate on the face of $\mathcal H$, is assigned to the intended side with a single
relative allowance of $10^{-12}$ (`boundary_rtol`). Decisions stated in the paper, such as
$\zeta\le1-\tau$, are applied literally.

## 8.1 Local-to-Regional Signal Identity

### Regional lower bound

The scan tests whether $\zeta\ge1$ throughout $\mathcal A_{b^\star}$ and whether the bound is
attained. Since $\mathcal A_{b^\star}$ grows with $C$, scanning $C=4/5$ covers every admissible
threshold.

- **Spheres.** Every point is $\theta^\star+\delta v$ with $\|v\|=1$. The relative distances
  $\delta/\|\theta^\star\|$ run from $10^{-10}$ to $0.1$ with six spheres per decade and from
  $0.1$ to $1.3$ in steps of $0.025$; the outermost sphere of every pair is empty.
- **Directions.** On each sphere, the angle $\chi$ from $v_{\min}$ takes the value $0$, 90
  logarithmic values from $10^{-10}$ to $0.1$ and 90 linear values up to $\pi$, and the azimuth
  takes 49 values in $[0,\pi]$. Two finer grids refine the minimum over the points of the
  region. Every evaluated point of the region counts toward the lower bound.
- **Recorded.** For each sphere, the smallest $\zeta-1$ over its points in the region, with its
  displacement, loss ratio and bias slack, and the number of evaluated points.
- **Verification.** Every sphere minimum is recomputed by both independent integrators.
- **Sharpness.** By `theory.md` §6 the sphere minima approach zero at the first-order rate of
  the weakest-direction ray; on the innermost sphere the minimum is the ray point itself.

### Necessity of the region conditions

Each condition of $\mathcal A_{b^\star}$ is removed in turn.

- **Bias condition removed.** The reverse ray $\theta^\star-\delta v_{\min}$, sampled at 101
  displacements $\delta\in[10^{-8},0.5]$, satisfies the loss condition but violates the bias
  condition; `theory.md` §5 gives its gap in closed form. The fixed example of the paper,
  $\theta=((1+\kappa/4)w^\star,-5/4)$ at $b^\star=-1$, is checked at 40 digits.
- **Loss condition relaxed.** With the bias condition kept, the smallest $\zeta-1$ is found on
  each of 29 loss levels $F_\alpha=cF_\alpha(0)$, $0.05\le c\le0.9999$. On every ray the level
  set consists of the two ends of the loss interval (`theory.md` §8), so only the two direction
  angles are searched: a $161\times161$ grid with 60 geometric offsets from the bias face,
  followed by four finer grids.
- **Verification.** Every level minimizer is recomputed by both integrators, and every
  violation found lies outside $\mathcal A_{b^\star}$.

### Critical loss threshold

The critical threshold
$\bar C_\alpha(b^\star)=\inf\{F/F(0):b\ge b^\star\|w\|,\ \zeta<1\}$ is the supremum of the
thresholds for which the conclusion of the theorem holds, so the theorem gives
$\bar C_\alpha(b^\star)\ge4/5$.

- **Grid.** $b^\star\in\{0,-0.5,-1,-2,-3,-4,-5,-6\}$ and 17 leak coefficients: $0$, twelve
  values from $10^{-5}$ to $0.05$, and $0.1,0.3,0.5,0.9$.
- **Search.** On a ray that satisfies the bias condition, the points with $\zeta\le1-\tau$ form
  an interval on which the loss is minimized in closed form (`theory.md` §7), so only the two
  direction angles are searched, on a grid refined as for the region conditions.
- **Counterexamples.** A point counts only when $\zeta\le1-\tau$ with $\tau=10^{-9}$, confirmed
  at 40 digits, since rounding cannot decide the sign of a much smaller gap. Every reported
  value is therefore attained by a counterexample and is an upper bound on
  $\bar C_\alpha(b^\star)$. Counterexamples with loss above $F(0)$ bound no threshold $C\le1$
  and are not shown in the figure.

The three experiments give evidence over finite grids; the inequality itself is established
by the proof in the paper.

## 8.2 Convergence Rate of Gradient Descent

The experiment asks whether $\Lambda_\alpha(b^\star)$ predicts, and not only bounds, the speed
of GD in the region, and whether it also sets the speed of plain GD, whose iterates may leave
the region.

- **Methods.** GD and oracle projected GD (`theory.md` §10), both at $\eta=1$, with the exact
  population gradient.
- **Starts.** 357 raw starts per pair: angles $10^\circ,20^\circ,\dots,170^\circ$ between
  $w_{\mathrm{init}}$ and $w^\star$, norms $\|w_{\mathrm{init}}\|\in\{0.5,1,2\}$, and bias
  ratios $b_{\mathrm{init}}/\|w_{\mathrm{init}}\|\in\{-3,-2,-1,-0.5,0,0.5,1\}$. GD starts at
  the raw start, and projected GD at its image under the start map.
- **Stopping.** A run stops when
  $\|\theta_t-\theta^\star\|^2\le10^{-12}\|\theta^\star\|^2$ or after $10^6$ iterations.
- **Speed.** The levels $10^{-2},10^{-4},\dots,10^{-12}$ of
  $\|\theta_t-\theta^\star\|^2/\|\theta^\star\|^2$ delimit phases; a phase runs from the last
  iterate above one level to the first iterate below the next, and its cost is its number of
  steps per decade of distance. For a run that does not converge, the window
  $10^5\le t\le10^6$ counts as a further phase. The speed of a run is the cost of its slowest
  phase, and it is compared with the certified cost $\ln10/(-\ln(1-\Lambda))$ per decade.
- **Verification.** Every step taken from $\mathcal A_{b^\star}$ must contract the squared
  distance by the certified factor $1-\Lambda$, every iterate of projected GD must lie in the
  region, and the closed-form gradient and loss are compared with the adaptive integrator at
  the first and last iterates of every 13th run.

## 8.3 Agnostic Noise Tolerance

The experiment compares the noise budget $B_\alpha(b^\star;\varepsilon)$ with
$U_\alpha(b^\star;\varepsilon)$, the least $\mathrm{OPT}$ of labels under which oracle projected
GD has a trap, a fixed point outside the $\varepsilon$-ball, while $\theta^\star$ remains a
minimizer of the loss (`theory.md` §11).

- **Grid.** 25 target biases $b^\star\in\{0,-0.25,\dots,-6\}$, the six leak coefficients, and
  $\varepsilon=10^{-2}F_\alpha(0)$: 150 pairs.
- **Cheapest labels.** For a fixed candidate trap $\hat\theta$, the trap condition and the
  stationarity of $\theta^\star$ are linear in the labels, so the cheapest labels and their
  $\mathrm{OPT}$ are available in closed form. The condition that $\theta^\star$ remain a minimizer
  reduces to the label-free constraint $E[f_{\hat\theta}^2]\le E[f_{\theta^\star}^2]$.
- **Search over traps.** Candidates lie in $\mathcal H\cap\mathcal B(\theta^\star,R)$ outside the
  $\varepsilon$-ball. Three families are searched separately: the interior, the face of
  $\mathcal H$, and the surface $E[f_{\hat\theta}^2]=E[f_{\theta^\star}^2]$, on which the cheapest
  ReLU traps lie in a valley of $\mathrm{OPT}$ too narrow for any grid. A coarse grid in
  (radius, two angles) seeds Nelder-Mead on $\log\mathrm{OPT}$ in each family, and three sweeps
  over $b^\star$ restart every pair from the cheapest traps of its neighbours.
- **Verification.** Each cheapest trap is recomputed by an independent adaptive integrator,
  and the optimality of $\theta^\star$ under its labels is checked by minimizing the loss over
  the plane $(w_\parallel,w_\perp,b)$, which suffices in every dimension.
- **Interpretation.** Adding independent zero-mean noise raises $\mathrm{OPT}$ without changing
  any gradient, so every noise level above a trap found also admits a trap. Each reported value
  is therefore an upper bound on $U_\alpha(b^\star;\varepsilon)$; a cheaper trap missed by the
  local search could only bring it closer to the budget.

## 8.4 Gains from Leakage at Matched Bias

The corollary Guarantee Comparison at Matched Bias gives the ratio
$\varrho=\Lambda_\alpha(b^\star)/\Lambda_0(b^\star)$ for the iteration bounds and the band
$\varrho\le(1-\alpha)^2B_\alpha/B_0\le\varrho^2$ for the noise budgets at a common accuracy.

- **Iterations.** The median costs of projected GD from Section 8.2, ReLU over Leaky ReLU, are
  compared with $\varrho$ through a least-squares fit of their logarithms with a free intercept.
- **Cheapest traps.** The search of Section 8.3 is repeated with the common accuracy
  $\varepsilon=10^{-2}F_0(0)$ for every $\alpha$ at each $b^\star$, which lies below $R^2$ in
  every pair because ReLU has the smallest $F(0)$. The scaled trap gain is written as
  $(1-\alpha)^2U_\alpha/U_0=\varrho^\beta$, so that it lies in the band exactly when
  $1\le\beta\le2$. Pairs with $\varrho<1.05$ are not shown, since $\beta$ amplifies small
  deviations there. Both $U_\alpha$ and $U_0$ are upper bounds, so their ratio is evidence on
  how the actual noise tolerance scales with leakage, not a bound.

## Outputs

Each directory `results/<experiment>/` holds the result tables (CSV), `audit.json` with the
verification results, `config.toml`, and `metadata.json` with the configuration, the software
versions and the running time. `results/numbers.md`, written by `uv run report`, lists every
number quoted in Section 8, together with the bias thresholds of the ReLU noise budget quoted
in Sections 6 and 7, computed from the closed form of `budget.py`.
