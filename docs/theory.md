# Derivations used by the experiments

This note collects the formulas behind the computations of Section 8 of the paper. Notation
follows the paper: $X\sim N(0,I_d)$, $\widetilde X=(X,1)$, $\theta=(w,b)$,
$\theta^\star=(w^\star,b^\star)$ with $\|w^\star\|=1$ and $b^\star\le0$, leak coefficient
$0\le\alpha<1$, $\Lambda=\Lambda_\alpha(b^\star)$ the smallest eigenvalue of the target
Hessian, and

$$\zeta(\theta)=\frac{\langle\theta-\theta^\star,\nabla F_\alpha(\theta)\rangle}
{\Lambda_\alpha(b^\star)\|\theta-\theta^\star\|^2},\qquad\theta\ne\theta^\star.$$

The Local-to-Regional Signal Identity states that $\zeta\ge1$ on the certified region
$\mathcal A_{b^\star}=\{b\ge b^\star\|w\|,\ F_\alpha\le CF_\alpha(0)\}$ for $0<C\le4/5$, with
infimum $1$. All experiments report the gap $\zeta-1$, computed without subtracting $1$ from a
rounded ratio.

## 1. Reduction to three coordinates

Rotational invariance of the Gaussian input lets us write every
candidate/target pair as
$\theta=(w_\parallel,w_\perp,b)$ and $\theta^\star=(1,0,b^\star)$, with
$w_\perp\ge0$. The input dimension plays no role. Directions from the origin
are $u(\psi,\beta)=(\cos\beta\cos\psi,\cos\beta\sin\psi,\sin\beta)$ with
$\psi\in[0,\pi]$ and $\beta\in[-\pi/2,\pi/2]$. The bias condition is
$\beta\ge\arctan b^\star$; the face $\beta=\arctan b^\star$ is the paper's
equal-threshold case $\hat a=a$, where $\hat a=-b/\|w\|$ is the candidate's threshold.

## 2. Target Hessian and a cancellation-free gap

The target Hessian acts on $v=(v_0,v_1,v_2)$ as
$\alpha^2\|v\|^2+(1-\alpha^2)\{\Phi(b^\star)\|v\|^2+\phi(b^\star)(-b^\star v_0^2+2v_0v_2)\}$.
Put $\kappa=2/(\sqrt{(b^\star)^2+4}-b^\star)$, so $\kappa-b^\star=1/\kappa$,
and $n=\sqrt{1+\kappa^2}$. An orthonormal eigenbasis is

$$v_{\min}=(-\kappa,0,1)/n,\qquad e_b=(1,0,\kappa)/n,\qquad e_c=(0,1,0),$$

with eigenvalues $\Lambda$, $\Lambda+(1-\alpha^2)\phi(b^\star)(\kappa+1/\kappa)$,
and $\Lambda+(1-\alpha^2)\phi(b^\star)\kappa$. Hence, for a unit vector $v$,

$$v^\top(\nabla^2F(\theta^\star)-\Lambda I)v=(1-\alpha^2)\phi(b^\star)
\bigl[(\kappa+\kappa^{-1})(v\cdot e_b)^2+\kappa v_1^2\bigr]\ge0.$$

`hessian_gap` uses this sum of nonnegative terms. It is never formed as a
difference of $O(1)$ eigenvalue terms.

## 3. Moments along a ray

For a unit direction $u$, write $U=\langle u,\widetilde X\rangle$,
$T=Z+b^\star$, and $\theta=su$. Define

$$\begin{aligned}
A&=E[\sigma_\alpha(U)^2],& B&=E[\sigma_\alpha(U)\sigma_\alpha(T)],\\
H&=E[s_\alpha(U)^2UT],& J&=E[s_\alpha(U)s_\alpha(T)T^2],\\
K&=\langle u,\theta^\star\rangle.
\end{aligned}$$

These letters are the fields of `gaussian.Coefficients`. Since $T$ is the target
preactivation, $E[\sigma_\alpha(T)^2]=2F(0)$.

Positive homogeneity gives quadratics in $s$:

$$2F(su)=As^2-2Bs+2F(0),\qquad
\langle su-\theta^\star,\nabla F(su)\rangle=As^2-(B+H)s+J,$$
$$\|su-\theta^\star\|^2=s^2-2Ks+1+(b^\star)^2.$$

Consequences used by the experiments:

- **Loss interval.** $F\le CF(0)$ holds iff $B>0$,
  $\Delta=B^2-2(1-C)AF(0)\ge0$, and $s\in[s_-,s_+]$ with
  $s_\pm=(B\pm\sqrt\Delta)/A$. The lower end is computed as
  $2(1-C)F(0)/(B+\sqrt\Delta)$.
- **Margin quadratic.** For a level $\ell$,
  $\langle\theta-\theta^\star,\nabla F\rangle-\ell\Lambda\|\theta-\theta^\star\|^2
  =(A-\ell\Lambda)s^2-(B+H-2\ell\Lambda K)s+(J-\ell\Lambda(1+(b^\star)^2))$.
  With $\ell=1$, dividing by $\Lambda\|\theta-\theta^\star\|^2$ gives the gap.
  On rays that satisfy the bias condition the gate probability is at least $\Phi(b^\star)$, so
  $A\ge\alpha^2+(1-\alpha^2)\Phi(b^\star)>\Lambda$ and the leading coefficient
  is positive for $\ell\le1$.
- **Stationary scales.** Differentiating the quotient of two quadratics gives
  one quadratic equation; its real roots, together with interval ends, contain
  the minimum of $\zeta$ along any ray segment.

## 4. Stable evaluation near the target

Let $\theta=\theta^\star+\delta v$ with $\delta>0$ and $\|v\|=1$; by symmetry
$v_1\ge0$. Write $D=v_0Z+v_1Y+v_2$ with independent standard Gaussians $Z,Y$,
so the candidate preactivation is $S=T+\delta D$. The baseline is the Hessian
quadratic form of §2. Outside gate-switching events, the normalized
correction and twice the normalized loss equal the baseline integrand. On the
switching events, change variables $Z=-b^\star+\delta u$, so that
$T/\delta=u$ without cancellation. Conditional on $u$, let
$m=v_0(-b^\star+\delta u)+v_2$ and $D=m+v_1Y$. Define
$P_\pm=\Pr(u+D\gtrless0)$ and $D_{k,\pm}=E[D^k\mathbf1\{u+D\gtrless0\}]$.
For $v_1>0$, putting $x=(u+m)/v_1$ gives

$$P_+=\Phi(x),\quad P_-=\Phi(-x),\qquad
D_{1,\pm}=mP_\pm\pm v_1\phi(x),\qquad
D_{2,\pm}=(m^2+v_1^2)P_\pm\pm v_1(m-u)\phi(x).$$

For $v_1=0$, use deterministic gate indicators. The correction and loss
changes are

$$G(u)=\begin{cases}
(1-\alpha^2)D_{2,+}+(1-\alpha)uD_{1,+},&u<0,\\
-(1-\alpha^2)D_{2,-}-\alpha(1-\alpha)uD_{1,-},&u>0,
\end{cases}$$

$$L(u)=\tfrac12\begin{cases}
(1-\alpha^2)D_{2,+}+2(1-\alpha)uD_{1,+}+(1-\alpha)^2u^2P_+,&u<0,\\
-(1-\alpha^2)D_{2,-}-2\alpha(1-\alpha)uD_{1,-}+(1-\alpha)^2u^2P_-,&u>0,
\end{cases}$$

and therefore

$$\Lambda(\zeta-1)=v^\top(\nabla^2F(\theta^\star)-\Lambda I)v+\delta\!\int G(u)\,\phi(-b^\star+\delta u)\,du,$$
$$F=\delta^2\Bigl[\tfrac12v^\top\nabla^2F(\theta^\star)v+\delta\!\int L(u)\,\phi(-b^\star+\delta u)\,du\Bigr].$$

These identities hold for every finite displacement with a positive first
weight $1+\delta v_0$; they are exact, not a Hessian approximation. The
quadrature splits at $u=0$, at the conditional switching centre
$(b^\star v_0-v_2)/(1+\delta v_0)$, and at its twelve-standard-deviation
shoulders.

Points are defined by $(\delta,v)$, not by rounded coordinates. At
$\delta=10^{-10}$, rounding $\theta$ to float64 would perturb $v$ by about
$10^{-6}$. For $b^\star=-4$, $\alpha=0.9$, that perturbation alone adds about
$10^{-16}$ to the gap, whereas the true gap is about $10^{-18}$. The bias slack
is also formed from $(\delta,v)$ without cancellation:
$b-b^\star\|w\|=\delta v_2-b^\star(\|w\|-1)$ with
$\|w\|-1=\delta(2v_0+\delta(v_0^2+v_1^2))/(\|w\|+1)$.

## 5. The two weakest-direction rays

Put $t=\delta/n$ and $\theta_\pm(\delta)=((1\mp\kappa t)w^\star,b^\star\pm t)$.
While the weight multiplier is positive, the bias slack is
$\pm t(1+b^\star\kappa)=\pm t\kappa^2$: the positive ray satisfies the bias condition, and
the reverse ray violates the bias condition at every positive displacement.
Set $p_\pm=1\mp\kappa t$, $\ell_\pm=t\kappa^2/p_\pm$,
$h_+(x)=\kappa^2+\kappa\ell_+(1-x)$, and $h_-(x)=\kappa^2-\kappa\ell_-x$. Then

$$q_+-\Lambda=\frac{\ell_+(1-\alpha)}{n^2}\int_0^1h_+(x)[\kappa^2x+\alpha h_+(x)]
\phi(-b^\star-\ell_+(1-x))\,dx>0,$$

$$q_--\Lambda=-\frac{\ell_-(1-\alpha)}{n^2}\int_0^1h_-(x)[h_-(x)+\alpha\kappa^2(1-x)]
\phi(-b^\star+\ell_-x)\,dx<0,$$

with $h_->0$ for every $t>0$. The loss identities are

$$\frac{F_+}{\delta^2}=\frac\Lambda2+\frac{\ell_+(1-\alpha)\kappa^2}{2n^2}
\int_0^1x[2\alpha h_+(x)+(1-\alpha)\kappa^2x]\phi(-b^\star-\ell_+(1-x))\,dx,$$

$$\frac{F_-}{\delta^2}=\frac\Lambda2-\frac{\ell_-(1-\alpha)\kappa^2}{2n^2}
\int_0^1(1-x)[2h_-(x)-(1-\alpha)\kappa^2(1-x)]\phi(-b^\star+\ell_-x)\,dx.$$

Their expansions are

$$\zeta_+-1=c_+\delta+O(\delta^2),\qquad
c_+=\frac{(1-\alpha)(\alpha+\tfrac12)\phi(b^\star)\kappa^6}{\Lambda n^3},$$

$$1-\zeta_-=\frac{(1-\alpha)(1+\alpha/2)\phi(b^\star)\kappa^6}{\Lambda n^3}\delta+O(\delta^2),
\qquad F_\pm=\tfrac\Lambda2\delta^2+O(\delta^3).$$

The fixed counterexample of the region-condition experiment uses $b^\star=-1$ and $t=1/4$:
$\theta=((1+\kappa/4)w^\star,-5/4)=(\tfrac{7+\sqrt5}{8}w^\star,-\tfrac54)$, with
$b-b^\star\|w\|=-\kappa^2/4<0$; the paper reports it at $\alpha=0.01$.

## 6. Prediction for the sphere envelope

Let $m(\delta)=\min\{\zeta(\theta)-1:\theta\in \mathcal A_{b^\star},\
\|\theta-\theta^\star\|=\delta\}$. By §4,
$\zeta-1=Q(v)+\delta g_1(v)+O(\delta^2)$ with
$Q(v)=\Lambda^{-1}v^\top(\nabla^2F(\theta^\star)-\Lambda I)v$ and
$g_1(v)=\Lambda^{-1}\phi(b^\star)\int G_0(u;v)\,du$, where $G_0$ is the
switching integrand at $\delta=0$.

Assume the remainder is uniform in $v$. On directions that satisfy the bias condition near
$v_{\min}$, $Q(v)\ge c\|v-v_{\min}\|^2$, and $g_1$ is Lipschitz with
constant $L$. Then

$$Q(v)+\delta g_1(v)\ge\delta g_1(v_{\min})-\frac{L^2}{4c}\delta^2,$$

while the positive ray gives $m(\delta)\le \zeta_+(\delta)-1=c_+\delta+O(\delta^2)$
with $c_+=g_1(v_{\min})$. Hence

$$\lim_{\delta\to0}\frac{m(\delta)}{c_+\delta}=1,\qquad m(\delta)\le \zeta_+(\delta)-1,$$

and the minimizing direction tilts from $v_{\min}$ by $O(\delta)$.

The regional envelope, not only one ray, therefore approaches the lower bound
at the ray's first-order rate. The direction $v_{\min}$ is interior to the
directions that satisfy the bias condition at small $\delta$, since its slack is
$\delta\kappa^2/n>0$ to first order. The scan of the regional lower bound tests this prediction. On the innermost sphere the minimum lies at $v_{\min}$ itself, so there the scan
compares the ray's $\zeta-1$ with its first-order term rather than resolving the $O(\delta)$
tilt.

## 7. Critical loss threshold along a ray

For $\tau\ge0$ the violation set on a ray that satisfies the bias condition is
$\{s>0:(A-\ell\Lambda)s^2-(B+H-2\ell\Lambda K)s+(J-\ell\Lambda(1+(b^\star)^2))\le0\}$
with $\ell=1-\tau$. This set is an interval $[s_1,s_2]\cap(0,\infty)$,
because the leading coefficient is positive (§3). The loss quadratic
$As^2-2Bs+2F(0)$ is minimized on that interval at $\mathrm{clip}(B/A,s_1,s_2)$.

Thus the least loss among points with $\zeta\le1-\tau$ reduces exactly to a
minimization over the two direction angles. Near the target, the quadratic coefficients cancel and
spurious intervals of rounding size appear. Every candidate is therefore
confirmed with the stable gap of §4 and re-sampled within its interval when
the confirmation fails.

The critical threshold is
$\bar C_\alpha(b^\star)=\inf\{F/F(0):b\ge b^\star\|w\|,\ \zeta<1\}$, the largest
threshold for which the theorem's conclusion holds, so the theorem gives
$\bar C_\alpha(b^\star)\ge4/5$. The tolerance $\tau>0$ is numerical: rounding cannot decide the
sign of a much smaller $\zeta-1$. The least loss on a ray lies at an end of its violation
interval, where $\zeta-1=-\tau$ exactly; the search therefore uses the intervals of
$\tau+10^{-12}$ (`boundary_rtol`), so that the 40-digit check $\zeta\le1-\tau$ holds
literally. Every reported value is attained by a counterexample with $\zeta\le1-\tau$ and is
therefore an upper bound on $\bar C_\alpha(b^\star)$.

## 8. Loss level sets

With the bias condition kept, the level set $F=cF(0)$, $0<c<1$, meets a ray that satisfies
the bias condition exactly at the two roots of $As^2-2Bs+2(1-c)F(0)=0$. These are the ends
$s_\pm$ of the loss interval of §3 at threshold $c$; both are positive when $B>0$ and the
discriminant is nonnegative, and otherwise the ray misses the level set. Hence
$m(c)=\min_u\min\{\zeta(s_-u),\zeta(s_+u)\}-1$ is an exact two-angle minimization. As
$c\to1$, $s_-\to0$ and $\zeta\to J/(\Lambda\|\theta^\star\|^2)$, the zero-predictor limit
along $u$.

## 9. The population gradient at arbitrary parameters

Gradient descent and the trap search need the full gradient, not only the correction
$\langle\theta-\theta^\star,\nabla F\rangle$.
Write $z=\langle\theta,\widetilde X\rangle$, $z^\star=\langle\theta^\star,\widetilde X\rangle$
and $e=\theta-\theta^\star$. Since $\sigma_\alpha(z)s_\alpha(z)=s_\alpha(z)^2z$ and
$\sigma_\alpha(z^\star)s_\alpha(z)=s_\alpha(z)s_\alpha(z^\star)z^\star$,

$$\nabla F(\theta)=\alpha^2e+(1-\alpha^2)\Sigma_+(\theta)e+\Omega(\theta),\qquad
\Sigma_+(\theta)=E[\mathbf 1\{z>0\}\widetilde X\widetilde X^\top],\quad
\Omega(\theta)=E[s_\alpha(z)(s_\alpha(z)-s_\alpha(z^\star))z^\star\widetilde X].$$

`gradient.population` evaluates the pieces as follows.

- **One gate.** With $r=\|w\|$, $a=b/r$, Gaussian integration by parts gives
  $E[\mathbf 1\{z>0\}XX^\top]=\Phi(a)I-(b\phi(a)/r^3)ww^\top$,
  $E[\mathbf 1\{z>0\}X]=(\phi(a)/r)w$ and $E[\mathbf 1\{z>0\}]=\Phi(a)$.
- **Two gates.** Far from the target,
  $\Omega=\alpha^2\theta^\star+(1-\alpha^2)\Sigma_+\theta^\star-K\theta^\star$ with
  $K\theta^\star=E[s_\alpha(z)s_\alpha(z^\star)z^\star\widetilde X]
  =\alpha^2\theta^\star+\alpha(1-\alpha)(\Sigma_+\theta^\star+E[(z^\star)_+\widetilde X])
  +(1-\alpha)^2E[\mathbf 1\{z>0\}(z^\star)_+\widetilde X]$. Let $U=(z-b)/r$,
  $c=\operatorname{corr}(U,X_1)=w_\parallel/r$, $s=\sqrt{1-c^2}=|w_\perp|/r$,
  $m=b^\star-ca$ and $n=a-cb^\star$. Integration by parts in $X$ gives
  $E[\mathbf 1\{z>0\}(z^\star)_+X]=E[\delta(z)(z^\star)_+]w+P(z>0,z^\star>0)e_1$, with
  $E[\delta(z)(z^\star)_+]=(\phi(a)/r)\{m\Phi(m/s)+s\phi(m/s)\}$, and
  $E[\mathbf 1\{z>0\}(z^\star)_+]=b^\star P(z>0,z^\star>0)+c\phi(a)\Phi(m/s)+\phi(b^\star)\Phi(n/s)$.
  The orthant probability $P(z>0,z^\star>0)=\Phi_2(a,b^\star;c)$ uses Owen's $T$
  function, with $1-c=w_\perp^2/(r(r+w_\parallel))$, $m$ and $n$ formed without
  cancellation for nearly collinear weights.
- **Near the target.** $\Omega$ vanishes unless the two gates disagree, which happens
  on a wedge of probability $O(\delta)$ where $|z^\star|=O(\delta)$, so
  $\Omega=O(\delta^2)$ while the closed form subtracts $O(1)$ moments. For
  $\|e\|<0.1\|\theta^\star\|$, `gradient.wedge` integrates over the wedge only.
  With $\theta=\theta^\star+\delta v$ and $X_1=-b^\star+\delta u$, the candidate
  preactivation given $X_1$ is $N(\delta(u+v_0X_1+v_2),(\delta v_1)^2)$, and

  $$\Omega=(1-\alpha)\delta^2\int u\,\phi(X_1)\,
  \bigl[\mathbf 1\{u\le0\}P(z>0\mid X_1)-\alpha\mathbf 1\{u>0\}P(z\le0\mid X_1)\bigr]
  (X_1,\ \cdot\ ,1)\,du,$$

  whose middle component carries $\phi((u+v_0X_1+v_2)/v_1)$ instead of the
  probability. The quadrature is that of §4, and the gradient keeps its full
  relative accuracy at every distance from the target.

## 10. Gradient descent and its oracle projected variant

Both methods stay in the plane spanned by $w^\star$ and the initial weight, so the three
coordinates of §1 are exact for every $d$. The coordinate $w_\perp$ may change sign; the loss
is even in it and the gradient odd.

- **Projection.** For $\mathcal H=\{b\ge b^\star w_\parallel\}$,
  $P_{\mathcal H}(w,b)=(w-sb^\star e_1,b+s)$ with $s=[b^\star w_\parallel-b]_+/(1+(b^\star)^2)$.
  The start map projects onto $\mathcal H$ and then pulls the point toward $\theta^\star$ into
  the ball of radius $R=\sqrt{2CF(0)}$. Since $\mathcal H\cap\mathcal B(\theta^\star,R)\subset
  \mathcal A_{b^\star}$, every iterate of projected GD lies in the region.
- **Certified contraction.** On $\mathcal A_{b^\star}$ the Local-to-Regional Signal Identity
  gives $\langle e,\nabla F\rangle\ge\Lambda\|e\|^2$ and the Gradient Norm Control lemma
  $\|\nabla F\|^2\le\langle e,\nabla F\rangle$, so a GD step from $\theta_t\in\mathcal A_{b^\star}$
  satisfies $\|\theta_{t+1}-\theta^\star\|^2\le[1-\eta(2-\eta)\Lambda]\|\theta_t-\theta^\star\|^2$.
  The projection is nonexpansive and fixes $\theta^\star$, so the same holds for projected GD.
  The experiment checks this inequality at every step taken from the region.
- **Cost per decade.** A factor $c$ per step certifies $\ln10/(-\ln c)$ iterations per decade
  of $\|\theta_t-\theta^\star\|^2$; at $\eta=1$ this is $\ln10/(-\ln(1-\Lambda))$. Near the
  target, the error of GD aligns with $v_{\min}$, along which $I-\nabla^2F(\theta^\star)$
  multiplies it by $1-\Lambda$, so the squared distance shrinks by $(1-\Lambda)^2$ per step and
  the measured cost is half the certified one. The iteration bound of the Regional Gradient
  Descent theorem relaxes $-\ln(1-\Lambda)$ to $\Lambda$, which matters only when $\Lambda$ is
  not small.

## 11. Trap labels for agnostic GD

The Closed-Form Noise Budget theorem considers labels $Y=f_{\theta^\star}(X)+\xi$ with
$\theta^\star$ a population minimizer of
$L(\theta)=\tfrac12E[(f_\theta-Y)^2]$ and $\mathrm{OPT}=L(\theta^\star)=\tfrac12E[\xi^2]$.
If $\mathrm{OPT}<B_\alpha(b^\star;\varepsilon)$, oracle projected GD with step
$\eta_\gamma=(1-\gamma)/(1+\gamma)^2$, $\gamma=\sqrt{\mathrm{OPT}/B}$, reaches
$\|\theta_t-\theta^\star\|^2\le\varepsilon$ from every start in $\mathcal H\cap \mathcal B(\theta^\star,R)$.
The experiment of Section 8.3 asks how far above $B$ the least $\mathrm{OPT}$ lies at which
some labels stop it.

**Traps.** A point $\hat\theta\in \mathcal H\cap \mathcal B(\theta^\star,R)$ is a fixed point of
$\theta\mapsto P_{\mathcal H}(\theta-\eta\nabla L(\theta))$ for every $\eta>0$ if and only if
$\nabla L(\hat\theta)=0$, or $\hat\theta$ lies on the face $b=b^\star w_\parallel$ and
$\nabla L(\hat\theta)=-\mu n$ with $\mu\ge0$ and $n=(b^\star,0,-1)/\sqrt{1+(b^\star)^2}$
the outward normal. A run started there never moves, so if
$\|\hat\theta-\theta^\star\|^2>\varepsilon$ it never reaches $\varepsilon$.

**Cheapest labels.** With $s(z)=\alpha+(1-\alpha)1\{z>0\}$,
$\nabla L(\theta)=\nabla F(\theta)-E[s(z_\theta)\widetilde X\xi]$, where $F$ is the
noise-free loss. Stationarity of $\theta^\star$ and the trap condition at $\hat\theta$ are
therefore linear in $\xi$ and see it only through $E[s(z^\star)\widetilde X\xi]$ and
$E[s(\hat z)\widetilde X\xi]$. The least $E[\xi^2]$ subject to them is attained in the span of
the six features $s(z^\star)\widehat X$ and $(s(\hat z)-s(z^\star))\widehat X$, and with the
Gram blocks $G_{11},G_{12},G_{22}$ of these two groups,
$$G_{11}\lambda_1+G_{12}\lambda_2=0,\qquad S\lambda_2=\widehat\nabla F(\hat\theta)+\mu\hat n,
\qquad \mathrm{OPT}=\tfrac12(\widehat\nabla F+\mu\hat n)^\top S^{-1}(\widehat\nabla F+\mu\hat n),$$
$S=G_{22}-G_{12}^\top G_{11}^{-1}G_{12}$, with $\mu\ge0$ minimizing the quadratic on the face
and $\mu=0$ inside $\mathcal H$. The second group vanishes outside the wedges where
$z^\star$ and $\hat z$ have different signs, so it carries the force that holds GD at
$\hat\theta$, while the first group keeps $\theta^\star$ stationary.

**Shifted basis.** On the wedges at strongly negative $b^\star$, $X_1\approx-b^\star$, so
$X_1$ and $1$ are nearly collinear and the Gram matrix in the basis $\widetilde X$ loses
all digits for ReLU below $b^\star\approx-3.75$. In $\widehat X=(z^\star,X_2,1)$,
$z^\star=X_1+b^\star$, the features stay well conditioned. Vectors change by
$\hat v=(v_1,v_2,v_3-b^\star v_1)$, gradients (dual vectors) by
$\hat g=(g_1+b^\star g_3,g_2,g_3)$.

**Atoms.** Every moment is a weighted sum of $A_p=E[1\{p\}\widehat X\widehat X^\top]$ over
the on/off patterns $p$ of the gates involved. For fixed $X_2$ each gate is a half-line in
$X_1$, so $A_p$ is an integral over $X_2$ of truncated Gaussian moments of an interval,
in closed form (or Gauss-Legendre in $z^\star$ on intervals shorter than 1). The
$X_2$-integral uses 12-point Gauss-Legendre on segments split wherever an interval end
crosses $z^\star\in\{0,\pm1,\pm3,\pm6,\pm10\}$. Wedge atoms are integrated directly,
never formed as differences. `labels.reference` integrates in the other order, adaptively
in $z^\star$ with the $X_2$-moments in closed form.

**Optimality of $\theta^\star$.** For any $\xi$ that keeps both $\theta^\star$ and
$\hat\theta$ stationary in the sense above, homogeneity of $f_\theta$
($\langle\theta,\nabla_\theta f_\theta\rangle=f_\theta$) gives
$E[f_\theta(f_\theta-Y)]=\langle\theta,\nabla L(\theta)\rangle$, which is $0$ at
$\theta^\star$ and $-\mu\langle\hat\theta,n\rangle=0$ at $\hat\theta$ (the face passes
through the origin). Hence
$$L(\hat\theta)-L(\theta^\star)=\tfrac12\big(E[f_{\theta^\star}^2]-E[f_{\hat\theta}^2]\big),$$
independent of $\xi$. The search keeps only traps with $E[f_{\hat\theta}^2]\le
E[f_{\theta^\star}^2]$, a label-free constraint (traps with equality are limits of traps
with strict inequality, at the same $\mathrm{OPT}$); it is evaluated as
$-\tfrac12E[(f_{\hat\theta}-f_{\theta^\star})(f_{\hat\theta}+f_{\theta^\star})]$, free of
cancellation. Global optimality of $\theta^\star$ is audited over the plane
$(w_\parallel,w_\perp,b)$ with $w_\perp$ of either sign. That suffices for every $d$: the
labels depend on $X$ only through $(X_1,X_2)$, and a weight component $w_3$ orthogonal to
the plane enters as $\langle w,X\rangle+b=w_1X_1+w_2X_2+(b+w_3X_3)$, so
$L(w_1,w_2,w_3,b)=E_{X_3}L(w_1,w_2,b+w_3X_3)$ is an average of in-plane losses, no smaller
than their minimum.

**Above the trap level.** Adding independent zero-mean noise to the cheapest labels raises
$\mathrm{OPT}$ and changes no gradient. So every
$\mathrm{OPT}$ at or above the least one found admits labels and a start from which GD never
reaches $\varepsilon$, and the computed minimum $U$ is an upper bound on the least trapping
noise. Since the theorem holds for every label distribution, $U\ge B$ for every trap.
