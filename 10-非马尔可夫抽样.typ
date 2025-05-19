#import "lib/lib.typ": *
#show: chapter-style.with(
  title: "Non-Markov 抽样",
  info: info,
)

= Bayesian 抽样

Bayesian 方法虽然在概念上很简单，但在数学和数值上却极具挑战性。主要原因是 Bayesian 定理中的分母（边际似然），通常采用难以求解或计算成本高昂的积分形式。因此，后验通常通过抽样进行数值估计，抽样不仅可以帮助我们求和、求积分，还可以被用于生成新的样本。这些抽样方法有时被称为推理引擎（inference engines），原理上，它们能够近似任何概率模型的后验分布。

现代抽样大多使用 Markovian 方法，但对于某些问题，这些方法要么计算复杂，要么太慢。这里，我们先从非 Markovian 方法开始，看看我们能用它们做些什么。

= 网格计算
<网格计算>

网格计算（grid computing）是一种简单粗暴的方法。即使我们无法计算整个后验，我们也可能能够计算出先验和似然点。

#algo[
  设计算一个单参数模型的后验，网格逼近（grid approximation）的步骤如下：

  + 为参数定义一个合理的区间
  + 在该区间上放置一个点的网格（一般是等距离的）
  + 对于网格中的每个点，乘以似然和先验
]

我们可选择对计算值进行归一化。假设我们抛了 13 次硬币，我们观察到三个头，通过网格得到格点 `grid` 及其对应的后验 `posterior`。

```python
# grid implementation for the coin-flipping problem
def posterior_grid(grid_points=50, heads=3, tails=10):
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1 / grid_points, grid_points)  # uniform prior
    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior
```

很容易注意到，更多的点可得到更好的近似。网格方法最大的问题是，这种方法随着参数数量（维度）的增加而缩放性很差。随着维度增加，除了点的数量增加之外，参数空间中大部分后验集中的区域相比抽样量越来越小。这是统计学和机器学习中普遍存在的现象，通常被称为维度诅咒（curse of dimensionality），数学家更喜欢称之为度量集中（concentration of measure）。

#figure(
  image("images/bbap/bap-10-grid.png", width: 40%),
  caption: "网格计算",
)

#tip[
  维度诅咒被用于谈论各种只存在于高维空间中的现象。如
  - 随着维度的增加，任何一对样本之间的 Euclidean 距离都会变得越来越近
  - 对于一个超立方体来说，大部分体积都在其角上，而非在中间；对于超球来说，大部分的体积在其表面，而非在中间
  - 在高维度上，一个多元 Gaussian 分布的大部分质量并不接近均值，而是在它周围的一个壳（shell）中。随着维度的增加，这个壳从均值向尾部移动，被称为典型集（typical set）
]

显然，网格法并非一种非常明智的选择评估后验分布的位置的方法，因此使得它作为高维问题的一般方法并非很有用。

= Laplace 方法
<Laplace-方法>

Laplace 方法（quadratic approximation）也被称为二次逼近或正态逼近，包括用 Gaussian 分布$q(x)$来近似后验$p(x)$。

#algo[
  二次近似包括 2 个步骤：
  + 找到后验分布的模式，这将是$q(x)$的均值
  + 计算 Hessian 矩阵，由此可计算出$q(x)$的标准差
]

第一步可用优化方法进行数值计算，即求函数最值的方法。为此有很多现成的方法。对于 Gaussian 来说，众数和均值是相等的，我们可用众数作为近似分布的均值$q(x)$。

第二步并不则透明。我们可通过评估$q(x)$的众数/均值的曲率来近似计算$q(x)$的标准差，这可通过计算 Hessian 矩阵的平方根的倒数来完成（Hessian 的倒数提供了协方差矩阵）。

严格来说，我们只能将 Laplace 方法应用于无界变量，即$ℝ^N$中的变量。Gaussian 是一个无界分布，故若我们用它来模拟一个有界分布，如 Beta 分布，我们最终会估计出一个正密度，而事实上的密度应该是$0$，即在 Beta 分布的$[0, 1]$区间外。尽管如此，若我们首先对有界变量进行变换，使其成为无界变量，就可使用 Laplace 方法。例如，我们通常用 Half Gaussian 来模拟标准差，正是因为它被限制在$[0, ∞)$区间内，我们可通过取 Half Gaussian 变量的对数来使它成为无界变量。

Laplace 法的局限性很大，但对于某些模型可很好地发挥作用，可用于获得分析表达式来逼近后验。为克服这种限制，在其基础上发展除了一种更高级的方法，被称为集成嵌套 Laplace 逼近（Integrated Nested Laplace Approximation，INLA），这是另一个话题。

= 变分方法
<变分方法>

从概率角度来看推断，对于$hat(x)$这样的新样本，需要得到：

$ p(hat(x)|X) = ∫_θ p(hat(x), θ|X) dd(θ) = ∫_θ p(θ|X) p(hat(x)|θ, X) dd(θ) $

若新样本和数据集独立，则推断就是概率分布依参数后验分布的期望。

可以看到，推断问题的中心是参数后验分布的求解，而当参数空间无法精确求解时，则需要使用近似推断，其分为：

- 确定近似，如变分推断
- 随机近似，如 MCMC，MH，Gibbs

对大数据集或计算成本太高的后验来说，变分方法（variational methods）可能是一个比随机推断更好的选择。变分方法的一般思路是用一个更简单的分布来逼近后验分布。我们可通过解决一个优化问题来找到这个分布，这个优化问题包括在某种测量紧密性（closeness）的方法下找到与后验最接近的分布。

== KL 散度

我们可使用 KL 散度来比较模型，这将给出哪个模型更接近真实分布的后验。由于不知道真实的分布，故 KL 散度不能直接应用。于是，我们使用 KL 散度的概率形式

$ D_("KL")(q(θ) ∥ p(θ|y)) = ∫ q(θ) log frac(q(θ), p(θ|y)) d(θ) $

这里，$q$是较简单的分布，我们用它来逼近后验$p(θ|y)$，$q$通常被称为变分分布（variational distribution）。通过使用优化方法，我们试图找出$q$的参数（variational parameters），使$q$在 KL 散度方面尽可能地接近后验分布。

上述表达式中，由于后验未知，仍不能直接使用它。此时，我们需要可用其定义替换条件分布

$
  D_("KL")(q(θ) ∥ p(θ|y))
  &= ∫ q(θ) log (frac(q(θ), frac(p(θ, y), p(y)))) d(θ)\
  &= ∫ q(θ) log (frac(q(θ), p(θ, y)) p(y)) d(θ) \
  &= ∫ q(θ) log frac(q(θ), p(θ, y)) d(θ) + ∫ q(θ) log p(y) d(θ)
$

其中，$q(θ)$的积分为 1，可将$log p(y)$从积分中移出，得

$
  D_("KL")(q(θ) ∥ p(θ|y)) &= ∫ q(θ) log frac(q(θ), p(θ, y)) d(θ) + log p(y) \
  &= underbrace(-∫ q(θ) log frac(p(θ, y), q(θ)) d(θ), "evidence lower bound (ELBO)") + log p(y)
$

由于$D_("KL") ≥ 0$，则$log p(y) ≥ "ELBO"$，换句话说，证据（边际似然）总是 ≥ ELBO，这就是它名字的由来。又因为$log p(y)$是一个常数，我们只关注 ELBO。最大化 ELBO 相当于最小化 KL 散度。故，最大化 ELBO 是使$q(θ)$尽可能地接近后验$p(θ|y)$的方法。

== 平均场近似

到目前为止，我们还没有引入任何近似。原则上，$q(⋅)$是我们想要的任何东西，但在实践中，我们应该选择容易处理的分布。一个解决方案是假设高维后验可用独立的一维分布来描述，在数学上，可表示为

$ q(θ) = ∏_j q_j (θ_j) $

这就是所谓的平均场近似（mean-field approximation，MFA）。MFA 在物理学中很常见，它被用于模拟具有许多相互作用部分的复杂系统，将其作为完全不相互作用的较简单的子系统的集合，或在一般情况下，只考虑了平均的相互作用。

我们可为每个参数$θ_j$选择不同的分布$q_j$。一般来说，$q_j$分布取自指数分布族，因为它们易于处理。有了这些要素，我们已经有效地将一个推理问题变成了一个优化问题。因此，至少在概念上，我们需要做的是利用现成优化器，最大化 ELBO。

MFA 的主要缺点是，我们必须为每个模型提出一个特定的算法。我们没有一个通用推理引擎的配方。不过，最近提出一些技术可以部分解决这个问题。其中一种是自动微分变分推理（Automatic Differentiation Variational Inference，ADVI）。

#algo[
  ADVI 的主要步骤是：
  + 变换所有的有界分布，使它们在实线上生存，就像二次逼近一样
  + 用 Gaussian 分布来逼近无界参数；注意到在变换后的参数空间上的 Gaussian 是在原始参数空间上的非 Gaussian
  + 使用自动微分来最大化 ELBO
]
