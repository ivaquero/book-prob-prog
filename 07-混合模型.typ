#import "lib/lib.typ": *
#show: qooklet.with(
  title: "混合模型",
  author: "Yāng Xīnbīn",
  footer-cap: "Yāng Xīnbīn",
  header-cap: "实用概率建模",
  lang: "zh",
)

= 混合模型

当总体是不同子总体的组合时，自然会产生混合模型。一个非常熟悉的例子是某一成年总体中的身高分布，它可被描述为女性和男性子总体的混合。若知道每个观测值属于哪个子总体，则将每个子总体作为一个独立总体建模固然很好。但当无法直接获得这些信息时，混合模型就派上用场了。

Gaussian 分布可用作许多单峰（unimodal）和对称分布的合理近似，而对多峰（multimodal）或倾斜分布，若使用 Gaussian 的混合，依然可以。在 Gaussian 混合模型中，分量是具有不同的均值 Gaussian，通常（但不一定）具有不同的标准差。通过组合 Gaussian，可增加模型的灵活性，以适应复杂的数据分布。事实上，可通过使用适当的 Gaussian 型组合来逼近任何想要的分布。分布的确切数量将取决于近似的准确性和数据的细节。实际上，核密度估计（KDE）技术正是这一思想的非 Bayesian 和非参数实现。

= 有限混合模型
<有限混合模型>

== 两种泛化分布

建立混合模型的一种方法是考虑 2 个或多个分布的有限加权混合。这就是所谓的有限混合模型。故，观测数据的概率密度是数据中$K$个子群的概率密度的加权和。

$ p(y|θ) = ∑_(i=1)^k w_i p_i (y|θ_i) $

其中，$w_i$是每个组分（或类）的权重。可将$w_i$解释为组分$i$的概率，因此其值被限制在$[0, 1]$区间内，且$∑_(i=1)^k w_i = 1$。组分$p_i (y|θ_i)$几乎可以是任何分布，如 Gaussian 或 Poisson，甚至更复杂的对象，如分层模型或神经网络。为了拟合一个有限混合模型，需要提供一个$K$的值，$K$的选取基于事先知道的真值，或是经验上的猜测。

这里我们需要先来认识两种需要用到的分布。首先是类别分布，其为Bernoulli 分布对$K$-组分的泛化，其次是 Dirichlet 分布，它是 Beta 分布的泛化。类别分布是最一般的离散分布，它使用 1 个参数来指定每个可能结果的概率，其参数$θ$的和为$1$。

#figure(
  image("images/distrs/distr_categ_pmf.png", width: 30%),
  caption: "类别分布",
  supplement: "图",
)
Dirichlet 分布存在于单纯形（simplex）中，满足

$ ∑_(i=1)^k x_i = 1 $
其 PDF 为

$ f(x) = frac(1, "Bin"(𝜶)) ∏_(i=1)^k x_i^(𝜶_i - 1) $

其中，$"Bin"(𝜶) = frac(∏_(i=1)^k Γ(𝜶_i), Γ(∑_(i=1)^k 𝜶_i))$，$𝜶 = (α_1, …, α_k)$称为浓度向量。

Dirichlet 分布就像一个$n$-维的三角形：1-单纯形是一条线，2-单纯形是一个三角形，3-单纯形是一个四面体，以此类推。设一个是概率$p$，另一个是$1 - p$。Beta 分布返回一个双元素向量$(p, 1 - p)$，但在实践中，省略了$1 - p$，因为一旦知道$p$，结果就完全决定了。若想将 Beta 分布扩展到 3 个结果，需要一个三元素向量$(p, q, r)$，其中，每个元素都是正值，由于$p + q + r = 1$，因此$r = 1 - (p + q)$。通常使用 1 个长度为$K$的$α$向量来表示这些元素。这里，可把 Beta 和 Dirichlet 看作是比例的分布。

== 模型描述

为了求解一个混合模型，需要引入一个随机变量$z$来指定某一特定观测值被分配到哪个组分中。这类变量通常被称为潜变量（latent variable）或隐变量（hidden variable），因为它不能直接被观测。

#figure(
  image("images/bbap/bap-07-gmm-2.png", width: 40%),
  caption: "有限混合模型",
  supplement: "图",
)

上图中的明显无法用单个 Gaussian 分布来正确描述，但也许 3 个或 4 个 Gaussian 分布就可以。实际上，这些数据来自于大约$40$个子总体的混合，只是它们之间有相当大的重叠。根据混合模型的直觉，可将其看作是在 Gaussian 估计模型上的一个$K$面抛硬币模型，而这里的观测变量$y$是以潜变量$z$为条件建立模型的，即$p(y|z, θ)$。可把$z$潜变量看作是一个扰动变量，可对其进行边际化，得到$p(y|θ)$。

```python
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

cs = pd.read_csv("../data/chemical_shifts_theo_exp.csv")
cs_exp = cs["exp"]

clusters = 2
with pm.Model() as model_mg:
    p = pm.Dirichlet("p", a=np.ones(clusters))
    means = pm.Normal("means", mu=cs_exp.mean(), sd=10, shape=clusters)
    sd = pm.HalfNormal("sd", sd=10)
    y = pm.NormalMixture("y", w=p, mu=means, sd=sd, observed=cs_exp)
    idata_mg = pm.sample(random_seed=123)

az.plot_trace(idata_mg, ["means", "p"])
```

#figure(
  image("images/bbap/bap-07-gmm-2-trace.png", width: 80%),
  caption: "混合模型的轨迹",
  supplement: "图",
)

#let csv1 = csv("python/bap-07-gmm-2.csv")
#figure(
  ktable(csv1, 10, inset: 0.31em),
  caption: "混合模型的抽样统计",
  supplement: "表",
  kind: table,
)

== 不可识别型

上图中，有 2 个明显不同的峰，但上表中的 2 个均值却几乎相等，这种现象被称为参数的不可识别性，又称做标签转换问题（label-switching problem），因为两个分布不论谁在左边，得到的结果都是一样的，或者说是因为其有相同的似然函数。要克服这个问题，通常有 2 种选择

- 对组分强制排序，使均值递增排列
- 加入更强的先验

对第一种方法，在分布中使用 `transform = pm.distributions.transforms.ordered` 来使均值排序。

```python
clusters = 2
with pm.Model() as model_mgp:
    p = pm.Dirichlet("p", a=np.ones(clusters))
    means = pm.Normal(
        "means",
        mu=np.array([0.9, 1]) * cs_exp.mean(),
        sigma=10,
        shape=clusters,
        transform=pm.distributions.transforms.ordered,
    )
    sd = pm.HalfNormal("sd", sigma=10)

    y = pm.NormalMixture("y", w=p, mu=means, sigma=sd, observed=cs_exp)
    idata_mgp = pm.sample(1000, random_seed=123)
```

#let csv1 = csv("python/bap-07-gmm-2p.csv")
#figure(
  ktable(csv1, 10, inset: 0.31em),
  caption: "混合模型均值排序后的统计",
  supplement: "表",
  kind: table,
)

另一个可能有用的约束条件是确保所有的组分都有 1 个非空概率，或换句话说，混合中的每个组分至少有 1 个观测值。

由之前提到的，$𝜶$控制着 Dirichlet 分布的浓度。上述模型中，$α = 1$对应的是扁平先验。$α$的值越大意味着信息量越大。经验证据表明，$α ≈ 4$或$α = 10$的值通常是一个很好的默认选择，因为这些值通常会导致每个分量至少有一个数据点的后验分布，同时减少高估分量的机会。

== 选择组分数量

有限混合模型的主要问题之一是如何决定组分的数量。一个经验法则是先从相对较少的组分数开始，然后增加组分数，以提高模型拟合度评价。改变前期模型的 `clusters`，通过 KDE 比较。

#figure(
  image("images/bbap/bap-07-gmm-k.png", width: 90%),
  caption: "不同组分数混合模型的 KDE",
  supplement: "图",
)

上图展示了总体拟合线（黑实）、平均拟合线（深蓝）、样本线（浅蓝）和平均 Gaussian 分量线（黑虚）。看起来，$K = 3$太低，$K = 4, 5, 6$可能是更好的选择。除了 KDE，还可以使用直方图，也可以计算 PPC 得到$p$值来进行预测。

#figure(
  image("images/bbap/bap-07-gmm-k-iqr.png", width: 90%),
  caption: "不同组分数混合模型的 IQR",
  supplement: "图",
)

#let csv1 = csv("python/bap-07-gmm-k.csv")
#figure(
  ktable(csv1, 10, inset: 0.31em),
  caption: "不同组分数混合模型的比较",
  supplement: "表",
  kind: table,
)

可看到$K = 6$是一个很好的选择，其 Bayesian 的$p ≈ 0.5$。而 WAIC 也发现$K = 6$是更好的模型。

聚类是统计或机器学习任务的无监督家族的一部分，是将对象进行分组的数据分析任务，使某组中的对象比其他组中的对象更接近。这些群组被称为簇（cluster），其紧密程度可通过许多不同的方式来计算，如 Euclidean 距离等。若走概率路线，则混合模型就会成为完成聚类任务的候选者。

使用概率模型允许计算每个数据点属于每个聚类的概率。这被称为软聚类（soft-clustering），而硬聚类中，每个数据点属于一个聚类的概率为$0$或$1$。可通过引入一些规则或边界，将软聚类变成硬聚类。

= 无限混合模型
<无限混合模型>

对于一些问题，当对组分的数量不确定时，可使用模型选择来帮助选择组数。但对于其他问题来说，先验地选择组数可能是一个缺点，此时我们更感兴趣的是如何对从数据中估计组分数。这类问题的概率解决方案与 Dirichlet 过程（DP）有关。

== Dirichlet 过程

目前为止，我们看到的所有模型均是参数模型。这些模型是具有固定数量的参数。自然地，也可以有非参数模型，或者叫非固定参数模型。非参数模型是理论上参数数量无限的模型。在实践中，数据会将理论上无限的参数数量减少到一些有限的数量，即，数据决定了实际的参数数量，因此非参数模型是非常灵活的。

DP 是 Dirichlet 分布的无穷维泛化。Dirichlet 分布是概率空间上的概率分布，而 DP 是分布空间上的概率分布，这意味着从 DP 中抽出的一个样本就是一个分布。对于有限混合模型，使用 Dirichlet 分布为固定数量的簇或组分配一个先验。DP 是为非固定数量的群组分配一个先验分布的方法，可把 DP 看成是从先验分布中抽样的方法。

DP 的正式定义在某种程度上是晦涩难懂的，故这里仅描述一下其一些特性，这些特性与理解它在混合模型建模中的作用有关：

- DP 是一组概率分布的分布
- DP 由 1 个基数分布$cal(H)$和 1 个正实数$α$指定，称为浓度参数
- $cal(H)$是 DP 的期望值，这意味着 DP 将围绕基分布产生分布
- 随着$α$的增加，各分量越来越不集中
- 实践中，DP 总是生成离散的分布
- $α → ∞$时，DP 的实现等于基分布，因此若基分布是连续的，DP 将产生一个连续的分布

== 断棍过程

为了使上述特性更加具体，让我们回顾一下类别分布。可通过指明$x$轴上的位置和$y$轴上的高度来完全指定这种分布。对于类别分布，$x$轴上的位置被限制为整数，而高度的总和必须为 1。

对于生成$x$轴上的位置，若选择一个 Gaussian 分布，原则上位置可是实线上的任何值；若选择一个 Beta 分布，位置将被限制在$[0, 1]$区间，若选择一个 Poisson 分布，位置将被限制在非负整数$0, 1, 2, …$。

如何选择$y$轴上的值呢？按照一个思想实验（Gedankenexperiment），即断棍过程（stick-breaking process）。想象一下，有一根长度为$1$的棍子，然后随机把它掰成两部分。把其中的一部分放在一边，然后把另一部分掰成两部分，一直这样做下去。在实践中，由于不能真正无限地重复这个过程，故将其截断在某个预定义的$K$值上。为了控制棍子的断裂过程，使用了一个参数$α$，当增加$α$的值时，将把棍子断成越来越小的部分。因此，当$lim_(α → 0)$中，我们不会断棍，而当$lim_(α → ∞)$中，我们将把棍子掰成无限的碎片。

```python
def stick_breaking_truncated(α, H, K):
    βs = stats.beta.rvs(1, α, size=K)
    weighs = np.empty(K)
    weighs = βs * np.concatenate(([1.0], np.cumprod(1 - βs[:-1])))
    locs = H.rvs(size=K)
    return locs, weighs

K = 500
H = stats.norm
αs = [1, 10, 100, 1000]

_, axes = plt.subplots(1, 4, sharex=True)

for α, ax in zip(αs, axes.flatten()):
    locs, weighs = stick_breaking_truncated(α, H, K)
    ax.vlines(locs, 0, weighs, color="C0")
    ax.set(title=f"α = {α}")
```

下图显示了在$α$的 4 个不同值的情况下，从一个 DP 中抽出的 4 次样本。可看到，DP 是一个离散分布。当$α$增加时，得到的是一个更分散的分布和更小的棍子，注意到$y$轴的比例变化，记住总长度固定在$1$。基准分布控制着位置，因为位置是从基准分布中抽取的，随着$α$的增加，DP 分布的形状越来越像基数分布$H$。在$lim_(α → ∞)$中，应该准确地得到基数分布。

#figure(
  image("images/bbap/bap-07-stick-breaking.png", width: 90%),
  caption: none,
  supplement: "图",
)

有限混合显示，若在每个数据点上放置一个 Gaussian，然后将所有 Gaussian 相加，就可近似地计算出数据的分布。使用 DP 也可以做类似的事情，但不是在每个数据点上放置一个 Gaussian，而是在 DP 分量分布的每个子棍的位置上放置一个 Gaussian，然后通过该子棍的长度来缩放或加权该 Gaussian。这个过程提供了一个无限 Gaussian 混合模型的一般公式。另外，也可将 Gaussian 替换为任何其他分布，这样就有了一个通用的无限混合模型的公式。下面，使用 Laplace 分布的混合给出一个例子。

```python
H = stats.norm
α = 10
K = 5

x = np.linspace(-4, 4, 250)
x_ = np.array([x] * K).T
locs, weighs = stick_breaking_truncated(α, H, K)
ys = stats.laplace(locs, 0.5).pdf(x_) * weighs

_, ax = plt.subplots()

ax.plot(x, np.sum(ys, 1), "C0", lw=2)
ax.plot(x, ys, "k--", alpha=0.7)
ax.set(yticks=[])
```

#figure(
  image("images/bbap/bap-07-stick-laplace.png", width: 40%),
  caption: "Laplace 混合",
  supplement: "图",
)

从数学上看，DP 的断棍过程视图可用下面的方式来表示

$ ∑_(k=1)^∞ w_k ⋅ δ_(θ_k)(θ) = f(θ) ∼ D P (α, H) $

其中

- $δ_(θ_k)$是指标函数，它将$i$评价为零，除了$δ_(θ_k)(θ_k) = 1$，这代表了从基本分布$H$中抽样的位置
- 概率$w_k$由以下公式给出

$ w_k = β_k^′ ⋅ ∏_(i=1)^(k - 1)(1 - β_i^′) $

其中

- $w_k$是一个子棍的长度
- $∏_(i=1)^(k - 1)(1 - β_i^′)$是剩余部分的长度
- $β_k^′$表示如何打断剩余的部分，$ρ_k^′ ∼ "Beta"(1, α)$，当$α$增加时$β_k^′$将平均变小

定义浓度参数$α$的先验，一个常见的选择是 Gamma 分布。

```python
def stick_breaking(α, K):
    β = pm.Beta("β", 1.0, α, shape=K)
    return β * pm.math.concatenate([[1.0], tt.extra_ops.cumprod(1.0 - β)[:-1]])

with pm.Model() as model_stick:
    α = pm.Gamma("α", 1, 1.0)
    w = pm.Deterministic("w", stick_breaking(α, K))
    means = pm.Normal(
        "means", mu=np.linspace(cs["exp"].min(), cs["exp"].max(), K), sigma=10, shape=K
    )
    sd = pm.HalfNormal("sd", sigma=10, shape=K)
    obs = pm.NormalMixture("obs", w, means, sigma=sd, observed=cs["exp"].to_numpy())
    idata_g = pm.sample(1000, tune=2000, nuts={"target_accept": 0.9})
```

从下图中可看到，$α$的值相当低，这说明描述数据所需的分量很少。因为是通过截断棍断程序来近似于无限 DP，故检查截断值$K$是否没有引入偏差是很重要的。一个简单的方法是绘制每个组分的平均权重，为了安全起见，应该有几个组分的权重可忽略不计，否则必须增加截断值。可看到，只有少数的第一个分量是重要的，因此可以相信$K = 20$对于这个模型和数据来说是足够大的。

#figure(
  image("images/bbap/bap-07-stick-k.png", width: 40%),
  caption: "不同组分数的断棍概率",
  supplement: "图",
)

= 连续混合模型
<连续混合模型>

上面的混合模型都是离散的，混合模型也有连续的。之前章节的 ZIP 回归就是 Poisson 分布和零生成过程的连续混合模型。另外，分层模型也可解释为连续混合模型，每组的参数都来自上层的连续分布。为了更具体，可考虑对几个组进行线性回归。可设每个组都有自己的斜率。实际上，对于混合模型，连续和离散的界线并不明显。

事实证明，混合分布模型具有更大的灵活性，可以更好地适应观测到的数据均值和方差。

== Beta-二项分布

当每次试验的成功概率未知时，Beta-二项分布通常用于描述 Bernoulli 试验的成功次数，并假定其服从 Beta 分布

$ "BetaBinonial"(y|n, α, β) = ∫_0^1 "Bin"(y|p, n) "Beta"(p|α, β) dd(p) $

即，为了求出观察到结果$y$的概率，我们要对所有可能的（连续的）值求均值。因此，Beta-二项分布可以被看作是一个连续的混合模型。

类似地，负二项分布可以理解为 Gamma-Poisson 混合分布。在此模型中，我们有一个 Poisson 分布的混合物，其速率参数是 Gamma 分布。这种分布经常被用来规避处理计数数据时遇到的一个常见问题，即过度分散（over-dispersion）。假设我们使用 Poisson 分布对计数数据建模，然后发现数据的方差超过了模型的方差；使用 Poisson 分布的问题在于均值和方差是相关联的。因此，解决这个问题的一种方法是将数据建模为（连续的） Poisson 分布的混合物，其比率来自 Gamma 分布。

== $t$分布

事实证明，$t$分布可被认为是一个连续的混合物。在这种情况下，有

$ t_ν (y|μ, σ) = ∫_0^∞ N (y|μ, σ) "Inv" χ^2 (σ|ν) d ν $

请注意，这与之前的负二项式表达式类似，只是这里有一个参数为$μ$和$σ$的 Gaussian 分布，以及参数为$ν$的逆$χ^2$分布，从中抽取$σ$的值，这个参数被称为自由度，或正态性参数。参数$ν$he Beta-二项分布的$p$，相当于有限混合模型的$z$潜变量。

对于一些有限混合模型，也可在推理之前对潜变量的分布进行边际化，这可能会带来一个更易抽样的模型。
