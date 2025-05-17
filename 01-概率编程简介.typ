#import "lib/lib.typ": *
#show: chapter-style.with(
  title: "概率编程简介",
  info: info,
)

= 概率编程
<概率编程>

通常，概率编程指 Bayesian 推断，其思想是对不断模型提出合理的质疑，在最小化质疑过程中，得到最优模型，其遵循以下流程

#block(
  height: 3em,
  columns(3)[
    + 定义样本的生成模型
    + 定义具体的估计量
    + 设计统计方法来产生估计量
    + 对步骤 3. 进行测试
    + 分析样本，并总结
  ],
)

由 Bayes' 法则

$
  ctext("后验") = frac(p(ctext("原因i")) × p(ctext("现象|原因i")), p(ctext("现象"))) = ctext("先验") × ctext("标准化的似然")
$

数学表达式可写作

$ p(θ|y) = frac(p(θ)p(y|θ), p(y)) ∝ p(y|θ) p(θ) $

= 抛硬币模型
<抛硬币模型>

考虑抛硬币问题，假设

- 只有 2 种可能的结果，正面或反面，且每次设抛硬币是相互独立的随机事件
- 所有抛硬币的概率都来自于同一个分布

考虑到这些假设，似然的一个很好的候选是二项分布。同时，使用 Beta 分布作为先验，理由在此前的章节中已经述及，即

+ Beta 分布取值在$0$到$1$之间，是标准均匀分布的泛化
+ Beta 分布是二项分布的共轭先验

于是有

$ p(y|θ, N) = frac(N!, y!(N - y)!) θ^y (1 - θ)^(N - y) $

$ p(θ) = frac(Γ(α + β), Γ(α) Γ(β)) θ^(α - 1)(1 - θ)^(β - 1) $

#sgrid(
  figure(
    image("images/distrs/distr_bin_pmf.png", width: 90%),
    caption: "二项分布",
  ),
  figure(
    image("images/distrs/distr_beta_pdf.png", width: 90%),
    caption: "Beta 分布",
  ),
  columns: (200pt,) * 2,
  gutter: 2pt,
  caption: "二项分布与Beta分布",
)

对许多问题来说，我们经常会知道一个参数可取的大致范围，或我们期望该值接近于或低于或高于某个值。在这种情况下，我们可使用先验在我们的模型中放入一些弱信息。因为这些先验的作用是使后验分布保持在一定的合理边界内，故也被称为正则先验（regularizing priors）。若有高质量的信息来定义这些先验，使用信息先验（informative prior）当然更好。

简单说，先验可让模型表现得更好，具有更好的泛化特性。实际上，每个模型不管是否 Bayesian 模型，都有某种先验，即便是由频率学派统计产生的。这些先验在某些情况下可看成是 Bayesian 模型的特例，如扁平先验。

根据 Bayes' 法则、似然和 Beta 分布的定义，可得

$
  p(θ|y) ∝ frac(N!, y!(N - y)!) θ^y(1 - θ)^(N - y) frac(Γ(α + β), Γ(α) Γ(β)) θ^(α - 1)(1 - θ)^(β - 1)
$

出于实际考虑，丢弃所有不依赖$θ$​的项，得

$ p(θ|y) ∝ θ^(y + α - 1)(1 - θ)^(N - y + β - 1) $

即

$ p(θ|y) ∼ "Beta"(α_"prior" + y, β_"prior" + N - y) $

不难看出，先验的分布与参数值的不确定性成正比。分布越分散，确定性越低。给予足够大的数据量，两个或多个具有不同先验的 Bayesian 模型将趋向于相同的结果。

== 获取后验

#figure(
  image(
    "images/bbap/bap-01-coin-post.png",
    width: 60%,
  ),
  caption: "Beta 分布后验",
)

- 均匀分布的先验（蓝），代表了对后验一无所知，即所有可能的偏差值是同等可能的先验
- Gaussian 先验（红）代表着公平硬币的先验
- 偏斜先验（绿）代表着偏向反面的硬币的先验
- 在 0.35 处的黑线，代表了$θ$的真实值。但在实际问题中，真实值往往是未知的\

在上图中可看到，来自均匀先验和偏斜先验的后验更快地收敛到几乎相同的分布，而 Gaussian 先验的后验则需要更长的时间。Bayesian 分析的结果是一个后验分布，是一个给定数据和模型的可信值的分布。最有可能的值由后验的众数（分布的峰值）给出。这种估计方法被称作最大后验估计（maximum a posteriori estimate，MAP）。

#tip[
  频率学派统计常用的最大似然估计（MLE）通过计算每个原因的似然，然后选择似然最大的原因。其只考虑了原因的解释力，忽视了该原因出现的可能性，故而有时会得出与事实不符的结果。
]

== 归纳后验

一个常用的总结后验分布的方法是使用最高后验密度（highest posterior density，HPD），其对应的区间称最高密度区间（highest density interval，HDI），是包含给定部分概率密度的最短区间。其中最常用的是 95% 的 HDI。若我们说某项分析的 95% HDI 是$[2, 5]$，意味着根据我们的数据和模型，我们认为有关参数在$[2, 5]$之间的概率为 95% 。

#figure(
  image("images/bbap/bap-01-coin-hdi.png", width: 40%),
  caption: "HPD",
)

#warning[
  请注意，HDI 区间不等于置信区间。
]

== 利用后验预测

一旦有了一个后验$p(θ|y)$，就可根据数据$y$和估计的参数$θ$，生成预测$hat(y)$。

$ p(hat(y)|y) = ∫p(hat(y)|θ) p(θ|y) dd(θ) $

故，后验预测分布是条件预测对后验分布的均值。概念上，可将此积分近似为一个两步迭代过程。

+ 从后验$p(θ|y)$中抽取一个值$θ$
+ 将该值反馈给似然，从而得到一个数据点$hat(y)$
\
当需要进行预测时，可使用生成的预测$hat(y)$。也可用它们来调试模型，通过比较观察到的数据$y$和预测到的数据$hat(y)$来发现这两组数据之间的差异，这就是所谓的后验预测核查（posterior predictive checks，PPC）。主要目标是检查自一致性（auto-consistency）。

= 编程示例
<编程示例>

== 描述后验

上面描述的模型，很容易转化为 PyMC 代码。

```python
import numpy as np
import pymc as pm
from scipy import stats

np.random.default_rng(123)
trials = 4
θ_real = 0.35
data = stats.bernoulli.rvs(p=θ_real, size=trials)

with pm.Model() as coin_flip:
    θ = pm.Beta("θ", alpha=1.0, beta=1.0)
    y = pm.Bernoulli("y", p=θ, observed=data)
    idata_coin = pm.sample(1000, random_seed=123)
```

通过 #raw("az.plot_trace(idata_coin, compact=False)", lang: "python", block: false) 得到了每个未观测变量的两个子图。在我们的模型中，唯一一个未观测的变量是$θ$，而$y$是代表数据的观测变量，不需要对其抽样。左图是核密度估计（kernel density estimation，KDE）图，我们希望每条链（chain）的 KDE 是相似的；在右图可得到了抽样过程中每一步的各个抽样值。

#figure(
  image("images/bbap/bap-02-coin-trace.png", width: 100%),
  caption: "plot_trace()",
)

使用 #raw("az.plot_trace(idata_coin, kind=\"rank_bars\", combined=True)", lang: "python", block: false) 可得到一个排序图（rank plot），这是检查样本可信度的另一种方法，我们为每个链获取一个直方图，并希望所有直方图尽可能均匀。当均匀性出现较大偏差则表明链正在探索后验的不同区域。理想情况下，我们希望所有链都能探索整个后验。

#figure(
  image("images/bbap/bap-02-coin-trace-rank.png", width: 100%),
  caption: "plot_trace(kind=\"rank_bars\")",
)

通过 #raw("az.summary(idata, kind=\"stats\").round(2)", lang: "python", block: false) 可以得到得到均值、标准差和 94% 的 HDI，这里使用 94% 是因为这是对 95% 值的任意性友好剩余。可通过向参数 `hdi_prob` 传递一个不同的值来改变这个值。

#let csv1 = csv("python/bap-02-coin.csv")
#figure(
  tableq(csv1, 5, inset: 0.31em),
  caption: "后验描述",
  supplement: "表",
  kind: table,
)

我们可以使用标准差报告类似的摘要。标准差相对于 HDI 的优势在于它是一种更受欢迎的统计数据。缺点是，我们必须更加谨慎地解释它；否则，它会导致毫无意义的结果。例如，若我们计算均值$±2$个标准差，我们将得到区间$(-0.02, 0.7)$；上限与我们从 HDI 获得的$0.65$相差不大，但下限实际上超出了$θ$的可能值。

= 后验决策

有时，描述后验是不够的，需要根据推论再做决定。我们必须将连续估计还原到二元估计：健康 - 疾病，污染 - 安全，等等。当决定硬币是否公平，可将$0.5$与 HDI 进行比较。@coin-post 中，HDI 为 `[0.03, 0.7]`，包含 0.5，我们可以将此解释为硬币可能存在尾部偏差，但不能完全排除硬币实际上是公平的可能性。若我们想要做出更明智的决定，则需要收集更多数据来减少后验区间，或者需要定义更具信息量的先验。

== 实际等价区

通常，我们关心的不关心不是精确结果，而是一定范围内的结果，如在$[0.45, 0.55]$这个区间内的任何值，我们将这个区间称为实际等价区（region of practical equivalence，ROPE）。将其与 HDI 比较，至少可得到三种情况：

- ROPE 与 HDI 不重叠，则可说硬币不公平
- ROPE 包含了整个 HDI，则可说硬币是公平的
- ROPE 与 HDI 部分重叠，则不能说硬币是公平或不公平

```python
_, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

az.plot_posterior(idata_coin, ax=axes[0])
az.plot_posterior(idata_coin, rope=[0.45, .55], ax=axes[1])
az.plot_posterior(idata_coin, ref_val=0.5, ax=axes[2])
```

ROPE 的定义是取决于上下文的，决定向来是主观的，我们的任务是根据我们的目标，做出最明智的决定。下图中，ROPE 显示为一条半透明的粗线。`plot_posterior()` 默认显示离散变量的直方图和连续变量的 KDE，还可得到分布的均值（使用 `point_estimate` 参数求取中数或众数）和`94%` 的 HDI，在图底用黑线表示。除此之外，也可选择将后验与参考值进行比较。

#figure(
  image(
    "images/bbap/bap-02-coin-posterior.png",
    width: 80%,
  ),
  caption: "评估后验",
) <coin-post>

== Savage-Dickey 密度比

评估后验对给定值的支持程度的另一种方法是比较该值处的后验密度和先验密度的比率。这称为 Savage-Dickey 密度比，我们可以使用 #raw("az.plot_bf()", lang: "python", block: false) 计算它。

```python
az.plot_bf(idata_coin, ref_val=0.5, var_name="θ", prior=np.random.uniform(0, 1, 10000))
```

#figure(
  image("images/bbap/bap-02-coin-trace-bf.png", width: 40%),
  caption: "Savage-Dickey 密度比",
) <coin-bf>

由@coin-bf，我们可以看到`BF_01 = 1.3`，这意味着`θ = 0.5`的值在后验分布下的可能性比在先验分布下的可能性高 1.3 倍。要计算这个值，我们只需将`θ = 0.5`处的后验高度除以`θ = 0.5`处的先验高度。`BF_10 = 1/1.3 ≈ 0.8`。我们可以将其视为`θ ≠ 0.5`的值在后验下的可能性比在先验下的可能性高 0.76 倍。Savage-Dickey 密度比是计算所谓贝叶斯因子的一种特殊方法 @kassBayesFactors1995，我们将在【模型选择】一章中详细谈论它。

#let data = csv("data/bayes-factor.csv")
#figure(
  tableq(data, 2),
  caption: "Bayes 因子的表述",
  supplement: [表],
  kind: table,
)

== 损失函数
<损失函数>

为了做出一个好的决策，对相关参数的估计量有尽可能高的精度是很重要的，但也要考虑到犯错误的成本。成本 - 收益的权衡可用损失函数在数学上度量。关键思想是使用一个函数来捕捉参数的真实值和估计量的差异。损失函数的值越大，估计量越差。一些常见的损失函数的例子有：

- 二次方损失，$(θ - hat(θ))^2$
- 绝对损失，$|θ - hat(θ)|$
- 0-1 的损失，$I(θ ≠ hat(θ))$，其中，$I$为指示函数

#sgrid(
  figure(
    image("images/bbap/bap-02-coin-loss.png", width: 90%),
    caption: "损失函数",
  ),
  figure(
    image("images/bbap/bap-02-coin-loss2.png", width: 90%),
    caption: "混合损失函数",
  ),
  <loss-func>,
  columns: (200pt,) * 2,
  gutter: 2pt,
  caption: "损失函数",
)

实践中，我们一般没有参数的真实值，而只有一个后验分布形式的估计。故，我们能做的就是找出最小化预期损失函数的值。所谓预期损失函数，通常指的是整个后验分布的平均损失函数。
可以看到，`lossf_mae` 和 `lossf_mse` 的结果有些相似，两者区别在于，前者=后验的中数，而后者=后验的均值。明确选择损失函数的优点是，我们可以根据问题定制该功能。通常，做出决定的成本是不对称的。因此，我们可以构建一个不对称的损失函数，如@loss-func 所示。

```python
lossf = []
for i in grid:
    if i < 0.5:
        f = 1 / np.median(θ_pos / np.abs(i**2 - θ_pos))
    else:
        f = np.mean((i - θ_pos) ** 2 + np.exp(-i)) - 0.25
    lossf.append(f)
```

= Gaussian 推断

核磁共振（NMR）允许测量与有趣的不可观察分子特性相关的不同类型的可观察量，这些可观测量之一称为化学位移（chemical shifts）。在这些例子中，变量是连续的，将它们视为均值加上离散值是有意义的。有时，若可能值的数量足够大，我们可以对离散变量使用 Gaussian 模型。现在，我们有 48 个化学位移值。由@box，可以看到中位数（箱内的线）在 53 左右，四分位距（箱线）在 52 和 55 左右。还可以看到有 2 个值远离其余数据（空心圆）。

#figure(
  image("images/bbap/bap-02-chem-box.png", width: 40%),
  caption: "化学位移",
) <box>\

由于不知道均值或标准差，我们必须为它们设定先验。因此，一个合理的模型可能是：

$
  mu &~ 𝒱(l, h) \
  sigma &~ ℋ 𝒩(sigma_sigma) \
  Y &~ 𝒩(mu, sigma)
$

对应的代码为

```python
with pm.Model() as model_g:
    μ = pm.Uniform("μ", lower=40, upper=70)
    σ = pm.HalfNormal("σ", sigma=10)
    y = pm.Normal("y", mu=μ, sigma=σ, observed=data)
    idata_g = pm.sample(1000)
```

我们可以使用#raw("az.plot_pair()", lang: "python", block: false)来查看二维后验分布以及$μ$和$σ$的边际分布。

#figure(
  image("images/bbap/bap-02-chem-pair.png", width: 40%),
  caption: none,
)

== 后验概率检查

一旦我们有了后验概率$p(θ∣Y)$，就可以使用它来生成预测$p(tilde(Y))$。从数学上讲，这可以通过计算来实现：

$
  p(tilde(Y)|Y) = ∫ p(tilde(Y)|θ) p(θ|Y) d θ
$

这种分布被称为后验预测分布。我们可以把它看作是给定模型和观察到的数据的未来数据分布。后验预测分布的一个常见用途是执行后验预测检查（posterior predictive check，PPC）。这是一组测试，可用于检查模型是否适合数据。我们可以使用 #raw("az.plot_ppc(idata_g, num_pp_samples=100)", lang: "python", block: false)来可视化后验预测分布和观察到的数据。

#figure(
  image("images/bbap/bap-02-chem-ppc.png", width: 40%),
  caption: none,
) <ppc>

在@ppc 中，黑线是数据的 KDE，灰线是从 100 个后验预测样本中的每一个计算出的 KDE。灰线反映了我们对预测数据分布的不确定性。这些图看起来毛茸茸的或不稳定；当数据点很少时，就会发生这种情况。

#tip[
  默认情况下，ArviZ 中的 KDE 在数据的实际范围内估计，并在数据范围之外假设为零。
]

从@ppc 中，我们可以看到模拟数据的均值略微向右偏移，并且模拟数据的方差似乎比实际数据更大。这种差异的来源可以归因于我们对可能性的选择以及两个离群值（@box）。

对于离群值，我们的一个选择是回溯错误，但数据并不一定是我们收集的。此时，我们可以考虑如下 2 种方法舍弃：

+ 超过数据四分位范围（interquartile range，IQR）边界 1.5 倍时
+ 超过数据标准差 2、3 倍时\

然而，从建模的角度来看，我们可以责怪模型并对其进行更改，而非责怪数据。一般来说，Bayesian 更喜欢使用不同的先验和可能性将假设直接编码到模型中，而非通过诸如异常值删除规则之类的临时启发式方法。简单说，Gaussian 模型可能并不适合一些实际问题，我们可以考虑使用其他模型，详见【线性回归】。

#bibliography("lib/prob.bib", style: "future-science")
