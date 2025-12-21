#import "lib/lib.typ": *
#show: chapter-style.with(title: "Gaussian 过程", info: info)

= 线性模型及其扩展
<线性模型及其扩展>


== 非线性的形式
线性回归有三大特性：线性、全局性和数据未加工性。其中，线性又分为特征线性、全局线性和系数线性，将其某一种线性改为非线性，即可得到一种新的模型。

- 线性 → ×
  - 特征非线性：多项式回归
  - 全局非线性：线性分类（在输出端加入激活函数）
  - 系数非线性：感知机、神经网络（反向传播）
- 全局性 → ×（不基于所有点进行回归）
  - 线性样条回归
  - 决策树
- 数据未加工性 → ×（在输入端降维）
  - PCA
  - 流形

== 建模函数

对于建模，有一个通式

$ θ = ψ(ϕ(𝑿)β) $

其中，$θ$是某个概率分布的参数，如 Gaussian 的均值、二项式的$p$参数等等。$ψ$为激活函数，$ϕ$是一个函数，可以是平方根或多项式函数。对于简单的线性回归情况，$ψ$是恒等函数（identity function）。

拟合一个 Bayesian 模型可看作是寻找权重$β$的后验分布，故这被称为近似函数的权重观。以多项式回归为例，通过令$ϕ$为一个非线性函数，我们可将输入映射到一个特征空间。然后我们在特征空间中拟合一个线性关系，而这个线性关系在实际空间中不是线性的。我们看到，通过使用适当项数的多项式，我们可完美拟合任何函数。但，除非我们应用某种形式的正则化（例如，使用先验分布），否则将导致记忆数据的模型。

Gaussian 过程（GP）为任意函数建模提供了一个原则性的解决方案，它有效地让数据决定了函数的复杂性，同时最小化了过拟合的机率。讨论 GP 时，需要将函数表示为概率对象。我们可把函数$f$看作是一组输入$x$到一组输出$y$的映射，写成

$ y = f(x) $
表示函数的一种方法是为每个$x_i$的值列出对应的$y_i$的值。一般情况下，$x$和$y$的值会在实线上；故，我们可把一个函数看作是一个无限的、有序的$(x_i, y_i)$配对值的列表。顺序是很重要的，因为若将这些值洗牌，将得到不同的函数。一个函数也可表示为一个无限数组，这个数组以$x$的值为索引，但$x$的值并不限于整数。

使用这些描述，我们可用数字表示任何我们想要的特定函数。但若用概率表示函数，则可通过让映射具有概率性质来实现，即令每个$y_i$的值是一个 Gaussian 随机变量，具有给定的均值和方差。为具体化这一想法，分别采用两种不同的函数生成数据。第一个函数中的每个点是完全独立的。第二个函数中，我们引入一些依赖性。点$y_(i + 1)$的均值就是$y_i$的值。

```python
_, ax = plt.subplots()

x = np.linspace(0, 1, 10)
y = np.random.normal(0, 1, len(x))
ax.plot(x, y, "o-", label="the first one")
y = np.zeros_like(x)
for i in range(len(x)):
    y[i] = np.random.normal(y[i - 1], 1)

ax.plot(x, y, "o-", label="the second one")
ax.legend()
```

下图表明，使用 Gaussian 分布的样本对函数进行编码并不愚蠢。尽管如此，用于生成数据的方法是有限的，且不够灵活。只有第二个函数能表达出来一定的结构或模式。

#figure(
  image("images/bbap/bap-08-gp-func.png", width: 40%),
  caption: "两种生成数据的方法",
)

== 多元 Gaussian

我们可用 Gaussian 来表示一维函数，以获得$n$个样本，也可使用$n$维多元 Gaussian 分布来获得长度为$n$的向量。这样 10 个点中的每个点的方差均是$1$，协方差是 $0$。若把这些$0$换成其他（正）数，我们就可得到协方差。故，为了用概率论的方式来表示函数，我们只需要一个具有合适协方差矩阵的多元 Gaussian。

在实践中，协方差矩阵由核函数（kernel function）指定。为方便说明，先以对称核函数为例，它接受 2 个输入，并在输入相同的情况下返回$0$，否则返回正值。若满足这些条件，我们就可将核函数的输出解释为 2 个输入之间相似性的度量。

一个常用的内核是指数二次核

$ K(x, x^′) = exp(-frac(norm(x - x^′)^2, 2 ℓ^2)) $

其中，$norm(x - x^′)^2$是 Euclidean 距离的平方。不难看出，指数二次核公式与 Gaussian 分布相似，因此这个核也被称为 Gaussian 核。$ℓ$被称为长度标度（或方差），控制着核的宽度。Gaussian 核的协方差值在$[0, 1]$之间，对于其他核，其他值亦可。

#tip[
  核是将数据点沿 x 轴的距离转化为预期函数值的协方差值（在 y 轴上）。故最接近的两个点在 x 轴上，最相似的我们期望其值在 y 轴上。
]

```python
def exp_quad_kernel(x, knots, ℓ=1):
    return np.array([np.exp(-((x - k) ** 2) / (2 * ℓ**2)) for k in knots])

points = np.linspace(0, 10, 200)
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True)

for ℓ, ax in zip((0.2, 1, 2, 10), axes.flatten()):
    cov = exp_quad_kernel(points, points, ℓ)
    ax.plot(points, stats.multivariate_normal.rvs(cov=cov, size=2).T)
    ax.set(title=f"ℓ ={ℓ}")

fig.text(0.51, -0.03, "x", fontsize="medium")
fig.text(-0.03, 0.5, "f(x)", fontsize="medium")
```

正如所看到的，Gaussian 核意味着由参数$ℓ$控制的各种各样的函数，$ℓ$的值越大，函数越平滑。

#figure(
  image("images/bbap/bap-08-gauss-kernal.png", width: 40%),
  caption: "Gaussian 核控制的函数",
)

== 正式定义

Gaussian 过程（GP）的取自维基百科的定义，如下

_"以时间或空间为索引的随机变量集合，这些随机变量的每个有限集合都具有多元 Gaussian 分布，即其每个有限线性组合均是 Gaussian 分布。"_

理解 GP 的诀窍在于认识到 GP 是一个数学上的拐棍。因为在实践中，我们不需要直接处理这个无限的数学对象，而是只在我们有数据的地方评估 GP。由此，我们将无限维的 GP 折叠成一个有限的多元 Gaussian 分布，其维度与数据点一样多。在数学上，这种折叠是通过对无限未观测维度的边际化来实现的。理论保证我们可以省略（实际上是边际化）那些我们正在观察的点之外的所有的点。它还保证我们将始终得到一个多元的 Gaussian 分布。

请注意，我们将多元 Gaussian 的均值设为$0$，只使用协方差矩阵，通过指数二次核来建立函数模型。在处理 GP 时，将多元 Gaussian 的均值设为$0$是常见的做法。

#tip[
  GP 对于建立 Bayesian 非参数模型很有用，因为我们可将其作为函数的先验分布。
]

= GP 回归
<GP-回归>

接下来我们将$y$作为$x$加上一些噪声的函数$f$来建模。

$ y ∼ 𝒩(μ = f(x)), σ = ɛ $

其中，$ɛ ∼ (0, σ_ɛ)$。在$f$上设置 GP 作为先验分布，从而有

$ f(x) ∼ cal("GP")(μ_x, K(x, x^′)) $

其中，$cal("GP")$代表一个 GP 分布，$μ_x$是均值函数，$K(x, x^′)$是核函数、协方差或函数。若先验分布是 GP，似然是 Gaussian 分布，则后验亦为 GP

$
  p(f(X_*))|X_*, X, y ∼ 𝒩(μ, Σ)\
  μ = K_*^⊤ K^(-1) y\
  Σ = K_(* *) - K_*^⊤ K^(-1) K_*
$

其中，$X$是观察到的数据点，$K_*$代表测试点，即我们想知道用来推断函数值的新点，且有

- $K = K(X, X)$
- $K_* = K(X_*, X)$
- $K_(* *) = K(X_*, X_*)$

```python
x = np.random.uniform(0, 10, size=15)
y = np.random.normal(np.sin(x), 0.1)
X = x[:, None]

with pm.Model() as model_reg:
    ℓ = pm.Gamma("ℓ", 2, 0.5)
    cov = pm.gp.cov.ExpQuad(1, ls=ℓ)
    gp = pm.gp.Marginal(cov_func=cov)
    ϵ = pm.HalfNormal("ϵ", 25)

    y_pred = gp.marginal_likelihood("y_pred", X=X, y=y, sigma=ϵ)
    idata_reg = pm.sample(2000)
```

这里，我们使用了 `gp.marginal_likelihood()`，而非像表达式所预期的那样使用 Gaussian 似然，因为边际似然是似然和先验的积分

$ p(y|X, θ) ∼ ∫ p(y|f, X, θ) p(f|X, θ) dd(f) $

其中，$θ$代表所有未知参数，$x$是自变量，$y$是因变量。请注意，我们正在对函数$f$的值进行边际化，对于 GP 先验和 Gaussian 似然，边际化可通过分析进行。通常情况下，对于$ℓ$，避开零的先验效果更好，一个默认先验是$"Gamma"(2, 0.5)$。

== 预测

现在，我们已经找到了$ℓ$和$ɛ$的值，我们可能想从 GP 后验中得到样本，即函数拟合数据的样本。我们可通过使用 `gp.conditional()` 计算在新输入位置上评估的条件分布来实现，进而从后验预测分布中获取样本（在 `X_new` 值处评估）。这里使用 `pm.gp.util.plot_gp_dist()` 来获取更好的可视化效果。下图中的每条点迹代表一个百分位数，范围从 51（浅色）到 99（深色）。

```python
X_new = np.linspace(np.floor(x.min()), np.ceil(x.max()), 100)[:, None]
with model_reg:
    # del model_reg.named_vars['f_pred']
    f_pred = gp.conditional("f_pred", X_new)
    pred_samples = pm.sample_posterior_predictive(idata_reg, var_names=[f_pred])

_, ax = plt.subplots(figsize=(12, 5))
pm.gp.util.plot_gp_dist(
    ax, pred_samples["f_pred"], X_new, palette="viridis", plot_samples=False
)
ax.plot(X, y, "ko")
ax.set(xlabel="x", ylabel="f(x)")
```

#figure(
  image("images/bbap/bap-08-gpreg.png", width: 45%),
  caption: "GP 回归",
)

另一种选择是，计算参数空间中给定点的条件分布的均值向量和标准差。这里使用 `gp.predict()` 来计算均值和方差。

```python
with model_reg:
    point = {
        "ℓ": idata_reg.posterior["ℓ"].mean(("chain", "draw")).values,
        "ϵ": idata_reg.posterior["ϵ"].mean(("chain", "draw")).values,
    }
    μ, var = gp.predict(X_new, point=point, diag=True)
    sd = var**0.5

_, ax = plt.subplots(figsize=(12, 5))
ax.plot(X_new, μ, "C1")
ax.fill_between(X_new.flatten(), μ - sd, μ + sd, color="C1", alpha=0.3)
ax.fill_between(X_new.flatten(), μ - 2 * sd, μ + 2 * sd, color="C1", alpha=0.3)
ax.plot(X, y, "ko")
ax.set(xlabel="X")
```

== 空间自相关

我们可使用具有非 Gaussian 似然和适当激活函数的线性模型来扩展线性模型的范围。我们同样也可用 GP 做。例如，可使用 Poisson 似然和指数激活函数。对于这样的模型，后验不再是可分析的，但我们还是可用数值方法来逼近它。

下面的数据中有 10 个不同的岛屿社会；对于每个岛屿社会，有他们使用的 `total-tools` 。一些理论预测，较大的人口比较小的人口更易开发和维护工具。另一个重要的因素是人口之间的接触率。由于 `total-tools` 作为因变量，可用人口作为自变量进行 Poisson 回归。这里使用人口的对数，因为人口的数量级更重要。

#let csv1 = csv("data/islands.csv")
#figure(tableq(csv1.slice(0, 5), 9, inset: 0.31em), caption: none, kind: table)

将接触率纳入模型的一种方法是收集这些社会在历史上的接触频率信息，并创建一个分类变量，如低/高接触率。另一种方法是使用社会之间的距离作为接触率的代用词，因为我们可合理地假设，较近社会比远处的社会接触得更频繁。

#let csv1 = csv("data/islands_dist.csv")
#figure(
  tableq(csv1.slice(0, 5), 11, inset: 0.31em),
  caption: "地理距离",
  kind: table,
)

我们的模型可描述为

$
  f ∼ cal("GP")([0, …, 0]), K(x, x^′)\
  μ ∼ 𝔼[α + β x + f]\
  y ∼ "Pois"(μ)
$

这里，省略了$α$和$β$的先验，以及内核的超先验。$x$是 `logpop`，$y$是 `total-tools`。这算是一个新的 Poisson 回归模型，线性模型中的$f$来自于 GP。使用距离矩阵 `islands_dist` 计算 GP 的核。故我们假设 `total-tools` 只是人口的结果，且独立于近邻社会。我们将把每个社会的 `total-tools` 作为其地理相似性的函数来建模。于是有

```python
islands_dist_sqr = islands_dist.values**2
index = islands.index.values
logpop = islands["logpop"]
total_tools = islands["total_tools"]
x_data = [
    islands["lat"].to_numpy().flatten()[:, None],
    islands["lon"].to_numpy().flatten()[:, None],
]

with pm.Model() as model_islands:
    η = pm.HalfCauchy("η", 1)
    ℓ = pm.HalfCauchy("ℓ", 1)
    cov = η * pm.gp.cov.ExpQuad(1, ls=ℓ)
    gp = pm.gp.Latent(cov_func=cov)
    f = gp.prior("f", X=islands_dist_sqr)

    α = pm.Normal("α", 0, 10)
    β = pm.Normal("β", 0, 1)
    μ = pm.math.exp(α + f[index] + β * logpop)

    tt_pred = pm.Poisson("tt_pred", μ, observed=total_tools)
    idata_islands = pm.sample(1000, tune=1000)
```

为了了解协方差函数在距离方面的后验分布，我们可绘制一些后验分布的样本。图中的粗线是一对社会之间的协方差的后验中值，它是距离的函数。使用中位数是因为$ℓ$和$η$的分布非常倾斜。可看到，平均而言，协方差并不高，且在大约 2000 公里处，协方差几乎降到了 0。细线代表了不确定性，可看到有很多不确定性。

#figure(
  image("images/bbap/bap-08-gpreg-island-post.png", width: 40%),
  caption: "后验样本",
)

为了探讨岛屿 - 社会之间的相关性，我们必须将协方差矩阵变成一个相关矩阵。可得到两个观察结果是，Hawaii 是非常孤独的。另外，Malekula（Ml）、Tikopia（Ti）和 Santa Cruz（SC），彼此高度相关。

#let csv1 = csv("./python/bap-08-island-cov.csv")
#figure(
  tableq(csv1.slice(0, 5), 11, inset: 0.31em),
  caption: "距离相关矩阵",
  kind: table,
)

现在我们要利用经纬度信息来绘制岛屿 - 社会的相对位置。左图显示了在相对地理位置的背景下计算的社会之间的后验中位数相关性的线条。有些线条不可见，因为我们已经使用相关性来设置线条的不透明度。右图中，显示了后验中位数的相关性。虚线代表 `total-tools` 的中位数和 HPD 94% 区间作为对数人口的函数。两图中，点的大小与每个岛屿社会的人口成正比。

#figure(
  image("images/bbap/bap-08-gpreg-island-dist.png", width: 40%),
  caption: "地理-人口-工具数量",
)

Malekula、Tikopia 和 Santa Cruz 之间的相关性如何描述了这样一个事实，即他们的 `total-tools` 相当低，接近中位数或低于其人口的预期 `total-tools`。Trobriands 和 Manus 也发生了类似的情况；他们在地理上很接近，但他们的 `total-tools` 比预期的 `logpop` 要少。Tonga 的 `total-tools` 比预期的 `logpop` 要多，而与 Lua Fiji 的相关性相对较高。在某种程度上，该模型告诉我们，Tonga 对 Lua Fiji 有积极影响，增加了 `total-tools`，抵消了它对近邻 Malekula、Tikopia 和 Santa Cruz 的影响。

= GP 分类
<GP-分类>

== 逻辑回归
<逻辑回归>

GP 不仅限于回归，也可用于分类。从最简单的分类问题开始：两个类，`setosa` 和 `versicolor`，只有一个独立变量 `sepal_length`。我们对分类变量 `setosa` 和 `versicolor` 进行 0-1 编码。对于这个模型，我们不使用 `pm.gp.Marginal` 类来实例化 GP 先验，而是使用 `pm.gp.Latent` 类。后者更通用，可用于任何似然。前者仅限于 Gaussian 似然。

```python
iris_df = iris.query("species == ('setosa', 'versicolor')")
y = pd.Categorical(iris_df["species"]).codes
x_1 = iris_df["sepal_length"].to_numpy().flatten()
X_1 = x_1[:, None]

with pm.Model() as model_iris:
    # ℓ = pm.HalfCauchy("ℓ", 1)
    ℓ = pm.Gamma("ℓ", 2, 0.5)
    cov = pm.gp.cov.ExpQuad(1, ℓ)
    gp = pm.gp.Latent(cov_func=cov)
    f = gp.prior("f", X=X_1)

    y_ = pm.Bernoulli("y", p=pm.math.sigmoid(f), observed=y)
    idata_iris = pm.sample(1000, chains=1, compute_convergence_checks=False)
```

现在，我们已经找到了$ℓ$的值，我们可像 GP 回归中那样借助 `gp.conditional()` 来计算在一组新输入位置上评估的条件分布。为了显示这个模型的结果，我们将使用下面的函数直接从 `f_pred` 中计算出边界决定。

```python
def find_midpoint(array1, array2, value):
    array1 = np.asarray(array1)
    idx0 = np.argsort(np.abs(array1 - value))[0]
    idx1 = idx0 - 1 if array1[idx0] > value else idx0 + 1
    if idx1 == len(array1):
        idx1 -= 1
    return (array2[idx0] + array2[idx1]) / 2
```

#figure(
  image("images/bbap/bap-08-gpcls-iris-hdi.png", width: 45%),
  caption: "无白噪声 HDI",
)

由上图，曲线整体拟合的效果还算不错，但尾部在 `x_1` 值较低时向上，而在 `x_1` 值较高时向下。这是在没有数据（或数据很少）时，预测函数向先验值移动的结果。若我们只关注边界判定，这应该不是一个真正的问题，但若我们想根据不同的萼片长度值建立属于 `setosa` 或 `versicolor` 的概率模型，那么我们就应该改进我们的模型，为尾部建立一个更好的模型。实现这一目标的方法之一就是为 GP 添加更多的结构。我们将 `cov` 建模为三个核的组合，通过添加线性核，来解决肥尾问题。

```python
with pm.Model() as model_iris2:
    # ℓ = pm.HalfCauchy("ℓ", 1)
    ℓ = pm.Gamma("ℓ", 2, 0.5)
    cov = (
        pm.gp.cov.ExpQuad(1, ℓ)
        + τ * pm.gp.cov.Linear(1, c)
        + pm.gp.cov.WhiteNoise(1e-5)
    )
...
```

需要说明的是，这里的白噪声核只是一个计算技巧，用于稳定协方差矩阵的计算。

#figure(
  image("images/bbap/bap-08-gpcls-iris2-hdi.png", width: 45%),
  caption: "含白噪声 HDI",
)

由此不难看出，逻辑回归是 GP 的一个特例，而简单线性回归是 GP 的一个更特别的特例。事实上，很多已知的模型都可看作是 GP 的特例，或至少它们与 GP 有某种联系。

#tip[
  需要指出的是，GP 核是有限制的，以保证得到的协方差矩阵是正定的。但数值误差会导致违反这个条件。这个问题的一个表现是，我们在计算拟合函数的后预测样本时，会得到 `nans`。减轻这种错误的一个方法是通过添加一点噪声来稳定计算。
]

== 更复杂的分类

在实践中，我们用 GP 来模拟一个我们用逻辑回归就能解决的问题，并没有太大意义。相反，我们希望用 GP 来模拟一些比较复杂的数据，而这些数据是不太灵活的模型所不能很好地捕捉到的。

假设将患某种疾病的概率作为年龄的函数来建模。结果发现，非常年轻和非常年长的人比中年人的风险更高。数据集 `space_flu` 是基于此想法伪造的数据集。

#let csv1 = csv("data/space_flu.csv")
#figure(tableq(csv1.slice(0, 6), 2), caption: none, kind: table)

```python
with pm.Model() as model_space_flu:
    ℓ = pm.HalfCauchy("ℓ", 1)
    cov = pm.gp.cov.ExpQuad(1, ℓ) + pm.gp.cov.WhiteNoise(1e-5)
    gp = pm.gp.Latent(cov_func=cov)
    f = gp.prior("f", X=age)

    y_ = pm.Bernoulli("y", p=pm.math.sigmoid(f), observed=space_flu)
    idata_space_flu = pm.sample(1000, chains=1, compute_convergence_checks=False)
```

同之前一样，为 `model_space_flu` 生成后验预测样本，然后绘制结果。

#figure(
  image("images/bbap/bap-08-gpcls-flu-hdi.png", width: 40%),
  caption: none,
)

显然 GP 能够很好地拟合这个数据集，即使数据要求函数比逻辑函数更复杂。
