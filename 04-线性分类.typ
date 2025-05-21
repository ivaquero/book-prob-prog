#import "lib/lib.typ": *
#show: chapter-style.with(
  title: "线性分类",
  info: info,
)

= 分类任务

== 激活函数

分类是关于给定一些输入变量，给一个输出变量分配一个离散值（代表一个离散类）。对于分类任务，需要一种能输出$[0, 1]$区间的值的函数。为实现这种泛化，我们可以在线性回归方程后加入一层激活函数（activation function），也称逆链接函数（inverse link function）。

$
  f & : w^⊤ x → {0, 1}\
  f^(-1) & : {0, 1} → w^⊤ x
$

#tip[
  激活函数的反函数称链接函数（link function）。
]

可选择的激活函数有很多，最简单是恒等函数，其返回其参数使用的相同值。线性回归建模中的所有模型都使用了恒等函数。原则上，Gaussian 对于连续变量在实线上取任何值都能很好地工作，而对离散变量是离散，需要改变对这些分布的均值的可信值进行建模的方式。一种方法是保留线性模型，但使用激活函数将输出限制在所需区间内，如

- 使用二项分布：需要一个线性模型来返回$[0, 1]$区间内的均值
- 使用 Gamma 分布或指数分布：希望对只能取正值的数据进行建模

== 线性分类方式

总的来说，我们有两种线性分类方式

- 软分类，产生不同类别的概率
  - 判别式（直接对条件概率进行建模）：逻辑回归
  - 生成式（根据 Bayes' 法则先计算参数后验，再进行推断）
    - 连续：高斯判别分析（GDA）
    - 离散：朴素贝叶斯（NB）
- 硬分类，直接需要输出观测对应的分类
  - 感知机
  - 线性判别分析（LDA），更多用于降维

= 逻辑回归
<逻辑回归>

逻辑回归是最常见的二分类模型，使用 sigmoid 函数作为激活函数

$ "logistic" = frac(1, 1 + e^(-z)) $

这个函数的结果总是在$[0, 1]$区间内，可把从线性模型中计算出来的值压缩成可输入进 Bernoulli 分布的值。首先，对这些类别进行编码，令$y ∈ {0, 1}$。这里的$θ$将由一个线性模型定义，并以 sigmoid 函数作为激活函数。省略先验，得

$
  θ &~ "logistic"(α + x β)\
  y &~ "Bern"(θ)
$

鸢尾数据集的每个物种有 50 个样本。对于每个样本，数据集包含 4 个变量，假设其均为独立变量（也称特征）：花瓣长度、花瓣宽度、萼片长度和萼片宽度。一种检查数据的方法是散点矩阵。矩阵是对称的，上下两个三角形显示相同的信息。主对角线上是每个特征的 KDE。在每个子图中，我们用不同的颜色表示 3 个物种。

#figure(
  image("images/bbap/bap-04-iris-matrix.png", width: 60%),
  caption: "鸢尾",
)

现在，我们从最简单的分类问题开始，选取两个类别，`setosa ` 和 `versicolor`，以及一个特征，萼片长度。用数字$0$和$1$对两个类别变量编码。注意 2 个确定性变量：$θ$和 `bd`。$θ$是应用于$μ$变量的 sigmoid 函数的输出，`bd` 是决策边界，即用于分隔类的值。

```python
iris2 = iris.query("species == ('setosa', 'versicolor')")
y_0 = pd.Categorical(iris2["species"]).codes
x_0 = iris2["sepal_length"].to_numpy()
x_c = x_0 - x_0.mean()

with pm.Model() as model_lg:
    α_lr = pm.Normal("α", mu=0, sigma=1)
    β_lr = pm.Normal("β", mu=0, sigma=5)
    μ_lr = α_lr + x_c * β_lr
    θ_lr = pm.Deterministic("θ", pm.math.sigmoid(μ_lr))
    bd = pm.Deterministic("bd", -α_lr / β_lr)
    yl = pm.Bernoulli("yl", p=θ_lr, observed=y_0)

    idata_lg = pm.sample(random_seed=123)
```

#let csv1 = csv("python/bap-04-iris-logreg.csv")
#figure(
  tableq(csv1, 10, inset: 0.31em),
  caption: "逻辑回归的系数",
  kind: table,
)

下图显示了萼片长度与鸢尾种类的关系（`setosa = 0`，`versicolor = 1`）。为了避免重叠，这里二进制响应变量是抖动的。S 形线是$θ$的均值。这条线可被解释为知道萼片长度的值时，花是 `versicolor`（x = 1）的概率。决策边界被表示为一条垂直线，左边对应于$0$（`setosa`），右边的值对应于 $1$ （`versicolor`）。。半透明带为各曲线的 94％ HPD。

#figure(
  image("images/bbap/bap-04-iris-logreg-hdi.png", width: 40%),
  caption: "sigmoid 函数的决策边界",
)

这里的决策边界被定义为$x_i$的值，其中，$y = 0.5$，结果是$- α / β$，其推导如下。

根据模型的定义，有

$ θ = "logistic"(α + x β) $

当逻辑回归的参数为$0$时，有$θ = 0.5$，即

$ 0.5 = "logistic"(α + x_i β) ⇒ 0 = α + x_i β $

重排上式，发现$x_i$的值在$θ = 0.5$时对应的表达式如下

$ x_i = - α / β $

#warning[
  决策边界并不必须使分类的代价对称。
]

= 鲁棒逻辑回归
<鲁棒逻辑回归>

逻辑回归中，我们可能会发现一个数据集有不寻常的$0$和/或$1$。在鸢尾数据集中增加一些入侵者。这里有一些 `versicolors` 的萼片长度特别短。我们用一个混合模型来解决这个问题。通过随机猜测，输出变量的概率为$π$，或是逻辑回归模型中的$1 - π$概率。在数学上有

$ p = π 0.5 + (1 - π) "logistic"(α + X β) $

当$π = 1$时，$p = 0.5$，当$π = 0$时，则恢复逻辑回归的表达式。对应代码如下

```python
y_0r = np.concatenate((y_0, np.ones(6, dtype=int)))
x_0r = np.concatenate((x_0, [4.2, 4.5, 4.0, 4.3, 4.2, 4.4]))
x_cr = x_0r - x_0r.mean()

with pm.Model() as model_rlg:
    α_lrr = pm.Normal("α", mu=0, sigma=10)
    β_lrr = pm.Normal("β", mu=0, sigma=10)
    μ_lrr = α_lrr + x_cr * β_lrr
    θ_lrr = pm.Deterministic("θ", pm.math.sigmoid(μ_lrr))
    bd_lrr = pm.Deterministic("bd", -α_lrr / β_lrr)
    π_lrr = pm.Beta("π", 1.0, 1.0)
    p_lrr = π_lrr * 0.5 + (1 - π_lrr) * θ_lrr
    yl_lrr = pm.Bernoulli("yl", p=p_lrr, observed=y_0r)

    idata_rlg = pm.sample(random_seed=123)
```

若我们将这些结果与 `model_0` 的结果进行比较，发现我们得到的边界大致相同，但 HPD 带明显变窄了一些。

#figure(
  image("images/bbap/bap-04-iris-logreg-t-hdi.png", width: 40%),
  caption: "sigmoid 函数的鲁棒决策边界",
)

#let csv1 = csv("python/bap-04-iris-logreg-t.csv")
#figure(
  tableq(csv1, 10, inset: 0.31em),
  caption: "鲁棒逻辑回归的系数",
  kind: table,
)

= 多元逻辑回归
<多元逻辑回归>

与多元线性回归类似，多元逻辑回归要使用多个独立变量。让我们尝试将萼片长度和萼片宽度结合起来。由上面的模型，有

$ 0.5 = "logistic"(α + β_1 x_1 + β_2 x_2) ⇒ 0 = α + β_1 x_1 + β_2 x_2 $

通过重排，发现$x_2$的值$x_2$中$θ = 0.5$对应于下面的表达式

$ x_2 = - α / β_2 + (-β_1 / β_2 x_1) $

```python
y_1 = pd.Categorical(iris2["species"]).codes
x_1 = iris2[["sepal_length", "sepal_width"]].to_numpy()

with pm.Model() as model_1:
    α = pm.Normal("α", mu=0, sigma=10)
    β = pm.Normal("β", mu=0, sigma=2, shape=2)
    μ = α + pm.math.dot(x_1, β)
    θ = pm.Deterministic("θ", 1 / (1 + pm.math.exp(-μ)))
    bd = pm.Deterministic("bd", -α / β[1] - β[0] / β[1] * x_1[:, 0])
    yl = pm.Bernoulli("yl", p=θ, observed=y_1)

    idata_1 = pm.sample(random_seed=123)
```

可以看到，94% HPD 带的是弯曲的。表面上的曲率是多条线围绕一个中心区域（大致在$x$均值和$y$均值的周围）旋转的结果。

#figure(
  image("images/bbap/bap-04-iris-logreg-multi.png", width: 40%),
  caption: "鸢尾的多元逻辑回归",
)

== 对系数的解释

解释逻辑回归的系数$β$并不像解释线性模型那样简单，因为激活函数引入了非线性因素。若$β$为正值，增加$x$会使$p(y=1)$增加一定的量，但这个量不是$x$的线性函数。让我们以数学视角，重新审视一下 sigmoid 函数。

由上，令$θ = p(y=1)$，有

$ θ = "logistic"(α + X β) $

这里引入对数几率（logit），它是 sigmoid 的逆函数

$ "logit"(z) = log (frac(z, 1 - z)) $

故

$ "logit"(θ) = α + X β $

即

$ log frac(p(y=1), 1 - p(y=1)) = α + X β $

其中，$frac(p(y=1), 1 - p(y=1))$就是之前介绍过的几率（odds），而逻辑回归中的系数$β$就代表随$x$增加而增加的对数几率。

概率到几率的变换是单调递增变换，概率被限制在$[0, 1]$区间，而几率则在$[0, ∞)$。对数是另一种单调变换，对数几率在$(-∞, ∞)$区间内。

#figure(
  image("images/funcs/logit-odds.png", width: 40%),
  caption: "几率-对数几率",
)

故，总结提供的系数均是以对数几率为标准的。

#let csv1 = csv("python/bap-04-iris-logreg-multi.csv")
#figure(
  tableq(csv1, 10, inset: 0.31em),
  caption: "多元逻辑回归的系数估计",
  kind: table,
)

== 处理关联变量

处理相关变量是经常需要面对的棘手事情。在鸢尾数据集，将此前的模型变量变更为 `petal_width` 和 `petal_length`。容易发现系数比以前更宽了，且 HPD 带也更宽了。这里绘制的是相关性的绝对值，因为此刻我们并不关心变量间相关性的符号，只关心其强度。

#figure(
  image("images/bbap/bap-04-iris-corr.png", width: 40%),
  caption: "鸢尾萼片相关性",
)

处理相关的变量时，除了删除一些相关变量，还可以选择在先验中加入更多的信息。对弱信息先验，可以将所有非二元变量的均值缩放为 0，然后使用$t$分布

$ β ∼ "t"(0, ν, s d) $

其中，`sd` 的选择是为了告知我们关于标度的期望值。正态参数$ν$建议在$3-7$左右。即，一般情况下，我们期望系数是小的，而使用肥尾是因为我们偶尔会发现一些较大的系数。

== 处理非平衡数据

鸢尾数据集是完全平衡的，每个类别有完全相同数量的观察值。然而，许多数据集由不平衡数据组成，即，来自一个类的数据点比来自另一个类的数据点多很多。当这种情况发生时，逻辑回归可能会遇到麻烦，即无法像数据集比较平衡时那样准确地确定边界。为了看到这种行为的一个例子，这里我们任意删除 `setosa` 类的一些数据点。

```python
iris_df = iris.query("species == ('setosa', 'versicolor')")[45:]
y_3 = pd.Categorical(iris_df["species"]).codes
x_3 = iris_df[["sepal_length", "sepal_width"]].to_numpy()
```
此时的决策边界向丰度较低的类别转移，且不确定性比以前更大。这是典型的非平衡数据的逻辑模型的行为。

#figure(
  image("images/bbap/bap-04-iris-unbalanced.png", width: 40%),
  caption: "非平衡多元逻辑回归",
)

那么，当发现不平衡的数据，我们往往有如下选择

- 控制生成数据
- 检查模型的不确定性，并运行后验预测检查，看看结果是否有用
- 输入更多的先验信息
- 运行一个替代模型

== 协方差矩阵

对于多元变量的相关性，还可以使用协方差矩阵。当描述一个二元 Gaussian 分布，需要 2 个均值，每个边际 Gaussian 都需要 1 个，因此需要一个$2 × 2$ 的协方差矩阵

$
  Σ = mat(
    delim: "[",
    σ_(x_1)^2, ρ_(x_1 x_2);
    ρ_(x_1 x_2), σ_(x_2)^2
  )
$

#figure(
  image("images/bbap/bap-05-covmat.png", width: 65%),
  caption: "协方差矩阵",
)

在协方差矩阵的主对角元是每个变量的方差，矩阵中的其他元素是协方差（变量之间的方差），用$ρ$来表示。由于不知道协方差矩阵的值，我们必须使用先验，这里有 3 个选择

- 使用 Wishart 分布，其可被认为是 Gamma 分布的高维泛化，也可被认为是$χ^2$分布的泛化
- 使用 LKJ 先验，其为相关矩阵的先验，一般来说，从相关性的角度考虑更有用
- 直接把先验给$σ_(x_1), σ_(x_2), ρ$，然后使用这些值来手动建立协方差矩阵

这里，暂时选择第三种。于是由协方差矩阵得到的$R^2$显示，我们模型中的变量是高度相关的。

#let csv1 = csv("python/bap-04-multireg-r2.csv")
#figure(
  tableq(csv1, 10, inset: 0.31em),
  caption: "多元线性回归的R²",
  kind: table,
)

```python
with pm.Model() as multireg_pearson:
    μ = pm.Normal("μ", mu=X.mean(0), sigma=10, shape=2)
    σ_1 = pm.HalfNormal("σ_1", 10)
    σ_2 = pm.HalfNormal("σ_2", 10)
    ρ = pm.Uniform("ρ", -1.0, 1.0)
    r2 = pm.Deterministic("r2", ρ**2)
    cov = pm.math.stack(([σ_1**2, σ_1 * σ_2 * ρ], [σ_1 * σ_2 * ρ, σ_2**2]))
    y_pred = pm.MvNormal("y_pred", mu=μ, cov=cov, observed=X)

    idata_p = pm.sample(random_seed=123)
```

= softmax 回归
<softmax-回归>

将逻辑回归泛化到两类以上的一种方法是用 softmax 回归，这里需要对逻辑回归进行 2 个改变，首先，用 softmax 函数代替逻辑函数

$ "softmax"_i (μ = frac(exp(μ_i), sum exp(μ_k))) $

softmax 保证得到加起来为$1$的正值，当$κ = 2$时，softmax 函数为 sigmoid 函数。实际上，softmax 函数与统计力学中的 Boltzmann 分布具有相同的形式，后者有一个称为温度的参数$T$，用于切分$μ$：

- 当$T → ∞$时，概率分布变得扁平，所有的状态均是同等可能的
- 当$T → 0$时，只有最可能的状态才会被填充

因此，softmax 的行为像 `max()` 函数。另一处修改是，用类别分布代替 Bernoulli 分布，前者是 后者的泛化，可用于 2 个以上的结果。另外，由 Bernoulli 分布是二项分布的特例，可知类别类分布是多项分布的特例。

为了说明 softmax 回归，我们使用使用鸢尾的 3 个类和 4 个特征。我们还将对数据标准化，这将有助于抽样器更高效地运行。用参数的均值来计算每个数据点属于 3 个类的概率，然后用 `argmax` 函数来分配类，然后将结果与观测值进行比较。

```python
y_s = pd.Categorical(iris["species"]).codes
x_s = iris[iris.columns[:-1]].to_numpy()
x_s = (x_s - x_s.mean(axis=0)) / x_s.std(axis=0)

with pm.Model() as model_s:
    α = pm.Normal("α", mu=0, sigma=5, shape=3)
    β = pm.Normal("β", mu=0, sigma=5, shape=(4, 3))
    μ = pm.Deterministic("μ", α + pm.math.dot(x_s, β))
    θ = pm.math.softmax(μ, axis=1)
    yl = pm.Categorical("yl", p=θ, observed=y_s)

    idata_s = pm.sample(random_seed=123)

data_pred = idata_s.posterior["μ"].to_numpy().mean(0).mean(0)
y_pred = [np.exp(point) / np.sum(np.exp(point), axis=0) for point in data_pred]
f"{np.sum(y_s == np.argmax(y_pred, axis=1)) / len(y_s):.2f}"  # 0.98
```

结果是，我们正确地对数据点进行了分类，只漏掉了 3 个样本。但真正评估我们模型性能的测试将是在没有用于拟合模型的数据上检查它。否则，我们可能高估了模型对其他数据的泛化能力。具体方法将在下一章中详述。

这里的后验，或者说，每个参数的边际分布都非常宽。宽后验是由于所有的概率之和必须为$1$的条件。考虑到这个条件，我们使用的参数超过了完全指定模型的需要。简单说，若有和为$1$的 10 个数，我们只需要知道其中的 9 个，其他的可计算获得。一个解决方案是将额外的参数固定为某个值，如$0$。下面的代码中，我们不再指定$μ$的先验，从而在不直接模拟产生$0$的因素的情况下修正过剩的$0$。

```python
import pytensor.tensor as at

with pm.Model() as model_sf:
    α = pm.Normal("α", mu=0, sigma=2, shape=2)
    β = pm.Normal("β", mu=0, sigma=2, shape=(4, 2))
    α_f = at.concatenate([[0], α])
    β_f = at.concatenate([np.zeros((4, 1)), β], axis=1)
    μ = α_f + pm.math.dot(x_s, β_f)
    θ = pm.math.softmax(μ, axis=1)
    yl = pm.Categorical("yl", p=θ, observed=y_s)

    idata_sf = pm.sample(random_seed=123)
```

= Poisson 回归
<Poisson-回归>

== ZIP 模型

Poisson 分布假设事件的发生是相互独立的，且是在一个固定的时间/空间间隔上发生的。这种离散分布的参数化只用一个值$λ$（这里统一用$μ$表示）对应于分布的均值和方差。虽然$μ$是一个浮点数，但分布的输出总是一个整数。

当使用 Poisson 分布，我们会注意到，在进行后验预测检查时，模型产生的$0$相比数据更少。通常情况下，我们需要假设有 2 个分布的混合

- 概率为$ψ$的 Poisson 分布
- 以概率$1 - ψ$产生额外的$0$

这被称为零膨胀 Poisson（zero-inflated Poisson，ZIP）模型。基本上，ZIP 分布是

$
  p(y_j = 0) &= 1 - ψ + (ψ) e^(-mu)\
  p(y_j = k_i) &= ψ frac(μ_i^x e^(-mu), x_i!)
$

即，零生成过程和 Poisson 分布的混合。其中，速率$μ$是一个随机变量的 Gamma 分布。当数据过于分散时，即数据的方差大于其均值时，负二项分布是 Poisson 分布的一个有用的替代方法。

```python
n = 100
θ_real = 2.5
ψ = 0.1
condition = np.random.random() > (1 - ψ)
noise = np.random.poisson(θ_real)
counts = np.array([(condition) * noise for _ in range(n)])

with pm.Model() as ZIP:
    ψ = pm.Beta("ψ", 1, 1)
    θ = pm.Gamma("θ", 2, 0.1)
    y = pm.ZeroInflatedPoisson("y", ψ, θ, observed=counts)

    idata_z = pm.sample(random_seed=123)
```

== ZIP 回归

现在试着使用 Poisson 或 ZIP 分布来进行回归分析，输出一个计数变量。这里使用指数函数$e$作为激活函数。这种选择保证了线性模型返回的值总是正值。

$ θ = e^((α + 𝑿 β)) $

假设一个公园有 250 组游客。以下是每组数据的组成

- 他们钓到的鱼的数量（`fish_caught`）
- 该组有多少孩子（`child`）
- 他们是否带了露营者来（`camper`）

利用这些数据，建立一个模型，以孩子和露营者预测钓到的鱼的数量。其中，`camper` 是一个二进制变量，当 `camper` 取值为$0$时，涉及$β_1$的项也将为$0$，模型简化为一个单自变量的回归。

#let csv1 = csv("data/fish.csv")
#figure(
  tableq(csv1.slice(0, 5), 8, inset: 0.31em),
  caption: "Fish 数据集",
  kind: table,
)

#tip[
  表示某项属性的缺失/存在的变量通常表示为哑变量（dummy variable）。
]

```python
with pm.Model() as ZIP_reg:
    ψ = pm.Beta("ψ", 1, 1)
    α = pm.Normal("α", 0, 10)
    β = pm.Normal("β", 0, 10, shape=2)
    θ = pm.math.exp(α + β[0] * fish["child"] + β[1] * fish["camper"])
    yl = pm.ZeroInflatedPoisson("yl", ψ, θ, observed=fish["count"])

    idata_zr = pm.sample(random_seed=123)
```

可看出，儿童人数越多，捕到的鱼数量越少。另外，带着露营者出行的人一般都能钓到更多的鱼。若我们检查孩子和露营者的系数，我们会发现

- 每多$1$个孩子，预期的捕鱼数量就会减少约$0.93$
- 乘坐露营车会增加约$0.81$的预期捕鱼数量

#let csv1 = csv("python/bap-04-fish-zipreg.csv")
#figure(
  tableq(csv1, 10, inset: 0.31em),
  caption: "ZIP 回归系数",
  kind: table,
)
