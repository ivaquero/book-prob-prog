#import "lib/lib.typ": *
#show: chapter-style.with(title: "线性回归", info: info)

= 简单线性回归
<简单线性回归>

对线性方程

$ y_i = α + x_i β $

参数$β$是控制着线性关系的斜率，$α$称为截距，告诉我们当$x_i = 0$时的$y$。

有几种方法可找到线性模型的参数，其中一种为最小二乘估计（least squares estimation，LSE）。LSE 返回观察值和预测值之间最小的均方误差。这种表达中，估计$α$和$β$的问题是一个优化问题。最优化的另一种方法是生成一个完全概率的模型。以概率论的方式思考给我们带来了几个优势：

- 可获得$α$和$β$的最优值
- 可估计对参数值的不确定性

#tip[
  使用 LSE 得出的点估算值将与 Bayesian 线性回归的最大后验值（后验值的众数）一致。
]

而优化方法则需要额外的工作来提供这些信息。对更广义的形式，有

$ y ∼ 𝒩(α + x β, σ) $

即，数据向量被假设为 Gaussian 分布，均值为$α + x β$，标准差（噪声）为$σ$。由于不知道$α, β, σ$的值，我们必须为它们设置先验分布。

当为线性模型设置先验时，我们通常假设它们是独立的。这一假设极大地简化了先验的设置，因为我们需要设置三个先验而非一个联合先验。至少原则上，$α$和$β$可以取实线上的任意值，因此通常对它们使用正态先验。对于$α$的先验，我们可通过将该值设置为数据标度相对较高的值，使用一个非常扁平的 Gaussian。对许多问题，我们至少可先验地知道斜率的正负。由于$σ$是正数，通常对其使用 HalfNormal 或 Exponential 先验。我们可在变量$y$的标度上设置一个大的值$σ$。这些模糊的先验保证了先验对后验的影响非常小，这很容易被数据克服。HalfNormal 分布替代选择有均匀分布或 HalfCauchy 分布。

- HalfCauchy 分布通常可作为一个很好的正则先验
- 在参数的硬边界的限制已知时，可使用均匀分布
- 若想围绕标准差的某个特定值使用强先验，可使用 Gamma 分布

== 自行车租赁

现在我们从非常简单的想法开始；我们有一个城市的温度和租赁自行车数量的记录。我们想要对温度和租赁自行车数量之间的关系进行建模。

#figure(
  image("images/bbap/bap-04-bike.png", width: 45%),
  caption: "自行车租赁数据集",
)

```python
with pm.Model() as model_lb:
    α = pm.Normal("α", mu=0, sigma=100)
    β = pm.Normal("β", mu=0, sigma=10)
    σ = pm.HalfCauchy("σ", 10)
    μ = pm.Deterministic("μ", α + β * bikes.temperature)
    y_pred = pm.Normal("y_pred", mu=μ, sigma=σ, observed=bikes.rented)
    idata_lb = pm.sample(random_seed=123)
```

== 解释后验均值

为了探索我们的推理结果，我们将生成后验图，但省略确定性变量 $μ$，否则我们会得到很多图，每个图对应一个温度值。

```python
az.plot_posterior(idata_lb, var_names=['~μ'])
```

#figure(
  image("images/bbap/bap-04-bike-post.png", width: 60%),
  caption: "自行车租赁的后验分布",
) <bike-post>

从@bike-post 中，我们可以看到$α$、$β$和$σ$的边际后验分布。若我们只读取每个分布的均值，例如$μ = 69 + 7.9X$，根据这些信息，我们可以说温度为 0 时租赁自行车的期望值为 69，并且对于每个温度度，租赁自行车的数量增加了7.9。因此，在 28 度的温度下，我们预计租用 69 + 7.9 ∗ 28 ≈ 278 辆自行车。这是我们的预期，但后验也告诉我们这一估计的不确定性。如，$β$ 的 94% HDI 为 (6.1, 9.7)，因此对于每度温度，租赁自行车的数量可能会从 6 辆增加到大约 10 辆。此外，即使我们忽略后验不确定性，只关注意味着，我们对租赁自行车的数量仍然存在不确定性，因为我们的$σ$值为 170。

现在让我们创建一些图来帮助我们可视化这些参数的组合不确定性。让我们从两个均值图开始（@bike-lines）。两者都是租赁自行车平均数量与温度的关系图。不同之处在于我们如何表示不确定性。我们展示了两种流行的方法。在左子图中，我们从后验中获取 50 个样本，并将它们绘制为单独的线。在右子图中，我们取所有可用的后验样本作为$μ$并使用它们来计算 94% HDI。

```python
posterior = az.extract(idata_lb, num_samples=50)
x_plot = xr.DataArray(
    np.linspace(bikes.temperature.min(), bikes.temperature.max(), 50),
    dims="plot_id",
)
mean_line = posterior["α"].mean() + posterior["β"].mean() * x_plot
lines = posterior["α"] + posterior["β"] * x_plot
hdi_lines = az.hdi(idata_lb.posterior["μ"])

fig, axes0 = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
lines_ = axes0[0].plot(x_plot, lines.T, c="C1", alpha=0.2, label="lines")
plt.setp(lines_[1:], label="_")
axes0[0].set(ylabel="rented bikes")

idx_bike = np.argsort(bikes.temperature.values)
axes0[1].fill_between(
    bikes.temperature[idx_bike],
    hdi_lines["μ"][:, 0][idx_bike],
    hdi_lines["μ"][:, 1][idx_bike],
    color="C1",
    label="HDI",
    alpha=0.5,
)
for ax0 in axes0.flatten():
    ax0.plot(bikes.temperature, bikes.rented, "C2.", zorder=-3)
    ax0.plot(x_plot, mean_line, c="C0", label="mean line")
    ax0.set(xlabel="temperature")
    ax0.legend()
```

#figure(
  image("images/bbap/bap-04-bike-lines.png", width: 60%),
  caption: "自行车租赁的线性模型",
) <bike-lines>\

@bike-lines 传达了基本相同的信息，但一个将不确定性表示为一组线，另一个表示为阴影区域。请注意，若重复代码来生成绘图，您将得到不同的线，因为我们是从后验中采样的。然而，阴影区域将是相同的，因为我们正在使用所有可用的后验样本。

要强调的是，有不同的方法来表示不确定性。哪一个更好？像往常一样，这取决于上下文。阴影区域是一个不错的选择；它很常见，并且计算和解释都很简单。除非有特殊原因需要显示单个后部样本，否则阴影区域可能是首选。

== 解释后验预测

若我们不仅对预期均值感兴趣，而且想根据预测（即租赁自行车）来思考怎么办？好吧，为此，我们可以进行后验预测采样。

```python
pm.sample_posterior_predictive(idata_lb, model=model_lb, extend_inferencedata=True)
```

@bike-pred 中的黑线是租赁自行车数量的均值。这与@bike-lines 中的相同。新元素是代表租赁自行车中心 50%（分位数 0.25 和 0.75）的深灰色带和代表中心 94%（分位数 0.03 和 0.97）的浅灰色带。不过，我们的模型预测自行车数量为负，因为我们对 `model_lb`中的可能性使用正态分布。一个肮脏的修复可能是将预测值限制为低于 0。在下一节中，我们将改进这个模型以避免无意义的预测。

#figure(
  image("images/bbap/bap-04-bike-pred.png", width: 45%),
  caption: "自行车租赁的后验预测",
) <bike-pred>

```python
from scipy import interpolate

mean_line = idata_lb.posterior["μ"].mean(("chain", "draw"))
temperatures = np.random.normal(bikes.temperature.values, 0.01)
idx = np.argsort(temperatures)
x = np.linspace(temperatures.min(), temperatures.max(), 15)
y_pred_q = idata_lb.posterior_predictive["y_pred"].quantile(
    [0.03, 0.97, 0.25, 0.75], dim=["chain", "draw"]
)
y_hat_bounds = iter(
    [PchipInterpolator(temperatures[idx], y_pred_q[i][idx])(x) for i in range(4)]
)

_, ax = plt.subplots(figsize=(12, 5))
ax.plot(bikes.temperature, bikes.rented, "C2.", zorder=-3)
ax.plot(bikes.temperature[idx], mean_line[idx], c="C0")
for lb, ub in zip(y_hat_bounds, y_hat_bounds):
    ax.fill_between(x, lb, ub, color="C1", alpha=0.5)

ax.set(xlabel="temperature", ylabel="rented bikes")
```

= 广义线性模型

广义线性模型（generalized linear model，GLM）是线性模型的推广，它允许我们使用不同的似然分布。在较高层次上，我们可以编写一个 Bayesian GLM，如下所示：

$
  α & ∼ "a prior" \
  β & ∼ "another prior" \
  θ & ∼ "some prior" \
  μ & = α + β X \
  Y & ∼ φ(f(μ), θ)
$

$φ$是任意分布；一些常见的情况是正态、Student’s t、Gamma 和 NegativeBinomial。$θ$表示分布可能具有的任何辅助参数，例如正态分布的$σ$。我们还有$f$，通常称为逆链接函数（inverse link function）。

- 当$φ$为正态时，$f$为恒等函数
- 当$φ$为 Gamma 或 NegativeBinomial 等分布，$f$通常是指数函数\

为什么需要$f$？因为线性模型通常位于实线上，但$μ$参数（或其等效参数）可能定义在不同的域上。如，NegativeBinomial 的$μ$是为正值定义的，因此我们需要变换$μ$。

让我们从一个具体示例开始。我们如何更改`model_lb`以更好地适应自行车数据？有两点需要注意：租赁自行车的数量是离散的，并且以 0 为界。这通常称为计数数据。计数数据有时使用正态分布等连续分布进行建模，特别是当计数数量很大时。但使用离散分布通常是个好主意。两个常见的选择是 Poisson 和 NegativeBinomial。主要区别在于，对于 Poisson 来说，均值和方差相同，但若这不是真的，则 NegativeBinomial 可能是更好的选择，因为它允许均值和方差不同。

```python
with pm.Model() as model_neg:
    α = pm.Normal("α", mu=0, sigma=1)
    β = pm.Normal("β", mu=0, sigma=10)
    σ = pm.HalfNormal("σ", 10)
    μ = pm.Deterministic("μ", pm.math.exp(α + β * bikes.temperature))
    y_pred = pm.NegativeBinomial("y_pred", mu=μ, alpha=σ, observed=bikes.rented)
    idata_neg = pm.sample(random_seed=123)
    idata_neg.extend(pm.sample_posterior_predictive(idata_neg, random_seed=123))
```

这里，NegativeBinomial 的方差为$μ + μ^2 / α$ ，因此$α$值越大，方差越大。`model_neg`的后验预测分布如@bike-pred2。后验预测分布也与我们通过线性模型获得的分布非常相似（@bike-pred）。主要区别在于，现在我们预测的租赁自行车数量不是负数。

#figure(
  image("images/bbap/bap-04-bike-pred2.png", width: 45%),
  caption: "自行车租赁的后验预测改进",
) <bike-pred2>

@bike-ppc 显示了左侧`model_lb`和右侧`model_neg`的后验预测检查。我们可以看到，当使用 Normal 时，最大的不匹配是模型预测的租赁自行车数量为负，但即使从积极的一面来看，我们也发现拟合度并不那么好。另一方面，NegativeBinomial 模型似乎更适合，尽管它并不完美。看右尾：预测比观察更重要。但还要注意的是，这种非常高的需求的概率很低。

#figure(
  image("images/bbap/bap-04-bike-ppc.png", width: 60%),
  caption: "自行车租赁的两个模型的后验预测检查",
) <bike-ppc>

= 鲁棒回归
<鲁棒回归>

== t 推断

在处理离群值和 Gaussian 分布时，一个非常有用的选择是用 t 似然代替正态似然。该分布有 3 个参数：均值、标度和自由度。自由度通常使用字母$ν$，可在$[0, ∞]$的内变化，也称为正态性参数（normality parameter）。t 分布的一个的特征是，该分布的方差只对$ν > 2$进行定义。要注意 t 分布的标度与标准差不一样，当接近无穷大时，标度近似于标准差。

#figure(
  image("images/distrs/distr_t.png", width: 45%),
  caption: "t 分布",
)

对 Anscombe 四重奏的第三组数据拟合，可以看出，离群点使回归线发生了明显的便宜。

#figure(
  image("images/bbap/bap-04-ans-linreg.png", width: 50%),
  caption: "Anscombe 第三组",
)\

相比之下，t 分布允许我们有一个更鲁棒的估计，因为离群值具有减少$ν$的效应，而非将均值拉向它们，增加标准差。故，均值和标度是通过对大部分数据点的加权来估计的，而非对那些离群值的加权。但标度与数据的分布有关；其值越低，分布越集中。此外，作为经验法则，对于$ν > 2$，标度的值往往非常接近去除离群值后的估计量。

```python
ans = pd.read_csv("../data/anscombe.csv")
x_3 = ans.query("group == III")["x"].to_numpy().flatten()
y_3 = ans("group == III")["y"].to_numpy().flatten()
x_3 = x_3 - x_3.mean()

with pm.Model() as anscombe_t:
    α = pm.Normal("α", mu=y_3.mean(), sigma=1)
    β = pm.Normal("β", mu=0, sigma=1)
    ϵ = pm.HalfNormal("ϵ", 5)
    ν_ = pm.Exponential("ν_", 1/29)
    ν = pm.Deterministic("ν", ν_ + 1)
    μ = pm.Deterministic("μ", α + β * ans.x)
    y_pred = pm.StudentT("y_pred", mu=α + β * x_3, sigma=ϵ, nu=ν, observed=y_3)
    idata_t = pm.sample(2000, random_seed=4951)
```

这里，我们使用一个移位的指数（shifted exponential）$ν_("_")$来避免正态参数$ν$的值接近于零，因为没有移位的指数给了$0$附近的值上太多权重。如@ans，$α, β$和$ϵ$的范围非常窄，$ϵ ≈ 0$。运行一个后验预测检查来探索模型对数据的捕捉程度。

#let csv1 = csv("python/bap-02-ans.csv")
#figure(
  tableq(csv1, 10, inset: 0.31em),
  caption: "Anscombe III 鲁棒估计",
  kind: table,
) <ans>

#sgrid(
  figure(
    image("images/bbap/bap-04-ans-t.png", width: 50%),
    caption: "鲁棒回归",
  ),
  figure(
    image("images/bbap/bap-04-ans-ppc.png", width: 60%),
    caption: "后验预测检查",
  ),
  columns: (220pt,) * 2,
  gutter: 1pt,
  caption: "鲁棒模型",
)

不难看出，非鲁棒拟合试图涵盖所有的点，而鲁棒模型则自动抛弃一些点，并拟合一条正好穿过所有剩余点的线。t 分布由于其较重的尾部，能够减少对远离大部分数据的点的重视。重尾的意思是，与 Gaussian 分布相比，它更有可能找到远离均值的值。对于目前的目的来说，这个模型的表现还不错。

= 多元线性回归
<多元线性回归>

多元线性回归（multiple linear regression）允许我们同时考虑多个因素的影响，其表示如下

$ μ = α + sum_(i=1)^m β_i x_i $
线性代数形式为

$ μ = α + 𝑿 β $

在多元线性回归模型下，拟合的结果为一个维度为$m$的超平面。一个重要的信息是，在多元线性回归中，每个参数只有在其他参数的背景下才有意义。我们将使用当天的温度和湿度来预测租赁自行车的数量：

```python
with pm.Model() as model_mlb:
    α_ml = pm.Normal("α", mu=0, sigma=1)
    β0_ml = pm.Normal("β0", mu=0, sigma=10)
    β1_ml = pm.Normal("β1", mu=0, sigma=10)
    σ_ml = pm.HalfNormal("σ", 10)
    μ_ml = pm.Deterministic(
        "μ", pm.math.exp(α_ml + β0_ml * bikes.temperature + β1_ml * bikes.hour)
    )
    _ = pm.NegativeBinomial(
        "y_pred", mu=μ_ml, alpha=σ_ml, observed=bikes.rented
    )
    idata_mlb = pm.sample(random_seed=123)
```

比较一下 `model_mlb`和 `model_neg`。 唯一的区别是，现在我们有两个$β$系数，每个自变量都有一个。模型的其余部分是相同的。请注意，我们可以写成#raw("pm.Normal('β1', mu=0, sigma=10, shape=2)", lang: "python", block: false)，然后在$μ$的定义中使用 `β1[0]`和`β1[1]`。

现在我们必须小心指定我们正在谈论哪个自变量。例如，我们可以说，在保持小时值不变的情况下，温度每升高一个单位，租赁自行车的数量就会增加 $β_0$个单位。或者说，在保持温度恒定的情况下，每小时增加一个单位，租赁自行车的数量增加 $β_1$个单位。

#figure(
  image("images/bbap/bap-04-multireg-comp.png", width: 40%),
  caption: "简单线性回归 vs. 多元线性回归",
)\

可以看到两个模型的温度系数不同。这是因为温度对租赁自行车数量的影响取决于一天中的时间。 更重要的是，$β$系数的值已按其相应自变量的标准差进行缩放，因此我们可以使它们具有可比性。我们可以看到，一旦模型中包含小时数，温度对租赁自行车数量的影响就会变小。这是因为时间的影响已经解释了租赁自行车数量的一些变化，而这些变化之前是通过温度来解释的。在极端情况下，添加一个新变量可以使系数变为0，甚至改变符号。

= 变量方差
<变量方差>

我们一直在使用线性主题来模拟分布的均值。在统计学中，当所有观测值中误差方差不恒定时，线性回归模型就会呈现异方差性。对于这些情况，我们可能希望将方差（或标准差）视为因变量的（线性）函数。

世界卫生组织（WHO）和世界各地的其他卫生机构收集新生儿和幼儿的数据，并设计了成长图表标准。这些图表是儿科工具箱的重要组成部分，亦为度量人口总体福祉的标准，以便制定与健康有关的政策、规划干预措施并监测其有效性。一个例子是新生儿的身长与年龄（月）的关系。

#figure(
  image("images/bbap/bap-04-babies.png", width: 40%),
  caption: "新生儿身长数据集",
)\

为了对这些数据进行建模，与之前的模型相比，我们将引入 3 个新的元素：

- $ϵ$现在是$x$的线性函数。为此，我们增加了 2 个新的参数，$γ$和$δ$，它们是$α$和$β$的直接类似。
- 均值的线性模型是$sqrt(x)$的函数。这只是将线性模型拟合到曲线上的一个简单技巧。
- 定义了一个共享变量`x_shared`。我们将用它来改变$x$变量的值，在模型拟合后，无需重新拟合模型。

```python
with pm.Model() as model_vv:
    x_shared = pm.MutableData("x_shared", data.month.values.astype(float))
    α = pm.Normal("α", sigma=10)
    β = pm.Normal("β", sigma=10)
    γ = pm.HalfNormal("γ", sigma=10)
    δ = pm.HalfNormal("δ", sigma=10)
    μ = pm.Deterministic("μ", α + β * x_shared**0.5)
    σ = pm.Deterministic("σ", γ + δ * x_shared)
    y_pred = pm.Normal("y_pred", mu=μ, sigma=σ, observed=data.length)
    idata_vv = pm.sample(random_seed=123)
```

在@babies-pred 左图中，我们可以看到黑色曲线代表$μ$的均值，两条半透明带代表一个和两个标准差。在右侧面板上，我们将估计方差作为长度的函数，方差随着长度的增加而增加。

```python
posterior_v = az.extract(idata_vv)
μ_m = posterior_v["μ"].mean("sample").values
σ_m = posterior_v["σ"].mean("sample").values

_, axes_v = plt.subplots(1, 2, figsize=(12, 4))
axes_v[0].plot(babies.Month, babies.Length, "C0.", alpha=0.1)
axes_v[0].plot(babies.Month, μ_m, c="k")
axes_v[0].fill_between(
    babies.Month, μ_m + 1 * σ_m, μ_m - 1 * σ_m, alpha=0.6, color="C1"
)
axes_v[0].fill_between(
    babies.Month, μ_m + 2 * σ_m, μ_m - 2 * σ_m, alpha=0.4, color="C1"
)
axes_v[0].set(xlabel="Months", ylabel="Length")

axes_v[1].plot(babies.Month, σ_m)
axes_v[1].set(xlabel="Months")
axes_v[1].set_ylabel(r"$\bar \sigma$", rotation=0)
```

#figure(
  image("images/bbap/bap-04-babies-pred.png", width: 55%),
  caption: "新生儿长度预测",
) <babies-pred>

现在我们已经拟合了模型，我们可能想使用该模型来找出特定女孩的长度与分布的比较情况。回答问题的一种方法是向模型询问 0.5 个月大婴儿的可变长度分布。我们可以通过从长度为 0.5 的后验预测分布中采样来回答这个问题。我们可以通过采样 #raw("pm.sample_posterior_predictive", lang: "python", block: false) 得到答案；唯一的问题是，默认情况下，该函数将返回已观察到的$x$值，即用于拟合模型的值。获得未观察到的值的预测的最简单方法是定义一个`Data`变量（示例中为`x_shared`），然后在对后验预测分布进行采样之前更新该变量的值。

```python
with model_vv:
    pm.set_data({"x_shared": [0.5]})
    ppc = pm.sample_posterior_predictive(idata_vv, random_seed=123)
    y_ppc_v = ppc.posterior_predictive["y_pred"].stack(sample=("chain", "draw"))
```

#figure(
  image("images/bbap/bap-04-babies-ppc.png", width: 30%),
  caption: "半个月时的预期长度分布",
)

现在我们可以绘制 2 周大女孩的预期长度分布并计算其他数量，例如该长度女孩的百分位数

```python
ref_v = 52.5
grid_v, pdf_v = az.stats.density_utils._kde_linear(y_ppc_v.values)
plt.plot(grid_v, pdf_v)
percentile_v = int((y_ppc_v <= ref_v).mean() * 100)
plt.fill_between(
    grid_v[grid_v < ref_v],
    pdf_v[grid_v < ref_v],
    label="percentile = {:2d}".format(percentile_v),
    color="C2",
)
plt.xlabel("length")
plt.yticks([])
plt.legend()
```

= 分层线性回归
<分层线性回归>

本节，我们创建八个相关组，其中一个组只有一个数据点。

```python
N = 20
groups = ["A", "B", "C", "D", "E", "F", "G", "H"]
M_g = len(groups)
idx_g = np.repeat(range(M_g - 1), N)
idx_g = np.append(idx_g, 7)
np.random.seed(314)
α_real_g = np.random.normal(2.5, 0.5, size=M_g)
β_real_g = np.random.beta(6, 1, size=M_g)
ϵ_real_g = np.random.normal(0, 0.5, size=len(idx_g))
y_m_g = np.zeros(len(idx_g))
x_m_g = np.random.normal(0, 1, len(idx_g))
y_m_g = α_real_g[idx_g] + β_real_g[idx_g] * x_m_g + ϵ_real_g

_, ax_g = plt.subplots(2, 4, figsize=(10, 5), sharex=True, sharey=True)
j_g, k_g = 0, N
ax_g = ax_g.flatten()
for i, g in enumerate(groups):
    ax_g[i].scatter(x_m_g[j_g:k_g], y_m_g[j_g:k_g], marker=".")
    ax_g[i].set_title(f"group {g}")
    j_g += N
    k_g += N
```

#figure(
  image("images/bbap/bap-04-hier-groups.png", width: 60%),
  caption: "合成数据集",
)

分层线性模型有两种常见的参数化：中心化和非中心化。 中心化的特点是直接估计各个组的参数；如，明确估计每组的斜率；非中心化估计所有组的共同斜率，然后估计每个组的斜率。值得注意的是，我们仍在对每个组的斜率进行建模，但相对于公共斜率，我们获得的信息是相同的，只是表示方式不同。

```python
with pm.Model(coords=coords) as hierarchical_centered:
    # hyper-priors
    α_μ_h = pm.Normal("α_μ", mu=y_m_g.mean(), sigma=1)
    α_σ_h = pm.HalfNormal("α_σ", 5)
    β_μ_h = pm.Normal("β_μ", mu=0, sigma=1)
    β_σ_h = pm.HalfNormal("β_σ", sigma=5)
    # priors
    α_h = pm.Normal("α", mu=α_μ_h, sigma=α_σ_h, dims="group")
    β_h = pm.Normal("β", mu=β_μ_h, sigma=β_μ_h, dims="group")
    σ_h = pm.HalfNormal("σ", 5)
    _ = pm.Normal(
        "y_pred", mu=α_h[idx_g] + β_h[idx_g] * x_m_g, sigma=σ_h, observed=y_m_g
    )

    idata_cen = pm.sample(random_seed=123)
```

不同之处在于，对于模型`hierarchical_centered`，我们定义了$β ~ 𝒩(β_μ, β_σ)$，而对于`hierarchical_non_centered`，我们定义了$β = β_μ + β_"offset" * β_σ$。非中心参数化更有效：当我运行模型时，我只得到 2 个散度，而非像以前的 148 个。为了消除这些剩余的分歧，我们可能仍然需要增加 `target_accept`。要完全理解为什么这种重新参数化起作用，需要了解后验分布的几何形状，我们将在【马尔可夫抽样】中讨论这个问题。

```python
with pm.Model(coords=coords) as hierarchical_non_centered:
    # hyper-priors
    α_μ_hn = pm.Normal("α_μ", mu=y_m_g.mean(), sigma=1)
    α_σ_hn = pm.HalfNormal("α_σ", 5)
    β_μ_hn = pm.Normal("β_μ", mu=0, sigma=1)
    β_σ_hn = pm.HalfNormal("β_σ", sigma=5)
    # priors
    α_hn = pm.Normal("α", mu=α_μ_hn, sigma=α_σ_hn, dims="group")
    β_offset_hn = pm.Normal("β_offset", mu=0, sigma=1, dims="group")
    β_hn = pm.Deterministic("β", β_μ_hn + β_offset_hn * β_σ_hn, dims="group")
    σ_hn = pm.HalfNormal("σ", 5)
    _ = pm.Normal("y_pred", mu=α_hn[idx_g] + β_hn[idx_g] * x_m_g, sigma=σ_hn, observed=y_m_g)

    idata_ncen = pm.sample(random_seed=123, target_accept=0.85)
```

H组的估计仍然是不确定性较高的估计。@hier-comp 显示了八组中每一组的拟合线。我们可以看到我们成功地将一条线拟合到一个点。每条线都受到其他组线的通知，因此我们并没有真正将线调整为单个点。相反，我们正在将其他组中的点通知的线调整为单个点。

#figure(
  image("images/bbap/bap-04-hier-fit.png", width: 70%),
  caption: "分层模型拟合",
) <hier-comp>
