#import "lib/lib.typ": *
#show: chapter-style.with(
  title: "分层模型",
  info: info,
)

= 组间比较
<组间比较>

一种非常常见的统计分析是组间比较。我们可能对患者对某种药物的反应如何、新交通法规的出台是否减少了车祸、学生在不同教学方法下的表现等感兴趣。有时，这类问题是在假设检验场景下提出的，目的是宣布结果具有统计显著性。仅依靠统计显著性可能会出现问题，原因有很多：一方面，统计显著性不等同于实际显著性；另一方面，只要收集足够的数据，就可以宣布非常小的影响具有显著性。

假设检验的概念与 p 值的概念相关。大量的研究和论文表明，p 值的使用和解释往往是错误的。这里我们不做假设检验，而专注于估计效应量（effect size），即量化组间差异。从效应量的角度思考的一个好处是，我们从"是 - 否"这样的问题转向更细微的问题"效果如何？"。当有人说某件事情更难、更好、更快、更强时，记得问一下用于比较的基线是什么。为了比较组别，必须决定我们要用哪几个特征来进行比较。一个很常见的特征是每组的均值。我们将努力获得组间均值差异的后验分布，而不仅仅是差异的点估计。

== 基本模型

本节中，我们将使用 tips 数据集@bryantPracticalDataAnalysis1995 。我们想研究星期几对在餐厅赚取的小费的影响。在这个例子中，不同的组是日期。请注意，没有对照组或治疗组。若我们可以任意设定一天（如星期四）作为参考或对照。

#let data = csv("data/tips.csv")
#figure(
  tableq(data.slice(0, 6), 7),
  caption: "tips 数据集",
  supplement: [表],
  kind: table,
)\

首先，我们使用#raw("az.plot_forest()", lang: "python", block: false) 对每天的小费金额可视化。然后对这 4 天进行编码。

```python
az.plot_forest(
    tips.pivot_table(index=tips.index, columns="day", values="tip").to_dict("list"),
     kind="ridgeplot", hdi_prob=1, colors="C1"
)
categories = np.array(["Thur", "Fri", "Sat", "Sun"])
tip = tips["tip"].to_numpy()
idx = pd.Categorical(tips["day"], categories=categories).codes
```

#figure(
  image("images/bbap/bap-03-tips-forest.png", width: 40%),
  caption: "tips 数据集森林图",
)

比照此前对 Gaussian 模型，将其中的参数变为向量。我们将指定两个坐标：`days`，其维度为`Thur`，`Fri`，`Sat`，`Sun`；以及`days_flat`，它将包含相同的标签，但根据与每个观察相对应的顺序和长度重复。`days_flat`稍后将有助于后验预测测试。

```python
coords = {"days": categories, "days_flat": categories[idx]}
with pm.Model(coords=coords) as comparing_groups:
    μ = pm.HalfNormal("μ", sigma=5, dims="days")
    σ = pm.HalfNormal("σ", sigma=1, dims="days")
    y = pm.Gamma("y", mu=μ[idx], sigma=σ[idx], observed=tip, dims="days_flat")

    idata_cg = pm.sample(random_seed=4591)
    idata_cg.extend(pm.sample_posterior_predictive(idata_cg, random_seed=4591))
```

一旦计算出后验分布，我们就可以进行所有我们认为相关的分析。从下图中可以看出，该模型可以捕捉到分布的一般形状，但仍然有一些细节难以捉摸。这可能是由于样本量相对较小，除日期之外还有其他因素影响小费，或者两者兼而有之。

```python
_, axes = plt.subplots(2, 2, figsize=(10, 5), sharex=True, sharey=True)
az.plot_ppc(
    idata_cg,
    num_pp_samples=100,
    colors=["C1", "C0", "C0"],
    coords={"days_flat": [categories]},
    flatten=[],
    ax=axes,
)
```

#figure(
  image("images/bbap/bap-03-tips-ppc.png", width: 60%),
  caption: "tips 数据集后验概率检查",
)

现在，我们将认为该模型对我们来说已经足够好了，并开始探索后验。我们可用均值来解释结果，然后找出哪些日子的均值更高。但还有其他选择；我们可能希望使用一些受观众欢迎的效应大小测量方法，如 Cohen's d 或优越性概率。

== Cohen's $d$

度量效应量的常用方法是 Cohen's $d$，其定义为

$ δ = frac(μ_2 - μ_1, sqrt((σ_2^2 + σ_1^2)\/ 2)) $

根据这个表达式，效应量是指两组的均值差与合并标准差（pooled standard deviation）的商。Cohen's $d$越小，则组间差异越小。我们可得到均值和标准差的后验分布，计算出 Cohen's $d$值的后验分布。一般来说，在计算合并标准差时，我们会明确考虑到每组的样本量，但上面的公式省略了两组的样本量（因为这里的样本量是相等的）。

Cohen's $d$通过使用其标准差来引入每组的差异性。故，包括各组的内在变化是一种将差异置于背景中的方法。标准化差异有助于理解组间差异的重要性。即使均值的差异是标准化的，我们可能仍然需要根据给定问题的背景来校准。有一个非常好的网页可探索 Cohen's $d$的不同值：#link("https://rpsychologist.com/d3/cohend")[Cohen's d]。

#tip[
  Cohen's $d$可解释为 Z-score。Z-score 是有符号的标准差，是一个值与被观察的均值之间的差异。故，Cohen's $d$为 0.5，可解释为一组与另一组的标准差为 0.5。
]

== 优越性概率
<优越性概率>

优越性概率（probability of superiority，PS）是报告效应量的另一种方式，被定义为从一组随机抽取的数据点比同样从另一组随机抽取的数据点具有更大数值的概率。若设我们使用的数据是 Gaussian 分布，我们可用下面的表达式从 Cohen's $d$来计算优越性概率

$ "ps" = Φ(δ / sqrt(2)) $

其中，$Φ$是 Gaussian CDF，$δ$是 Cohen's $d$。我们可计算出优越性概率的点估计量，也可计算出整个后验分布的值。若我们对正态性设没有意见，可用这个公式从$δ$中得到优越性概率，否则，考虑到我们有来自后验的样本，我们可使用 MCMC 直接计算它。

== 均值差的后验分析

为了结束我们之前的讨论，让我们计算均值差异、Cohen’s d 和优越性概率的后验分布，并将它们整合到一个图中，即@tips-post。

#figure(
  image("images/bbap/bap-03-tips-post.png", width: 60%),
  caption: "均值差、Cohen's d 和 优越性概率",
) <tips-post>

解读@tips-post 的一种方法是将零差异的参考值与 HDI 间隔进行比较。只有一种情况是 94% HDI 排除了参考值，即周四和周日的小费差异。对于所有其他比较，我们不能排除差异为零的可能性，至少根据 HDI 参考值重叠标准。但即使在那种情况下，平均差异也约为 0.5 美元。正式地说，这需要定义一个损失函数，或者至少定义一些效应大小的阈值，来帮助我们做决策。

= 共享信息
<共享信息>

分层模型又称为多级模型或混合效应模型。它们在处理可描述为分组或具有不同级别的数据时特别有用，例如嵌套在地理区域内的数据（例如，属于一个省份的城市和属于一个国家的省份），或具有分层结构的数据（例如嵌套在学校内的学生，或嵌套在医院内的患者）或对同一个体的重复测量。

#figure(
  image("models/model-groups.png", width: 60%),
  caption: "分层模型的三种建模方法",
) <groups> \

分层模型是群体间共享信息的一种自然方式。在分层模型中，先验分布的参数本身被赋予先验分布。这些更高级别的先验通常称为超先验。拥有超先验允许模型在群体间共享信息，同时仍允许群体间存在差异。换句话说，我们可以将先验分布的参数视为属于一个共同的参数群体。@groups 显示了池化模型（单个组）、非池化模型（所有分离的组）和分层模型（也称为部分池化模型）之间的差异。

== 分层漂移

蛋白质是由 20 个氨基酸单元组成的分子。每个氨基酸可以在蛋白质中出现 0 次或更多次。正如旋律由音符序列定义一样，蛋白质由氨基酸序列定义。一些音符变化会导致旋律的细微变化，而其他变化则会导致完全不同的旋律。蛋白质的情况也类似。研究蛋白质的一种方法是使用核磁共振（与医学成像所用的技术相同）。这种技术使我们能够测量各种量，其中之一称为化学位移。

假设我们想将计算化学位移的理论方法与实验观测值进行比较，以评估理论方法重现实验值的能力。现在我们有了数据，我们该怎么做呢？一种选择是采用经验差异并拟合 Gaussian 或学生 t 模型。由于氨基酸是一类化合物，因此假设它们都是相同的，并估计所有差异的单个 Gaussian 分布是合理的。但你可能会说，每种氨基酸都有不同的化学性质，因此更好的选择是拟合 20 个独立的 Gaussian 分布。我们该怎么办？

- 若将所有数据结合起来，我们的估计值会更准确，但我们将无法从单个组（氨基酸）中获取信息。
- 若我们将它们视为单独的组，我们将获得更详细的分析，但准确性会降低。\
  此时，我们可以建立一个分层模型；这样，我们就可以进行组级估计，但限制是它们都属于更大的组或群体。为了了解非分层（非池化）模型和分层模型之间的区别，我们将构建2个模型。首先是非分层模型

```python
with pm.Model(coords=coords_chem) as cs_nh:
    μ_ = pm.Normal("μ", mu=0, sigma=10, dims="aa")
    σ_ = pm.HalfNormal("σ", sigma=10, dims="aa")
    y_ = pm.Normal("y", mu=μ_[idx_chem], sigma=σ_[idx_chem], observed=diff)

    idata_cs_nh = pm.sample(random_seed=4591)
```

现在，我们将构建模型的分层版本。我们添加两个超先验，一个用于$μ$的均值，一个用于$μ$的标准差。我们不给$σ$添加超先验；换句话说，我们假设观测值和理论值之间的方差对于所有组都应该相同。

```python
with pm.Model(coords=coords_chem) as cs_h:
    # hyper_priors
    μ_mu = pm.Normal("μ_mu", mu=0, sigma=10)
    μ_sd = pm.HalfNormal("μ_sd", 10)
    # priors
    _μ_ = pm.Normal("μ", mu=μ_mu, sigma=μ_sd, dims="aa")
    _σ_ = pm.HalfNormal("σ", sigma=10, dims="aa")
    # likelihood
    _y_ = pm.Normal("y", mu=_μ_[idx_chem], sigma=_σ_[idx_chem], observed=diff)
    # sample
    idata_cs_h = pm.sample(random_seed=4591)
```

当我们想要比较不同模型的参数值时，#raw("az.plot_forest()", lang: "python", block: false)很有用。@forest 中，我们绘制了 40 个估计均值的图，两个模型中每个氨基酸（20）一个。我们还有它们的 94% HDI 和分位数间距（分布的中心 50%）。垂直虚线是根据分层模型得出的全局均值。该值接近于零，正如预期的那样，理论值忠实地再现了实验值。

```python
_axes = az.plot_forest(
    [idata_cs_nh, idata_cs_h],
    model_names=["non_hierarchical", "hierarchical"],
    var_names="μ",
    combined=True,
    r_hat=False,
    ess=False,
    figsize=(8, 6),
    colors="cycle",
)
y_lims = _axes[0].get_ylim()
_axes[0].vlines(
    idata_cs_h.posterior["μ_mu"].mean(), *y_lims, color="k", ls=":"
)
```

#figure(
  image("images/bbap/bap-03-chem-forest.png", width: 50%),
  caption: "蛋白质化学漂移的分层模型",
) <forest> \

该图最相关的部分是，分层模型的估计值被拉向部分合并的均值，或者说，与未合并的估计值相比，它们缩小了。您还会注意到，对于那些远离均值的组（例如 PRO），这种影响更为明显，并且不确定性与非分层模型的不确定性相当或更小。估计值是部分合并的，因为我们对每个组都有一个估计值，但各个组的估计值通过超先验相互限制。因此，我们得到了一个中间情况，介于一个包含所有化学位移的组和 20 个独立组（每个氨基酸一个）之间。

= 球员

各种数据结构都适合分层描述，可​​以涵盖多个级别。如，考虑职业足球运动员。与许多其他运动一样，球员有不同的位置。我们可能有兴趣估计每个球员、位置和整个职业足球运动员群体的一些技能指标。

#let data = csv("data/football_players.csv")
#figure(
  tableq(data.slice(0, 6), 4),
  caption: "足球运动员数据集",
  supplement: [表],
  kind: table,
)\

我们收集了4年间（2017 年至 2020 年）英超、法甲、德甲、意甲和西甲的数据。假设我们对射门进球数指标感兴趣，我们可用二项式模型来估计它，其中参数$n$是射门次数，观测值$y$是进球数。这给我们留下了一个未知的$p$值。我们用$θ$表示每个球员的成功率，它是一个大小为`n_players`的向量，我们现在使用 Beta 分布对其进行建模。

Beta 分布的超参数将是向量$μ_p$和$ν_p$，它们是大小为 4 的向量，代表我们数据集中的4个位置（后卫 DF、中场 MF、前锋 FW 和守门员 GK）。我们需要正确索引向量$μ_p$和$ν_p$以匹配球员总数。最后，我们将有2个全局参数$μ$和$ν$，代表职业足球运动员。

在 PreliZ 的帮助下，#raw("pm.Beta('μ', 1.7, 5.8)", lang: "python", block: false)被选为先验，其 95% 的质量介于 0 和 0.5 之间。体育统计数据已经得到充分研究，并且有大量先验信息可用于定义更强的先验。可以对先验 #raw("pm.Gamma('ν', mu=125, sigma=50)", lang: "python", block: false) 进行类似的论证，我们将其定义为最大熵 Gamma 先验，其 90% 的质量介于 50 和 200 之间：

```python
pos_idx = football.position.cat.codes.to_numpy()
pos_codes = football.position.cat.categories
n_pos = pos_codes.size
n_players = football.index.size
coords_ball = {"pos": pos_codes}

with pm.Model(coords=coords_ball) as model_football:
    # Hyper parameters
    μ_f = pm.Beta("μ", 1.7, 5.8)
    ν_f = pm.Gamma("ν", mu=125, sigma=50)
    # Parameters for positions
    μ_p = pm.Beta("μ_p", mu=μ_f, nu=ν_f, dims="pos")
    ν_p = pm.Gamma("ν_p", mu=125, sigma=50, dims="pos")
    # Parameter for players
    θ_f = pm.Beta("θ", mu=μ_p[pos_idx], nu=ν_p[pos_idx])
    _ = pm.Binomial(
        "gs", n=football.shots.values, p=θ_f, observed=football.goals.values
    )

    idata_football = pm.sample(
        draws=3000, target_accept=0.95, random_seed=4591
    )
```

@ball-post 的上图中，我们得到了全局参数$μ$的后验分布。后验分布接近 0.1。这意味着，对于一名职业足球运动员（来自顶级联赛）来说，总体而言，进球的概率平均为 10%。这是一个合理的值，因为进球不是一件容易的事，而且我们没有区分位置。在中图中，我们得到了前锋位置的估计$μ_p$值；正如预期的那样，它高于全局参数$μ$。在下图中，我们得到了 Messi 的估计值$θ=0.17$，高于全局参数$μ$和前锋位置$μ_p$值。

#figure(
  image("images/bbap/bap-03-football-post.png", width: 50%),
  caption: "进球的后验分布",
) <ball-post>

@ball-forest 显示了参数$μ_p$的后验分布的森林图。正如我们已经看到的，前锋位置的后验分布以 0.13 为中心，是四个中最高的。$μ_p$的最低值是守门员位置。有趣的是不确定性非常高；这是因为我们的数据集中进球的守门员很少，准确地说是三个。后卫和中场位置的后验分布略居中，中场的后验分布略高。我们可以解释这一点，因为中场的主要作用是防守和进攻，因此进球的概率高于后卫，但低于前锋。

#figure(
  image("images/bbap/bap-03-football-forest.png", width: 40%),
  caption: "进球的森林图",
) <ball-forest>

= 因果推断

概率编程的另一个重要课题是，因果推断。其思想来源于，数据的原因不能仅从数据中提取。*没有原因，就不会有结果*。因果推断重点关注如下 3 个问题：

- 变量的关联（association，≠相关）
- 干预（intervention）的预测
- 缺失观测的填补（imputation）\
  知道原因意味着

- 能够预测干预的后果
- 能够构建未观察到的反事实结果\
  即使目标是描述性的，也需要因果模型。因为样本与总体不同，要描述总体需要对因果思考。

#warning[
  经典假设检验的风险
  - 许多过程会产生相似的概率分布
  - 不同的零假设，会得到相互矛盾的结论
]

== DAG

有向无环图（directed acyclic graph，DAG），是一个没有有向循环的、有限的有向图。它可以帮助明晰问题，"在没有额外假设的情况下，我们能做出什么决定？"。

#figure(
  image("models/srt-ch01-dag.png", width: 30%),
  caption: "DAG",
)

在一个因果模型中，加入所有元素往往是不明智的。通过 DAG 递进查询（query），可以逼近真相。

由上图，可以写出如下关系

$
  Y &∼ X \
  Y &∼ X + A \
  Y &∼ X + A + B \
  Y &∼ X + C \
  Y &∼ X + A + C \
  Y &∼ X + B + C
$

但，实际上的模型只需要最后一个关系式，即

$ Y ∼ X + B + C $

#bibliography("lib/prob.bib", style: "future-science")
