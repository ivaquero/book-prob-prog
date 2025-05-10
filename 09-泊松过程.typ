#import "lib/lib.typ": *
#show: qooklet.with(
  title: "Poisson 过程",
  author: "Yāng Xīnbīn",
  footer-cap: "Yāng Xīnbīn",
  header-cap: "实用概率建模",
  lang: "zh",
)

= Poisson 过程
<Poisson-过程>

Poisson 过程（PP）是在时间轴上的不同点发生的到达序列，因此特定时间间隔内的到达数具有 Poisson 分布。

指数分布与 Poisson 分布密切相关，这一点从使用$λ$来表示这两种分布的参数就可以看出。在下面我们会看到两者间的更多关联。

#figure(
  table(
    columns: 3,
    align: center + horizon,
    inset: 0.4em,
    stroke: three-line(rgb("000")),
    table.header([], [Poisson 分布], [指数分布]),
    [表达式], [$frac(λ^k, k!)e^(-λ)$], [$λ e^(-λ x)$],
    [期望], [$λ$], [$1 / λ$],
    [方差], [$λ$], [$1 / λ^2$],
    [$λ$含义], [单位时间内的平均发生次数], [平均发生时间间隔],
    [例子], [医院平均每小时出生的婴儿], [医院婴儿出生的时间间隔],
  ),
  caption: "Poisson 分布和指数分布",
  supplement: "表",
  kind: table,
)

== 一维 PP

#definition[
  若以下 2 个条件成立，则连续时间内的到达过程称为速率为$λ$的 PP：

  + 在长度为$t$的时间间隔内，到达的人数是一个服从$"Pois"(λ t)$的随机变量。
  + 不相邻区间内的到达人数是相互独立的。如$(0, 10)$、$[10, 12]$和$[15, ∞]$区间内的到达数是独立的。
]

通常，我们会假设时间线从$t = 0$开始，在这种情况下，我们在$(0, ∞)$上有一个 PP。但若我们希望时间线在两个方向上都是无限的，我们也可以使用相同的条件来定义$(-∞, ∞)$上的 PP。

考虑$(0, ∞)$上的 PP。设$N(t)$为$(0, t]$中的到达数。那么在$0 < t_1 < t_2$时，$(t_1，t_2]$中的到达人数为$N(t_2) - N(t_1)$。设$T_j$为第$j$次到达的时间。因为$T_1 > t$与$N(t) = 0$是同一事件

$ P(T_1 > t) = P(N(t) = 0) = e^(-λ t) $

所以$T_1 ∼ "Expo"(λ)$。那么，以第一次到达时间$T_1$为条件，第二次到达前的额外时间$T_2 - T_1$，也服从同参数的指数分布

$ (T_2 - T_1)|T_1 ∼ "Expo"(λ) $

这是因为有一个从$T_1$开始的全新 PP。由于$(T_2 -T_1)|T_1$的条件分布不依赖于$T_1$，因此，$T_2 - T_1$与$T_1$无关，所以$T_2 - T_1 ∼ "Expo"(λ)$也是无条件的。以此类推，到达时间间隔$T_j - T_j-1$相互独立

$ T_j - T_(j - 1) ∼ "Expo"(λ) $

因此，PP 可以被描述为到达次数为 Poisson 的过程，也可以被描述为到达时间为指数的过程。还要注意的是，由于$T_j$是$j$个独立同分布于$"Expo"(λ)$的随机变量的和

$ T_j ∼ "Gamma"(j, λ) $

#algo[
  从速率为$λ$的 PP 在$(0, ∞)$中生成$n$个到达：

  + 生成$n$个随机变量$X_1, …, X_n limits(∼)^("i.i.d") "Expo"(λ)$
  + 对于$j = 1, …, n$，设$T_j = X_1 + … + X_j$
  则我们可以得到$T_1, …, T_n$作为到达时间。
]

与指数函数的联系为我们提供了从 PP 中产生$n$个到达的简单方法。

== 理解 PP

PP 有 3 个最重要的性质，即条件性（conditioning）、叠加性（superposition）和稀释性（thinning）。

=== 条件性

当把 PP 作为区间内事件总数的条件时，得到的第一个结果是，以区间内的事件总数为条件，固定子区间内的事件数是二项分布的。

#theorem("条件计数")[
  设$(N(t): t > 0)$是一个速率为$λ$的 PP ，且$t_1 < t_2$。给定$N(t_2) = n$时，$N(t_1)$的条件分布为
  $ N(t_1)|N(t_2) = n ∼ "Bin"(n, t_1 / t_2) $
]

根据到达次数的条件这一思路，我们惊奇地发现：在 PP 中，给定$N(t) = n$，到达时间的分布就好像我们撒下$n$个独立同分布于$"Unif"(0, t)$的点。

#theorem[
  在速率为$λ$的 PP 中，以$N(t) = 1$为条件，第一次到达时间$T_1$服从$"Unif"(0, t)$。
]

#theorem("条件次数")[
  在速率为$λ$的 PP 中，$N(t) = n$的条件下，到达时间$T_1, …, T_n$与$n$个独立同分布于$"Unif"(0, t)$的顺序统计量的联合分布相同。
]

通过之前的章节，我们知道$"Unif"(0, 1)$随机变量的顺序统计量服从 Beta 分布，因此$T_j$的条件分布是缩放的 Beta 分布；要得到 Beta 分布，我们只需将$T_j$除以 $t$，这样它们的支撑集就是$(0, 1)$

$ t^(-1) T_j|N(t) = n ∼ "Beta" (j, n - j + 1) $

#algo[
  从速率为$λ$的 PP 在$(0, t\]$中生成$n$个到达：
  + 生成区间内的事件总数$N(t) ∼ "Pois"(λ t)$
  + 给定$N(t) = n$，生成$n$个随机变量$U_1, …, U_n limits(∼)^("i.i.d") "Unif"(0, t)$
  + 对于$j = 1, …, n$，设$T_j = U(j)$
]

=== 叠加性

叠加性是说，把 2 个独立的 PP 叠加起来，就会得到另一个 PP。（合并的过程本身也是 PP，但到达的时间会被标记，以表明它们来自于哪个基本过程）。

#theorem("叠加性")[
  设$(N_1(t): t > 0)$和$(N_2(t): t > 0)$分别是速率为$λ_1$和$λ_2$的独立 PP。那么组合过程$N(t) = N_1(t) + N_2(t)$是一个速率为$λ_1 + λ_2$的 PP。
]

#algo[
  生成速率为$λ_1$的独立 PP $(N_1(t): t > 0)$和速率为$λ_2$的独立 PP $(N_2(t): t > 0)$的叠加：
  + 生成 PP $(N_1(t): t > 0)$的到达
  + 生成 PP $(N_2(t): t > 0)$的到达
  + 叠加步骤 1 和 2 的结果
]

#theorem("事件 I 先于事件 II 的概率")[
  考虑 2 个独立 PP ：一个是速率为$λ_1$的事件I到达的 PP ，另一个是速率为$λ_2$的事件II到达的 PP。在这两个过程的叠加中，第一个到达者为事件I的概率为
  $λ_1 / (λ_1 + λ_2)$。
]

#algo[
  生成速率为$λ_1$和$λ_2$的 2 个独立 PP 的叠加：

  + 生成$X_1, X_2, … limits(∼)^("i.i.d") "Expo"(λ_1 + λ_2)$的随机变量，，并令第$j$个到达时间为$T_j = X_1 + … + X_j$
  + 生成独立于$X_1, X_2, …$的$I_1, I_2, … limits(∼)^("i.i.d") "Bern"(λ_1 / (λ_1 + λ_2))$的随机变量。若$I_j = 1$，则第$j$次到达为事件I，否则为事件II
]

#theorem("叠加到离散时间的投影")[
  考虑 2 个参数为$λ_1$和$λ_2$的独立 PP 的叠加$(N(t): t > 0)$，对于$j = 1, 2, …$，令$I_j$为第$j$个事件来自 PP 的示性，其速率为$λ_1$。则
  $ I_j limits(∼)^("i.i.d") "Bern"(λ_1 / (λ_1 + λ_2)) $
]

#theorem[
  假设$X ∼ "Expo"(λ)$，$Y|X = x ∼ "Pois"(x)$，则
  $ Y ∼ "Geom"(λ / (λ+1)) $
]

#theorem[
  假设$X ∼ "Gamma"(r, λ)$，$Y|X = x ∼ "Pois"(x)$，则
  $ Y ∼ "NBin"(r, λ / (λ + 1)) $
]

=== 稀释性

我们要讨论的最后一个性质是稀疏性：若将一个 PP，每次到达时都独立地掷一枚硬币来决定它是事件 I 还是事件 II，则我们最终会得到两个独立的 PP。

#theorem("稀释")[
  假设$(N(t): t > 0)$是一个速率为$λ$的 PP，并将每次到达分为概率为$p$的事件I和概率为$1 - p$的事件II，这两类时间相互独立，且与到达时间无关。则事件I形成一个速率为$λ p$的 PP，事件II形成一个速率为$λ(1 - p)$的 PP，这两个过程也是独立的。
]

#theorem("染色")[
  假设$(N(t): t > 0)$是一个速率为$λ$的 PP，C 是一个有限的"颜色"集合，从$1$到$c$。颜色分布相互独立，且与到达时间无关。令$(N_i (t): t > 0)$成为颜色$i$进程，即$N_i (t)$是$\(0, t\]$中带有颜色$i$的到达人数。则对于$i = 1, 2, …, c$，$(N_i (t): t > 0)$是一个速率为$λ p_i$的 PP，且这$c$个单色过程是独立的。
]

== 多维 PP

多维 PP 的定义与一维 PP 类似：我们只是用面积或体积的概念代替了长度的概念。为了具体化，我们将定义二维 PP，之后通过类比，我们也应该清楚如何定义更高维的 PP。

#definition[
  若以下条件成立，则平面$ℝ^2$中的事件是强度为$λ$的二维 PP：
  + 区域 A 中的事件数分布为$"Pois"(λ - "area"(A))$
  + 不相连区域中的事件数相互独立
]

条件性、叠加性和稀释性也适用于二维 PP。假设 $N(A)$是区域 A 中的事件数。假设$B ⊆ A$，给定$N(A) = n$，则$N(B)$的条件分布是二项分布：

$
  N lr((B)) divides N lr((A)) eq n tilde.op "Bin" lr((n comma frac("area" lr((B)), "area" lr((A)))))
$

以大区域 A 中的事件总数为条件，事件落入子区域的概率与子区域的面积成正比；因此，事件的位置是条件均匀的，我们可以在 A 中生成一个二维 PP，首先生成事件数$N(A) ∼ "Pois"(λ - "area"(A))$，然后在 A 中均匀随机地放置事件。

与一维情况一样，独立的二维 PP 的叠加性也是一个二维 PP ，且强度相加。我们还可以将一个二维 PP 稀释，得到独立的二维 PP。

正如 GP 是一个随机变量的集合，其中这些随机变量的每个有限集合都具有多元 Gaussian 分布一样，PP 是一个随机变量的集合，其中，这些随机变量的每个有限集合都具有 Poisson 分布。我们可把 PP 看成是给定空间中的点集合上的分布。由于 Poisson 分布的速率仅限于正值，所以使用指数作为激活函数。

PP 有许多扩展。我们可以允许$λ$作为时间或空间的函数而变化，而非保持不变；这被称为非均质 PP。下面我们将探索这些颇具代表性的随机过程。

= Cox 过程
<Cox-过程>

当 PP 的速率本身是一个随机过程时，就有了 Cox 过程（CP）。回到对计数数据建模的例子，我们将使用 Poisson 似然，速率将使用 GP 建模。由于在文献中，速率也以强度的名称出现，故这类问题又被称为强度估计（intensity estimation）。

== 矿难

矿难包括 1851 年至 1962 年英国的矿难记录。灾难的数量被认为是受这一时期安全法规变化的影响。我们希望将灾难发生率作为时间的函数进行建模。

#let csv1 = csv("data/coal.csv")
#figure(
  ktable(csv1.slice(0, 4), 1),
  caption: "矿难",
  supplement: "表",
  kind: table,
)

我们的拟合模型是

$
  f(x) & ∼ cal("GP")(μ_x, K(x, x^′))\
  y & ∼ "Poi"(f(x))
$

这是一个 Poisson 回归问题。但当只有灾害日期一列，则需要对数据进行离散化处理，就像建立直方图一样。我们使用 `bin` 的中心作为变量$x$，而每个 `bin` 的计数作为变量$y$。

```python
# discretize data
years = int((coal_df.max() - coal_df.min()).values[0])
bins = years // 4
hist, x_edges = np.histogram(coal_df, bins=bins)
# compute the location of the centers of the discretized data
x_centers = x_edges[:-1] + (x_edges[1] - x_edges[0]) / 2
# arrange xdata into proper shape for GP
x_data = x_centers[:, None]
# express data as the rate number of disaster per year
y_data = hist / 4

with pm.Model() as model_coal:
    ℓ = pm.HalfNormal("ℓ", x_data.std())
    cov = pm.gp.cov.ExpQuad(1, ls=ℓ) + pm.gp.cov.WhiteNoise(1e-5)
    gp = pm.gp.Latent(cov_func=cov)
    f = gp.prior("f", X=x_data)

    y_pred = pm.Poisson("y_pred", mu=pm.math.exp(f), observed=y_data)
    idata_coal = pm.sample(1000, chains=1)
```

下图用一条白线显示了灾害率中值随时间变化的情况。带状线描述了 50% HPD 区间（较深）和 94% HPD 区间（较浅）。可看到，事故率随着时间的推移而降低，除了最初的短暂增加。请注意，即使我们对数据进行了分层，结果也得到了一条平滑的曲线。在此意义上，我们可把 `model_coal` 这类模型看作是建立一个直方图，然后平滑它。

#figure(
  image("images/bbap/bap-08-cp-coal-hdi.png", width: 40%),
  caption: "矿难 HDI",
  supplement: "图",
)

== 红木

现在将刚才的同类型模型应用到二维空间问题上，使用红木数据。数据集由给定区域内红木的位置组成，目标是确定树木的速率在此区域是如何分布的。

#let csv1 = csv("data/redwood.csv")
#figure(
  ktable(csv1.slice(0, 4), 2),
  caption: "红木",
  supplement: "表",
  kind: table,
)

和流感模型一样，先对数据离散化。这里，我们没有做筛网（mesh grid），而是将 $x_1$ 和$x_2$数据分开处理，从而为每个坐标建立一个协方差矩阵，有效地减少了计算 GP 所需矩阵的大小。我们只需要在使用 `LatentKron()` 来定义 GP。需要注意的是，这不是一个数值技巧，而是这类矩阵结构的数学属性，故我们并没有在模型中引入任何近似或误差，我们只是用一种可以加快计算速度的方式来表达。

```python
with pm.Model() as model_rw:
    ℓ = pm.HalfNormal("ℓ", rw_df.std().values, shape=2)
    cov_func1 = pm.gp.cov.ExpQuad(1, ls=ℓ[0])
    cov_func2 = pm.gp.cov.ExpQuad(1, ls=ℓ[1])

    gp = pm.gp.LatentKron(cov_funcs=[cov_func1, cov_func2])
    f = gp.prior("f", Xs=x_data)

    y = pm.Poisson("y", mu=pm.math.exp(f), observed=y_data)
    idata_rw = pm.sample(1000)
```

图中，浅色的树木比深色的树木生长率高。可想象，我们对寻找高生长率区感兴趣，因为我们可能对木材如何从火灾中恢复感兴趣，或者我们可能对土壤的特性感兴趣而用树木作为近似（proxy）。

#figure(
  image("images/bbap/bap-08-cp-redwood-heatmap.png", width: 40%),
  caption: "红木的速率",
  supplement: "图",
)
