#import "lib/lib.typ": *
#show: chapter-style.with(
  title: "交互模型",
  info: info,
)

= 多项式回归
<多项式回归>

使用线性回归模型拟合曲线的一种方法是建立一个多项式

$ μ = ∑_(i = 0)^m β_i x^i $

如二次多项式回归

$ μ = β_0 + β_1 x + β_2 x^2 $

使用 Anscombe 四重奏的第二组构建模型

```python
x_2 = ans.query("group == 'II'")["x"].to_numpy().flatten()
y_2 = ans.query("group == 'II'")["y"].to_numpy().flatten()
x_2 -= x_2.mean()

with pm.Model() as ans_poly:
    α = pm.Normal("α", mu=y_2.mean(), sigma=1)
    β1 = pm.Normal("β1", mu=0, sigma=1)
    β2 = pm.Normal("β2", mu=0, sigma=1)
    ϵ = pm.HalfCauchy("ϵ", 5)
    μ = α + β1 * x_2 + β2 * x_2**2

    pm.Normal("y_pred", mu=μ, sigma=ϵ, observed=y_2)
    idata_poly = pm.sample(2000)
```

这里的系数$β$不再是斜率，交互作用取决于$x$的区间。原则上，我们可使用多项式回归来拟合一个任意的复杂模型。但在实践中，阶数高于$3$的多项式一般不是很有用的模型，任何真实的数据集都会包含噪声。一个任意的过于复杂的模型会拟合噪声，导致糟糕的预测。这就是所谓的过拟合（overfitting），是统计学和机器学习中普遍存在的现象。此时可另选其他模型，如 Gaussian 过程。

#figure(
  image("images/bbap/bap-04-ans-poly-fit.png", width: 40%),
  caption: "Anscombe 第二组",
)

= 交互作用

对多元模型，很多时候还需考虑变量间的交互作用，引入交叉项

$ μ = α + β_1 x_1 + β_2 x_2 + β_3 x_1 x_2 $

重写可得

$
  μ &= α + underbrace((β_1 + β_3 x_2), "slope of" med x_1) x_1 + β_2 x_2\
  μ &= α + β_1 x_1 + underbrace((β_2 + β_3 x_1), "slope of" med x_2)
$

这向我们展示了以下视角

- 交互项可理解为一个线性模型。故均值$μ$是一个线性模型
- 交互作用是对称的
- 没有交互作用的多元回归模型中，得到的是一个超平面，而交互项在超平面中引入了曲率
