#import "lib/lib.typ": *

#cover(info)
#contents(depth: 1, info: info)

#let chapter(filename) = {
  include filename
  context counter(heading).update(0)
}

#chapter("01-概率编程简介.typ")
#chapter("02-分层模型.typ")
#chapter("03-线性回归.typ")
#chapter("04-线性分类.typ")
#chapter("05-模型选择.typ")
#chapter("06-交互模型.typ")
#chapter("07-混合模型.typ")
#chapter("08-高斯过程.typ")
#chapter("09-泊松过程.typ")
#chapter("10-非马尔可夫抽样.typ")
#chapter("11-马尔可夫抽样.typ")
#chapter("12-样本诊断.typ")
