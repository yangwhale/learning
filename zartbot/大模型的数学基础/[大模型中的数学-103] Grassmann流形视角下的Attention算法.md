# [大模型中的数学-103] Grassmann流形视角下的Attention算法

> 作者: zartbot  
> 日期: 2026年5月25日 00:37  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247498836&idx=1&sn=44dfae6cd437e26c5d96e393e844345b&chksm=f995ec96cee2658099def1a5ae67f7c3d11a4670084a93af941a06aa3b88e342bd2a66930c4d#rd

---

本系列专题的前两篇:

[《[大模型中的数学-101] 大语言模型中的“语言”: 是符号的幻影, 还是意义的载体?》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247498668&idx=1&sn=30e82d5b87ad65ccc7881eb64d112a57&scene=21#wechat_redirect)

[《[大模型中的数学-102] 范畴论视角下的Attention算法》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247498689&idx=1&sn=14fb7e34fedb7335922cd3ed117b500b&scene=21#wechat_redirect)

### TL;DR

前两篇介绍了一些范畴论的概念和相应的算法, 这篇我们进一步考虑压缩, 当序列长度  增长到  量级时, 稀疏注意力机制如何设计的问题, 主要目的是解决分块如何构建, 基于分块下的稀疏选择该如何进行? 因此在这里引入了Grassmann流形来进行处理.

![图片](assets/f2ffe61332d4.png)

它将块映射到 Grassmann 流形上, 然后针对稀疏的top-k选块机制, 换成了路由+DPP的机制, 最后在选中的块上执行Attention算法.

![图片](assets/ec4f223556c7.png)

渣注
其实利用分块和Sparse Attention的做法来处理长上下文已经成为显学, 例如DeepSeek-V4 的 CSA 和 HCA. 但是我们如何进一步提高算法的效率来应对未来 5M Context下可计算的需求? 这里是引入Grassmann流形的根本原因, 主要区别有两个, 一个是子空间的构建映射到Grassmann流形上, 然后Top-K块的选择利用了 Plucker-DPP的算法, 最后通过在纤维丛的Attention机制来处理. 最终达到压线性复杂度的Attention计算.

本文目录入如下:

```
5. 压缩! Grassmann流形视角: 长序列注意力的几何本质5.1 从"点对点"到"子空间对子空间"5.2 Grassmann 流形的几何基础5.2.1 从正交基到子空间: 商空间的诞生5.2.2 切空间的分解: 水平方向与纤维方向5.2.3 对称空间结构与曲率5.2.4 平行移动与 Riemannian Adam5.2.5 子空间的三种表示及其适用场景5.2.6 度量族与距离选择5.3 无分解约束下的流形计算5.3.1 为什么禁止 SVD 和 QR: 从理论到工程的考量5.3.2 Retraction 的设计演进: 从指数映射到 Cayley 变换5.3.3 Riemannian Adam: 从欧氏到流形的三处修改5.4 从几何到注意力: 子空间路由的设计演进5.4.1 核心思想: 子空间级别的信息选择5.4.2 Stage 1: 子空间构建 5.4.3 Stage 2: 子空间路由, 从投影能量到外积对齐5.4.4 多样性保证: 从 top-r 到 Plücker-DPP5.4.5 梯度稳定性: 几何视角下的风险与对策5.5 纤维丛驱动的注意力架构5.5.1 从描述性框架到可执行算法5.5.2 完整的纤维丛建模5.5.3 联络: 跨块信息传输的三种方案5.5.4 Holonomy 与自适应策略5.5.5 纤维丛注意力: 等变性保证与维度优势5.5.6 双层注意力架构5.5.7 规范场论视角: Yang-Mills 正则化5.5.8 层间子空间传递: 三种策略5.5.9 规范等变池化与全局截面5.5.10 纤维丛算法的复杂度汇总5.6 GSA算法5.6.1 Overview5.6.2 一些通用的算子5.6.3 Stage 1: 子空间构建模块5.6.4 Stage 2: 三级路由策略5.6.5 Stage 3: 块内注意力计算 (纤维内注意力)5.6.6 完整的GSA算法5.6.7 KVCache处理5.7  Grassmann注意力与范畴论的统一
```

## 5. 压缩! Grassmann流形视角: 长序列注意力的几何本质

前述范畴论算法在短-中等序列上具有理论优势, 但当序列长度  增长到  量级时(如长文档理解、代码仓库分析、基因组序列), 即使是窗口化的TC-SimAttn也面临全局信息捕获的困难. 本节引入**Grassmann流形** 的几何视角, 为长序列注意力的复杂度问题提供一个新解法.

### 5.1 从"点对点"到"子空间对子空间"

标准注意力的  复杂度源于一个隐含假设: **每个token是注意力计算的原子单位**. 在长序列中, token 的语义表示往往呈现低维子空间结构. 相邻位置的 Key 向量倾向于落在某个低维子空间附近, 而不同语义段落对应不同的子空间方向. 这种观察自然引出一个问题: 能否在子空间层面而非 token 层面做路由和选择?

Grassmann 流形  恰好提供了回答这一问题的数学框架. 它是  中所有  维子空间的集合, 天然配备了度量、曲率和优化结构. 将 KV 序列按块划分, 每个块提取一个  维主方向子空间, 就得到了 Grassmann 流形上的一组点. Query 的路由问题随之转化为 Grassmann 流形上的最近邻搜索, 而 KV Cache 的压缩则对应于从  维表示到  维子空间坐标的投影.

接下来, 我们从 Grassmann 流形的微分几何基础出发, 经由纤维丛理论的算法化, 构建 Grassmann 子空间注意力 (Grassmann Subspace Attention, GSA) 机制. 其核心设计约束包括:

*亚线性复杂度*: 当块大小  时, 计算和内存均为 

*完全可微分*: 端到端训练, 梯度信号不被截断

*梯度稳定*: 利用 Grassmann 流形的紧致性和非负曲率保证优化稳定

### 5.2 Grassmann 流形的几何基础

#### 5.2.1 从正交基到子空间: 商空间的诞生

理解 Grassmann 流形需要从一个看似简单的问题开始: 如何表示  中的一个  维子空间?

设我们在三维空间 () 中, 如何描述一个二维平面 (一个  的子空间)?

我们可以给出一个方程, 比如 .

我们可以给出一组基, 比如向量  和 .

![图片](assets/374da802c891.png)

对于高维空间和机器学习应用, "给出一组基" 的方式更方便、更具操作性. 但这立刻引出了下一个关键问题: **同一个子空间, 可以被无穷多组不同的基所表示**

最直接的方式是给出一组正交基. 令  满足 , 则  唯一确定一个  维子空间. 满足正交约束的所有这样的矩阵构成 *Stiefel 流形*:

**: 这是一个  的矩阵. 我们可以把它想象成  个列向量, 每个向量都是  维的.

**: 这是关键的**正交约束**.  是  的转置.  是  的单位矩阵. 这个等式展开来看, 意味着  的任意一列的长度都是1 (归一性), 并且任意两列都相互垂直 (正交性).换句话说, ** 的  个列向量构成了一组标准正交基**.

**: 由  的这  个列向量所**张成 (span)** 的空间. 这唯一地确定了一个  维子空间.

**Stiefel 流形 **: 所有满足这个正交约束的  矩阵的集合, 构成了一个**流形**. 流形是一个局部看起来像欧几里得空间的光滑空间.  就是所有可能的"k维标准正交标架(k-frame)"的集合.

然而, 同一个子空间有无穷多组正交基. 对于任意正交矩阵 ,  和  张成同一个子空间, 却是 Stiefel 流形上的不同点.

**: 这是一个  的正交矩阵. 它的作用是在那个  维子空间**内部**进行一次**旋转或反射**.

**: 将  的  个列向量进行一次线性组合, 这个组合由  定义. 由于  是正交的, 组合后的新向量组  仍然是标准正交的.

**: 虽然  和  是两个不同的矩阵 (Stiefel 流形上的两个不同点), 但它们张成的是**同一个子空间**.

这意味着 Stiefel 流形包含了冗余信息: 子空间内部的基选择.

![图片](assets/9a8521c0ba13.png)

Stiefel 流形不仅记录了"子空间是什么", 还记录了"我们是如何在子空间内部建立坐标系的". 后者是我们不关心的冗余信息. 通过将 "张成相同子空间" 的正交基视为等价类, 即令  (), 商掉这一冗余, 就得到了 **Grassmann 流形**:

![图片](assets/5c2ab9de1143.png)

其维度为 , 恰好等于 Stiefel 流形的维度  减去  的维度 .

这一商空间结构天然地诱导出一个 **主 -纤维丛**:

![图片](assets/ef16707ebdbb.png)

此处  是商映射, 每个点  上方的纤维  同构于 , 对应子空间  的所有正交标架. 这一纤维丛结构不仅是抽象的拓扑概念, 它直接约束了 GSA 的算法设计: 所有可观测量 (路由分数、注意力权重、输出向量) 必须对  右作用不变, 即**规范不变** (gauge invariant). 具体而言, 投影矩阵  是规范不变的, 而正交基  本身不是.

![图片](assets/c4ccdf9a36d9.png)

#### 5.2.2 切空间的分解: 水平方向与纤维方向

在 Stiefel 流形上每一点  的切空间自然分解为两个正交分量:

**垂直空间** 由反对称矩阵  参数化, 维度为 . 沿垂直方向的运动不改变子空间本身, 仅改变子空间内部的基选择 -- 这正是规范冗余的无穷小版本.

**水平空间** 由满足正交性条件的矩阵构成, 维度为 , 与 Grassmann 流形的切空间  同构. 水平方向的运动代表子空间的真实变化: 旋转子空间使其指向不同的方向.

这一分解直接决定了黎曼梯度的计算方式. 给定欧氏空间中的梯度 , Grassmann 流形上的黎曼梯度是其到水平空间的投影:

![图片](assets/de9ece4d4b03.png)

投影算子  去除了沿纤维方向的分量, 保留了子空间变化的有效方向. 计算量仅为 , 且自动满足规范不变性.

#### 5.2.3 对称空间结构与曲率

Grassmann 流形不仅是一般的黎曼流形, 它具有更丰富的 **对称空间** 结构:

在 Cartan 分类中, 这是 AIII 型紧致对称空间. 其截面曲率的范围:,

![图片](assets/9668e13a1244.png)

直观地说, Grassmann 流形上的子空间距离不会出现 "锐利山谷", 这对梯度优化至关重要.

对称空间结构对 GSA 的意义可以从四个层面理解.

**齐性** 保证流形上每一点的局部几何完全相同, 这意味着子空间路由函数无需针对特定子空间位置做特殊处理.

**测地完备性** 确保任意两个子空间之间存在测地线, 子空间的插值和平均有良好定义.

**紧致性** 意味着流形直径有限 (), 所有子空间距离有界, 路由分数有自然的归一化.

**非负曲率** 结合紧致性保证了梯度下降不会陷入锐利的局部极小, 且 Ricci 曲率下界  使得函数值偏差服从指数尾的集中不等式.

#### 5.2.4 平行移动与 Riemannian Adam

对称空间的一个重要优势是, 沿测地线的平行移动有显式公式, 无需解微分方程. 从  到  的平行移动:

其中 ,  是投影矩阵. 这一公式的计算量仅为 , 仅需投影矩阵, 完全不涉及任何矩阵分解. 它在 Riemannian Adam 中用于将上一步的动量项搬运到当前切空间:

当精确平行移动不在测地线上或开销需要进一步控制时, 可用**投影型 vector transport** 替代:

直接将切向量投影到新点的水平空间, 复杂度同样为 .

#### 5.2.5 子空间的三种表示及其适用场景

同一个子空间可以用三种不同的数学对象表示, 每种都有其计算优势和适用范围.

**正交基矩阵** () 是最紧凑的表示, 参数量为 , 但不唯一 (模 ). 内积计算  仅需 , 梯度投影  同样为 , 因此在实际计算中效率最高.

**投影矩阵** 唯一表示子空间, 满足 , , . 映射  是 Grassmann 流形到对称矩阵空间的等距嵌入 (关于弦距离), 其像集构成一个代数簇 . 投影矩阵的核心优势是规范不变性自动成立, 适合定义路由分数等可观测量.

**Plücker 坐标** 通过外代数将 Grassmann 流形嵌入射影空间. 给定正交基 , Plücker 坐标定义为:

其中  是取  的第  行构成的  子矩阵. Plücker 坐标的维度为 , 在射影意义下唯一表示子空间. Plücker 嵌入携带了丰富的几何信息. Cauchy-Binet 公式给出:

由此导出 **Binet-Cauchy 核**: , 其中  是两个子空间之间的主角. 这个核函数同时编码了所有主角的信息, 比投影能量  (仅为主角余弦平方的加法聚合) 更加精细.

然而, Plücker 坐标的维度  在 GSA 的典型参数下急剧增长, 这一维度爆炸决定了完整 Plücker 坐标在  时不可直接使用, 必须采用截断策略

![图片](assets/fc521b89f90c.png)

三种表示在 GSA 不同阶段的选择策略如下: 实际的注意力计算 (Stage 1/3) 使用正交基形式以获得最高效率; 子空间路由 (Stage 2) 使用投影矩阵形式以保证规范不变性; 子空间指纹和哈希 (超长上下文加速) 使用截断 Plücker 坐标以获得紧凑标识和快速比较.

#### 5.2.6 度量族与距离选择

Grassmann 流形上定义了一族度量, 它们在拓扑上等价但在数值性质上各有特点. 两个子空间  之间的 **主角** (principal angles)  是所有度量的基本构建块, 其余弦平方  对应  的非平凡特征值.

测地距离  是内蕴度量, 理论上最自然, 但计算需要特征值分解, 与无分解约束冲突. 弦距离  是等距嵌入距离, 仅需  计算, 且完全可微. Binet-Cauchy 距离  基于所有主角的乘积, 计算量 , 适合外积路由和 DPP 核.

![图片](assets/9249ea27c200.png)

所有度量满足关键不等式:

弦距离对测地距离的近似比为 , 当  小 (GSA 中 ) 时, 近似质量高. 因此, 弦距离是 GSA 中子空间路由的默认推荐度量.

### 5.3 无分解约束下的流形计算

#### 5.3.1 为什么禁止 SVD 和 QR: 从理论到工程的考量

经典的 Grassmann 流形算法几乎无一例外地依赖矩阵分解. 指数映射  需要对切向量  做 thin SVD; 对数映射  需要对两个子空间间的过渡矩阵做 SVD; QR 分解是标准的正交化工具. 然而, 在深度学习的训练循环中, 这些分解面临三重困境:

**反向传播的数值不稳定性**. SVD 的梯度公式涉及  形式的项, 当存在接近或重复的奇异值时, 梯度趋于无穷. 在 BF16 混合精度训练中, 这一问题因有限精度进一步恶化, 导致训练中出现 NaN 或梯度爆炸.

**GPU 并行效率低下**. SVD 和 QR 的核心计算 (Householder 变换、Givens 旋转) 本质上是序列化的, 难以充分利用 GPU 的大规模并行架构. 对于 GSA 中  个子空间各自  矩阵的并行正交化需求, 矩阵乘法型算法可以通过 batched GEMM 实现近乎线性的加速, 而分解型算法的加速比有限.

**与自动微分框架的兼容性**. 虽然 PyTorch 提供了 `torch.linalg.svd` 的梯度支持, 但其实现在特定条件下 (如极小奇异值、矩形矩阵) 的正确性和效率仍有已知问题. 完全基于矩阵乘法的算法则天然兼容 `autograd`, 无需任何自定义 backward 函数.

因此, 本文采用以下替代方案:

![图片](assets/eebb12bc04a6.png)

#### 5.3.2 Retraction 的设计演进: 从指数映射到 Cayley 变换

黎曼流形上的优化需要一种操作将切空间中的更新方向 "拉回" 到流形上, 这就是 retraction. 理想的 retraction 是指数映射 , 它沿测地线前进, 但对 Grassmann 流形而言, 其计算依赖 SVD. 这迫使我们寻找替代方案, 设计路径经历了三个阶段.

![图片](assets/82b3d60b7d91.png)

**阶段一: 投影 Retraction (朴素方案)**. 最直接的想法是 "先迈步, 再投影": , 其中  将矩阵投影到最近的正交矩阵. 经典的投影方法是极分解, 但它本身需要 SVD. Newton-Schulz 迭代提供了纯矩阵乘法的替代:

每次迭代仅涉及两次矩阵乘法, 复杂度 , 3-5 次迭代即可收敛到机器精度. 然而, 这是一阶 retraction, 即 , 逼近精度有限.

**阶段二: Cayley Retraction (推荐方案)**. 构造反对称矩阵 , Cayley 变换给出:

看似涉及  矩阵逆, 但  是秩  矩阵, 利用 Woodbury 恒等式可将逆运算降至  矩阵: 复杂度 . Cayley retraction 自动保持正交性, 无需后处理, 且是 **二阶 retraction**: . 其矩阵逆的梯度通过  计算, 完全可微.

**阶段三: 投影矩阵参数化 (彻底绕开约束)**. 如果放弃正交基表示, 直接学习投影矩阵  并加软约束 , 则完全无需 retraction, 可使用标准反向传播. 代价是参数量从  增至 , 且约束满足是近似的.

三种方案的对比:

![图片](assets/26c4e50aa61d.png)

Retraction 与指数映射的偏差受曲率约束: , 其中  (一阶) 或  (二阶), . 当学习率  使得  时, 两种 retraction 的误差均可忽略. 对于  这一 GSA 的典型规格, Cayley 的额外精度带来的实际收益极小, 但其二阶性质为较大步长提供了更好的稳定性保证.

#### 5.3.3 Riemannian Adam: 从欧氏到流形的三处修改

将 Adam 优化器推广到 Grassmann 流形, 仅需在三处做修改:

其中  是黎曼梯度 (修改 1: 梯度投影), 动量  通过 vector transport  从上一步的切空间搬运到当前切空间 (修改 2: 动量搬运), 参数更新使用 retraction 代替加法 (修改 3: retraction). 二阶矩  是标量, 不需要 transport. 偏差校正 ,  与标准 Adam 完全一致.

```
Riemannian Adam 伪代码:输入: 初始子空间基集合, 学习率 η, 衰减 β1, β2For t = 1, 2, ...:  1. 欧氏梯度:    G_i = ∂L/∂U_i  2. 黎曼梯度:    g_i = I - U_i·U_i^T · G_i  3. 动量 transport: m_i ← β1·I - U_i·U_i^T·m_i + 1 - β1·g_i  4. 二阶矩:      v_i ← β2·v_i + 1 - β2·‖g_i‖²  5. 偏差校正:    m_hat = m/1 - β1^t,  v_hat = v/1 - β2^t  6. Retraction:   U_i ← Cayley_Retr U_i, -η·m_hat/sqrt v_hat + ε  7. 周期性重正交化: 每 T 步 Newton-Schulz 修复浮点漂移
```

#### 5.3.4 收敛性保证

Grassmann 流形的几何性质为 Riemannian SGD 提供了与欧氏 SGD 相当的收敛保证. 对于 -Lipschitz smooth 函数:

其中  依赖 retraction 的阶和曲率 . 收敛率  与欧氏情形相同, 最优学习率 .

紧致性提供了额外奖励: 黎曼梯度范数自动有界. 由 , 对 -smooth 函数:

对于 GSA (), 这给出梯度范数上界 , 为梯度裁剪提供了理论阈值, 且理论上不需要额外的梯度裁剪.

截面曲率  和 Rauch 比较定理给出学习率上界:

这与欧氏情形的  相差不大, 说明 Grassmann 的曲率不会显著限制学习率.

### 5.4 从几何到注意力: 子空间路由的设计演进

#### 5.4.1 核心思想: 子空间级别的信息选择

GSA 的核心转变可以用一句话概括: 将注意力的选择粒度从 token 提升到子空间. 具体而言, 将长度为  的 KV 序列按块 ( tokens/block) 划分为  个块, 每个块  提取一个  维主方向子空间 , 作为该块的 "语义摘要". Query 不再遍历所有  个 Key, 而是首先在  个子空间中选出  个最相关的 (路由), 然后仅在这  个块内的  个 token 上计算精确注意力.

整个架构由三个阶段构成:

![图片](assets/21bceed02caf.png)

#### 5.4.2 Stage 1: 子空间构建

子空间构建需要从每个块的  个 token 中提取  维主方向. 在禁止 SVD 的约束下, 三种方案各有取舍.

**方案 A: 幂迭代 (Power Iteration)** 是最直接的替代. 对每个块的协方差矩阵  做  轮幂迭代加 Newton-Schulz 正交化, 收敛到前  个特征方向. 复杂度 , 当 ,  时高效. 但协方差矩阵的显式计算引入了  的中间存储.

这一不足催生了 **方案 B: Oja's Rule 在线追踪**. 对块内 token  逐步更新:

然后周期性 Newton-Schulz 正交化. 复杂度  per token, 无需存储协方差矩阵. 理论上收敛到数据协方差矩阵的前  个特征子空间 (Oja, 1982). 缺点是收敛速度依赖于学习率调度, 且固定的统计子空间未必最优适应下游注意力任务.

进一步的改进导向了 **方案 C: 可学习投影 (推荐)**. 用轻量网络  直接预测子空间基:

其中 mean pool 将块压缩为单个向量, MLP 映射到  矩阵, Newton-Schulz 正交化保证正交性. 这一方案是端到端可微的, 子空间质量随任务目标优化而非固定于数据统计量, 且复杂度为  per block, 是三种方案中综合效率最高的.

![图片](assets/e947bbad6056.png)

#### 5.4.3 Stage 2: 子空间路由, 从投影能量到外积对齐

路由阶段的目标是为每个 query  从  个子空间中选出  个最相关的. 这等价于在 Grassmann 流形上做最近邻搜索, 不同的 "最近" 定义对应不同级别的路由策略.

![图片](assets/e4b9bdffc10f.png)

**Level 1: 投影能量路由 (默认)**. 计算 query 在每个子空间上的投影能量:

这是  与子空间  之间最小角度的余弦平方: . 从黎曼几何视角看, 这等价于 Grassmann 流形  上子空间  到射影空间  中  的投影距离的互补量:

可微分性是完备的: ,  (需投影到切空间). 复杂度  per query.

然而, 投影能量仅捕捉了 "总投影强度": 一个大主角即可贡献高分, 即使其余方向完全不对齐. 在需要精确检索的场景中 (例如检索特定事实), 这不够严格. 这一局限催生了外积路由.

**Level 2: 外积路由 (Plücker-enhanced)**. 将 query 通过可学习投影扩展为  维: , 然后计算外积路由分数:

 等于 -子空间与 -子空间之间的 "有向体积重叠", 即所有主角余弦的乘积. 当  时退化为标准内积注意力 ; 当  时, 它捕捉 **多方向同时对齐**: 只有所有主角都小时分数才高, 比投影能量对正交分量更敏感.

![图片](assets/1cff4279c236.png)

实践中使用 **混合路由分数**:

投影能量外积对齐

 可学习, 初始化为 0 (退化为纯投影能量), 训练中自动学习. 当需要 "全方向对齐" 时  增大, 否则保持在低值以节省计算.

**Level 3: Plücker-LSH 路由 (超长上下文, )**. 当块数极大时, 即便投影能量路由的  也可能成为瓶颈. 此时采用局部敏感哈希 (LSH) 加速.

核心思路是利用截断 Plücker 指纹作为子空间的紧凑标识. 预选  个随机行索引集 , 每个 , 对每个子空间计算指纹:

归一化后,  是  的无偏估计 (当  足够大). 当 ,  时, 指纹计算仅 32,768 FLOPs, 指纹维度 64, 极其紧凑.

在指纹空间上定义 LSH:

碰撞概率 , 与 Binet-Cauchy 距离单调相关. 使用  个独立哈希表, 每个  位, query 路由仅需查哈希表 + 对候选集精排.

![图片](assets/ee3d4d4f5a6c.png)

LSH 引入的 false negative (漏选) 通过与 local window 注意力互补来弥补, 保证近距离信息不被遗漏.

#### 5.4.4 多样性保证: 从 top-r 到 Plücker-DPP

选择 top- 个子空间时, 一个朴素的 argmax 策略可能导致选中的  个子空间高度冗余, 它们可能覆盖非常接近的语义方向. 行列式点过程 (DPP) 提供了同时考虑相关性和多样性的数学框架: , 行列式天然鼓励正交 (多样) 的选择.

构造 DPP 核矩阵:

其中  是 query 相关性分数 (品质因子),  是 Plücker 核.

渣注
简单的说, 这个算法是如何使用`行列式点过程` (Determinantal Point Process, DPP) 来解决一个经典问题: *如何在保证"质量"的同时, 提高所选元素的"多样性"*.

想象一下, 你要从一个视频网站(如 YouTube)的众多视频中, 为用户推荐  个视频.

**视频的"质量"**: 每个视频都有一个"品质分", 比如它的点击率、好评率等.

**视频的"相似度"**: 视频之间有相似性. 比如, 有很多视频都是关于"如何做番茄炒蛋"的, 它们内容高度重叠.

一个朴素的 argmax 策略可能导致选中的  个子空间高度冗余...

如果我们只按"品质分"从高到低选择 top- 个视频, 很可能会推荐给用户  个不同厨师做的"番茄炒蛋". 这虽然保证了每个视频本身质量都很高, 但用户体验会很差, 因为内容太单一, 缺乏多样性.
DPP 的目标
DPP 是一种概率模型, 它在选择一个元素子集  时, 不仅考虑每个元素的独立品质, 还考虑元素之间的相互排斥性 (或相似度). 行列式点过程 (DPP) 提供了同时考虑相关性和多样性的数学框架: 

这是 DPP 的核心定义.

: 一个半正定的`核矩阵 (Kernel Matrix)`, 描述了所有元素之间的关系.

: 一个你可能选择的元素子集.

: 从  中只保留索引在  中的行和列, 构成的子矩阵.

: 选中子集  的概率.

: `概率正比于这个子矩阵的行列式`.

利用行列式的几何含义是这些元素在某个特征空间中张成的平行多面体的体积平方. 多个元素互相正交 → 体积大 → 概率高; 多个元素重叠 (共线) → 体积坍塌为 0 → 概率为 0. DDP的实质是用 "体积 = 多样性" 的几何性质改造路由选择.

![图片](assets/9bc811ce0e3d.png)

假设每个元素是一个向量.

行列式  几何上对应于  中这些向量所张成的平行多面体的(有向)体积.

如果  中的向量彼此都很相似(几乎共线), 它们张成的"平行多面体"就会被"压扁", 体积接近于0. 因此, 选中这个子集的概率也接近于0.

如果  中的向量彼此都非常不同(几乎正交), 它们张成的"平行多面体"就会非常"饱满", 体积很大. 因此, 选中这个子集的概率就很高.

DPP 通过行列式这个工具, 内在地、优雅地实现了"鼓励多样性, 惩罚冗余"的目标.

举个例子吧, 为什么要DPP, 例如公司要从 100 个候选人里挑 5 人组队. 每人有"能力分" , 两两之间有"技能相似度" 

![图片](assets/88915685948c.png)

现在, 我们把这个通用框架应用到"选择 top- 个子空间"的问题上.构造 DPP 核矩阵:

这个  矩阵的每一个元素  (描述第  个子空间和第  个子空间的关系) 由两部分相乘构成:

**品质因子 (Quality Term): **

 是第  个子空间与当前 query 的相关性得分(可以理解为"品质分").

 将得分转换成一个正数.

. 矩阵的对角线元素代表了每个子空间自身的"品质平方". 品质越高的子空间, 对应的对角线元素越大, 它被选中的基础概率就越高.

**多样性/相似度因子 (Diversity/Similarity Term): **

 是一个用来衡量子空间  和 **相似度**的核函数. 当  时, . 当  和  非常相似时,  接近1. 当它们非常不同(正交)时,  接近0.

这里使用了 **Plücker 核** (即 Binet-Cauchy 核, ), 这是一个对子空间相似度非常敏感的、精细的度量.

 是归一化后的 Plücker 坐标,  是它们内积的平方.

**总结  的构造**: 矩阵的设计非常巧妙. 它的行列式  会同时被两件事影响:

子集  中每个元素的**独立品质** (体现在对角线元素  上).

子集  中元素之间的**两两相似度** (体现在非对角线元素  上).

当  中的子空间彼此非常相似时,  这个矩阵的行(或列)会变得线性相关, 从而导致其行列式趋近于0, 即使这些子空间本身的品质都很高.
DPP 的两种可微实现方案
直接从 DPP 模型中采样  是一个 NP-hard 问题, 而且通常需要对  进行特征分解, 这在深度学习中是不可微的. 因此有两种近似的、可微的替代方案.
方案 A: 贪心 DPP + Gumbel-Softmax (逐步构建多样性集合)
![图片](assets/4788fc07d3df.png)

这是一种模拟 DPP 采样过程的*贪心算法*.我们不一次性选出  个, 而是像组建篮球队一样, 一个一个地选.

**第一步**: 选择品质最高的那个子空间, 放入已选集合 .

**第二步**: 要选择下一个队员. 我们不仅要看候选者的个人能力, 更要看他能为团队带来多少*新的、不重叠的价值 (边际增益)*.

**边际增益 **: 在 DPP 的数学框架下, 当已选集合为  时, 加入一个新元素  所带来的"体积增量"可以通过*Schur Complement* 精确计算: . 这个公式的计算只涉及到对一个很小的  矩阵求逆, 计算成本很低 ().

**Gumbel-Softmax**: 纯粹的贪心选择(每次都选  最大的)是不可微的. Gumbel-Softmax 是一种技巧, 它可以将一个离散的选择问题(从候选者中选一个)松弛(relax)成一个可微的"软选择"(给每个候选者分配一个概率权重).

**流程**: 循环  次, 每一步都计算所有未选元素的边际增益, 然后用 Gumbel-Softmax 进行软选择, 将选中的元素(或其概率加权)加入集合 .

**优点**: 模拟了 DPP 的序贯采样过程, 保证了多样性, 且完全可微.

**缺点**: 是一个串行过程, 需要循环  次.
方案 B: 软 DPP 正则项 (事后惩罚冗余)
这是一种更直接、更简单的方法. 我们不改变原始的选择机制(比如还是用朴素的 top-), 但我们在**损失函数**中增加一项**惩罚**, 来惩罚那些缺乏多样性的选择.**多样性正则项 **:

![图片](assets/de7c89543c33.png)

: 一个  的矩阵, 它是你选中的那  个子空间的 *Gram 矩阵*.  度量了第  个选中子空间和第  个选中子空间之间的相似度 (这里用 Plücker 指纹的内积).

: 这个 Gram 矩阵的行列式, 同样, 它的几何意义是这  个子空间(在 Plücker 空间中)的特征向量所张成的*平行多面体的体积平方*.

*当选中的  个子空间两两正交时*:  变成单位矩阵, , . 惩罚为零, 这是我们最希望的情况.

*当选中的子空间有重叠(冗余)时*:  的行/列变得线性相关,  趋近于0,  趋近于负无穷, 于是  变成一个巨大的正数, 对模型进行严厉的惩罚.

**流程**: 正常选择 top- 个子空间. 在计算总损失时, 除了主要的任务损失外, 再加上这一项 . 优化器在最小化总损失时, 就会被迫学会选择那些能让  变小的、更多样化的子空间组合.

**优点**: 实现简单, 不改变前向传播的选择逻辑, 只是一个正则项. 计算成本  可忽略不计.

**缺点**: 是一种"事后惩罚", 可能不如方案 A 那样直接在选择过程中就保证多样性.

![图片](assets/19b29f0772d3.png)

#### 5.4.5 梯度稳定性: 几何视角下的风险与对策

将流形约束引入深度学习训练带来了独特的梯度稳定性挑战. Grassmann 流形的几何性质既提供了保护机制, 也引入了需要主动管理的风险.
![图片](assets/a37ae585b0cc.png)

紧致性是 Grassmann 流形提供的最重要的稳定性保障: 梯度范数自动有界于  (), 所有子空间距离有限, 优化景观中不存在趋于无穷的方向.

### 5.5 纤维丛驱动的注意力架构

#### 5.5.1 从描述性框架到可执行算法

前面建立了 GSA 的纤维丛数学结构, 但那主要是描述性的: 它告诉我们子空间、标架和路由在几何上意味着什么. 本节的目标是将这一结构提升为 **可执行的算法原语**, 将联络、曲率、截面等概念转化为具体的计算模块, 并证明这些模块带来了实质性的计算优势.

#### 5.5.2 完整的纤维丛建模

GSA 的纤维丛由三层结构组成.

**底空间** 是序列的块划分, 每个点对应一个 token 块. 赋予  混合拓扑:

第一项捕捉 **位置邻近** (局部注意力), 第二项捕捉 **语义邻近** (Grassmann 弦距离). 这为底空间赋予了位置与语义兼顾的结构.

**主纤维** 在每个块  上是 , 即子空间  的标架集合. 主丛  的结构群为 .

**关联向量丛** 的纤维  是块  内  个 token 的表示空间. 向量丛的截面  就是块  的 Value 向量集.

实际存储使用 **KV 坐标**:  和 , 分别是 Key 和 Value 在标架  下的局部坐标. 这些坐标在规范变换  下 co-transform: .

![图片](assets/d319a232a2a8.png)

#### 5.5.3 联络: 跨块信息传输的三种方案

当 query 需要同时访问块  和块  的信息时, 两个块使用不同的标架 . 如何在不同标架下一致地比较和聚合信息? 纤维丛理论的回答是: **联络** (connection) 提供了在不同纤维之间平行移动信息的规则.

在离散底空间上, 联络退化为每条边上的平行移动算子 . 三种方案逐步演进.

**方案 1: Grassmann 平行移动 (精确)**. 给定块  中的坐标 , 传输到块 : (1) 提升到环境空间 ; (2) 对称空间平行移动 ; (3) 投影到块  的标架 . 总效果由 **转移矩阵** 给出:

![图片](assets/6dfd67c74672.png)

这一公式的关键优势: 复杂度  计算 ,  计算 , **纯矩阵乘法, 无需任何分解**.

**方案 2: 投影联络 (最简洁)**. 直接将坐标从标架  投影到标架 :

不保距 (除非 ), 但当子空间接近时近似保距. 与方案 1 的关系: 当  小时, .

**方案 3: 可学习联络**. 以几何量  和  为输入, 学习偏离精确平行移动的修正项:

Newton-Schulz 保证输出正交, 实现保距传输. 可端到端训练, 学到任务特定的跨块信息变换.

![图片](assets/aa030de4195d.png)

#### 5.5.4 Holonomy 与自适应策略

沿闭合回路  平行移动后, 向量一般不回到原处:

偏离程度受几何约束: , 其中 .

![图片](assets/680bec673967.png)

Holonomy 的大小直接反映了跨块信息传输的可靠性. 当  小时, 不同路径传输同一信息给出一致结果, 可以信任远距离聚合; 当  大时, 传输不一致, 需要回退到局部注意力兜底. 这催生了自适应策略: 监控 , 当超过阈值时自动切换到精确块内注意力.

#### 5.5.5 纤维丛注意力: 等变性保证与维度优势

利用联络, 定义 **纤维丛注意力 (Fiber Bundle Attention, FBA)**. 给定 query , 选中块集合 , 在局部坐标下:

其中  是输出标架,  是从块  到输出标架的转移矩阵.

**等变性保证**. 对主丛上的规范变换  (): 坐标变换 , 转移矩阵变换 , 合成 . 注意力分数和输出对规范变换不变 -- 丛联络自动保证了等变性, 无需额外工程.

![图片](assets/6c2fa547246f.png)

**维度优势**是 FBA 最重要的计算收益. 注意力在  维坐标空间而非  维环境空间中计算:

![图片](assets/a3bba589ba34.png)

 是转移矩阵和最终提升  的一次性开销, 均摊到每个 token 后可忽略.

#### 5.5.6 双层注意力架构

纤维丛的底空间/纤维两层结构自然诱导出双层注意力:

![图片](assets/5a2c86f9fef6.png)

Stage A 的复杂度为 ; Stage B 的复杂度为  (注意是  而非 !); Stage C 的复杂度为 .

对 5M 配置 A, 块内注意力从  FLOPs/query (在  中) 降至  FLOPs/query (在  中), 实现额外  加速, 最终相对 Vanilla 达到  加速.

#### 5.5.7 规范场论视角: Yang-Mills 正则化

将 GSA 的子空间配置视为离散规范场, 可以借鉴物理学中的 Yang-Mills 理论来设计正则项.

定义 **离散规范场强**:

当联络 "平坦" (子空间兼容) 时 ;  衡量从  到  和反向传输信息的不对称程度. 场强可用于调制注意力权重:

高场强自动降低该块的注意力权重,  为可学习温度参数. 物理直觉是: "弯曲空间中远距离传输的信息可靠性降低."

**Yang-Mills 正则化** 定义联络能量:

![图片](assets/b1e377c582a4.png)

作为正则项加入总损失: . 最小化  使联络趋向平坦, 子空间之间的信息传输更一致. 这一正则项的额外优势在于诊断功能:  的值直接反映子空间配置的 "协调程度", 可作为训练质量的监控指标.

#### 5.5.8 层间子空间传递: 三种策略

Transformer 的不同层可视为同一底空间上不同纤维丛的级联. 层间子空间如何传递, 决定了参数效率和表达力的平衡.

**方案 A: 独立子空间 (默认)**. 每层独立计算 , 无跨层约束. 最灵活但参数量最大 ().

**方案 B: 联络传递 (Bundled Propagation)**. 第  层的子空间从第  层出发, 沿 Grassmann 流形平行移动:

 是学到的切向量增量. 层间子空间平滑变化, 参数量降至增量  per layer.

**方案 C: 共享主丛 + 层特异截面**. 所有层共享同一套子空间 , 每层仅学习不同的坐标变换 . 参数量 , 大幅减少. 适用于注意力模式在各层相对稳定的深层 Transformer.

![图片](assets/d8be163172fe.png)

#### 5.5.9 规范等变池化与全局截面

跨块聚合信息时, 必须使用规范等变的池化操作. **联络池化** 选定参考标架  (如路由分数最高的块), 将所有块的信息先传输到参考标架再平均:

朴素池化  是 **错误** 的, 因为不同标架下的坐标直接平均在几何上无意义 -- 这类似于将不同坐标系下的向量分量直接相加.

丛的 **非平凡性** 可通过 Holonomy 障碍量化: 回路. 当  时丛平凡, 可用全局标架, GSA 退化为简单的子空间投影; 当  时丛非平凡, 联络是必要的, 这也是纤维丛注意力相比简单投影更有效的理论依据. 在语义丰富的长文本中,  将显著非零.

#### 5.5.10 纤维丛算法的复杂度汇总

![图片](assets/35007eb07346.png)

核心节省: 纤维内注意力将 per-token 计算从  降至 , 联络开销  仅 per-block 一次性付出. 当  大时 (如 4096), 均摊到每个 token 的联络开销为 .

### 5.6 GSA算法

文章后面用到的相关符号如下:

![图片](assets/08f08c0e214a.png)

### 5.6.1 Overview

GrassmannAttention 将标准 self-attention 的  复杂度降至 的亚线性注意力机制, 通过以下三阶段实现.

![图片](assets/80338efed325.png)

整个算法从输入序列到注意力输出的完整链路描述如下:

```
Algorithm 1: GrassmannAttention (Top-Level)Input:  序列 X in R^{n x d}        Query/Key/Value 投影 W_Q, W_K, W_V in R^{d x d}        超参数: 块大小 b, 子空间维度 k, 选中块数 r                混合系数 lambda in [0,1] (可学习)                Gumbel 温度 tau (训练) / 退火至 0+ (推理)Output: 注意力输出 H in R^{n x d}================ Stage 1: 子空间构建 (一次性) ================1:  Q <- X W_Q,   K <- X W_K,   V <- X W_V        # all in R^{n x d}2:  m <- ceil(n / b)3:  for i = 1..m do in parallel:                  # batched GEMM4:      B_i  <- K[(i-1)b+1 : ib, :]               # in R^{b x d}5:      U_i  <- BuildSubspace(B_i)                # Algorithm 5C, in St(k,d)6:      C_i^K <- U_i^T · K[(i-1)b+1 : ib, :]^T    # in R^{k x b}7:      C_i^V <- U_i^T · V[(i-1)b+1 : ib, :]^T    # in R^{k x b}8:  end for9:  # 可选: 计算 Plucker 指纹用于 LSH/DPP10: for i = 1..m do:  f_i <- PluckerFingerprint(U_i)   # Algorithm 4================ 主循环: 对每个 query ================11: for q in Q (rows of Q) do in parallel:12:     # ---- Stage 2: 路由 ----13:     for i = 1..m do:14:         s_i^proj  <- ||U_i^T q||_2^2                          # Alg 615:         s_i^wedge <- det(Q_q^T U_i)^2,  Q_q = [q | W_1 q | ...] # Alg 716:         s_i       <- (1-lambda) s_i^proj + lambda s_i^wedge    # Alg 817:     end for18:19:     # ---- Plucker-DPP 选择 ----20:     S <- GreedyDPP_Gumbel({s_i}, {f_i}, r, tau)   # Algorithm 1021:22:     # ---- Stage 3: 纤维内注意力 ----23:     j*       <- argmax_{j in S} s_j           # 输出标架: 路由最强块24:     U_out    <- U_{j*}25:     q_tilde  <- U_out^T q                     # in R^k26:     for j in S do:27:         A_{out,j} <- U_out^T U_j              # in R^{k x k}28:         T_{out,j} <- A_{out,j} (4 A_{out,j}^T A_{out,j} - 3 I_k)   # Alg 1329:         k_tilde_j <- T_{out,j} · C_j^K        # in R^{k x b}30:         v_tilde_j <- T_{out,j} · C_j^V        # in R^{k x b}31:     end for32:     scores <- concat_{j in S}(q_tilde^T k_tilde_j) / sqrt(k)   # in R^{r b}33:     a      <- softmax(scores)                                  # in R^{r b}34:     h_local <- sum_{j in S} sum_t a_{j,t} v_tilde_j[:, t]      # in R^k35:     h      <- U_out · h_local                                  # in R^d36:     write h to row of H corresponding to q37: end for38: return H
```

三阶段的复杂度如下:

![图片](assets/efc5ea87c579.png)

当 , ,  固定时, per-token 复杂度退化为 .

#### 5.6.2 一些通用的算子

在 Grassmann 流形的计算中, 通常会涉及SVD / QR分解等算法, 这些算法对GPU的计算效率并不好. 因此本节列出贯穿全文的三个原语. 它们共同构成 GrassmannAttention 的`无分解`运算基础, 它们全部仅用矩阵乘法 + Woodbury 矩阵逆 + 行列式调用, 全部 GPU friendly 且自动可微.
Newton-Schulz 正交化 (代替 QR)

将近似正交矩阵  (例如  或 MLP 输出) 拉回到 Stiefel 流形 . 函数  是 sign function 的 Newton-Schulz 迭代格式, 二次收敛到 , 即  的极分解的正交因子.

![图片](assets/a1a4d29862d9.png)

```
Algorithm 2: Newton-Schulz OrthogonalizationInput:  X in R^{d x k} (近似正交), 迭代次数 t (默认 3-5)Output: Y in R^{d x k}, Y^T Y ≈ I_k (机器精度)1: # 谱缩放: 保证 ||X||_2 < sqrt(3), 否则不收敛2: alpha <- ||X||_F / sqrt(min(d, k))3: Y <- X / alpha4: for j = 1..t do:5:     G <- Y^T Y                 # k x k6:     Y <- 0.5 · Y · (3 I_k - G) # 两次 GEMM, 全部矩阵乘法7: end for8: return Y
```

**复杂度**: , 全部 batched GEMM 友好.

**梯度**: PyTorch autograd 天然支持, 无需自定义 backward.

Cayley Retraction with Woodbury (代替 SVD-based Exp)

**用途**: 将切向量  ( 上下文中可放宽) 拉回到流形上, 用于 Riemannian Adam 更新.

 是反对称的  矩阵但秩 , 因此可以分解为  形式. 通过 Woodbury 恒等式将  求逆降至  求逆, 复杂度从  降至 .

![图片](assets/8de5eb90a159.png)

标准 Woodbury 在中间需要求解  线性系统, 虽然  很小, 但计算对GPU也不友好, 我们改用 **Neumann 级数 + Horner 展开**: 设 , 则

**收敛性保证**: Riemannian Adam 步长 , 故  (实际更小, 因  中部分块有相消结构). 取  即得截断误差 , 远低于 Cayley 自身  的 retraction 阶, 不影响二阶逼近精度.

```
Algorithm 3: Cayley Retraction with Woodbury + NeumannInput:  Y in R^{d x k} (Y^T Y = I_k), 切向量 Delta in R^{d x k}        Neumann 截断阶 J (默认 5)Output: Y_+ in R^{d x k}, (Y_+)^T Y_+ = I_k 精确至 O(||Delta||^{2J})1: # 构造低秩因子 W = U V^T - V U^T  (反对称, 秩 <= 2k)2: U <- [Delta | Y]                           # d x 2k3: V <- [Y     | -Delta]                      # d x 2k4: # Woodbury 形式: (I - 1/2 U V^T)^{-1} = I + 1/2 U (I_{2k} - A)^{-1} V^T5: A <- 0.5 · V^T · U                          # 2k x 2k 小矩阵6: # 右端项: (I + 1/2 W) Y = Y + 1/2 U (V^T Y)7: rhs    <- Y + 0.5 · U · (V^T · Y)           # d x k8: y_sm   <- V^T · rhs                         # 2k x k9: # Neumann + Horner: w = (I - A)^{-1} y_sm10:        ≈ y_sm + A·(y_sm + A·(y_sm + ... + A·y_sm))11: w <- y_sm12: for j = 1..J-1 do                          # 共 J-1 = 4 次 GEMM13:    w <- y_sm + A · w14: end for15: Y_+ <- rhs + 0.5 · U · w                   # 应用 Woodbury 第二项16: return Y_+
```

**复杂度**: . 全部 batched GEMM

**逼近阶**: 二阶 retraction,  (Neumann 截断引入的额外误差为 , 可忽略).

**可微分**: 全 GEMM 计算图, autograd 直接反向传播, 无隐式微分二次 solve.

**数值**:  较大时 (如初始化未 warm-up), 可加 1-2 步 Newton-Schulz 矩阵求逆细化, 仍纯 GEMM.

Plucker 截断指纹

**用途**: 为子空间  计算紧凑标识, 用于 LSH (Level 3 路由) 和 DPP 核 (多样性选择).

**关键事实**: 完整 Plucker 坐标  维度爆炸 ( 时 ), 故采用随机行采样的截断版本. Cauchy-Binet 公式保证内积  是  的无偏估计.

![图片](assets/dcca32f7d8a6.png)

```
Algorithm 4: Plucker Truncated FingerprintInput:  U in R^{d x k}, 行索引集 {I_1,...,I_s}, 每个 |I_l| = k        (索引集预生成, 整个模型共享, 训练时固定)Output: f in R^s, ||f|| 归一化后是子空间的紧凑指纹1: for l = 1..s do in parallel:2:     M_l <- U[I_l, :]                # k x k 子矩阵 (取行)3:     f_l <- det(M_l)                 # k x k 行列式, 用 Bareiss 算法4: end for5: f <- (f_1, ..., f_s)6: return f / ||f||_2                  # 归一化
```

**复杂度**: , 当  时仅  FLOPs, 可忽略. **重要性质** :

即指纹内积平方  Binet-Cauchy 核, 同时编码所有主角.

复杂度分析汇总

![图片](assets/356512827b19.png)

#### 5.6.3 Stage 1: 子空间构建模块

子空间构建模块的形式接口为:

满足两条硬约束:

**正交约束**:  (机器精度);

**规范约束**: 输出  视为 Stiefel 流形上的点, 但所有下游可观测量 (路由分数、注意力输出) 仅依赖等价类 .

![图片](assets/50af9c26992a.png)

具体算法在前面章节已经介绍过了, 复杂度对比如下:

![图片](assets/a7848a383f15.png)

我们详细来看看方案 C: 可学习投影, 它用轻量 MLP 直接预测子空间基, 端到端可微, 子空间随任务目标优化.

其中  可以是 mean-pool 或 attention-pool,  输出  矩阵 (拍成  维向量再 reshape).

```
Algorithm 5C: Learnable Subspace Projection (Recommended)Input:  B_i in R^{b x d}, MLP 参数 phiOutput: U_i in R^{d x k}, U_i^T U_i = I_k1: # 池化: 单向量摘要2: p_i <- (1/b) sum_t B_i[t, :]              # in R^d, mean pool3: # 也可用更丰富的统计: p_i <- [mean(B_i); std(B_i); B_i[0]; B_i[-1]]4:5: # MLP 预测: 输出展平的 d x k 矩阵6: z_i <- MLP_phi(p_i)                       # in R^{d k}7: M_i <- reshape(z_i, (d, k))               # in R^{d x k}8:9: # Newton-Schulz 拉回 Stiefel 流形10: U_i <- NewtonSchulzOrtho(M_i, n_iter=5)  # Algorithm 211: return U_i
```

**复杂度**: 池化  + MLP  + NS , per block 总计 . 当  时主导项是 , 否则 .

**关键性质**:

**端到端可微**:  对  和  全程可导;

**任务最优**:  通过反向传播优化, 不绑定 PCA;

**批处理友好**: 所有  个块的 MLP 调用合并为一次大 batched GEMM.

子空间确定后, 立即将块内 Key/Value 投影到子空间坐标:

![图片](assets/ebb34e14f3e1.png)

存储时每块只需保留 , 总参数从  降至 . 当  时, 压缩比  .

```
def project_kv(B_K, B_V, U):    # B_K, B_V: (..., b, d), U: (..., d, k)    C_K = B_K @ U                # (..., b, k)    C_V = B_V @ U                # (..., b, k)    return C_K.transpose(-1,-2), C_V.transpose(-1,-2)   # (..., k, b)
```

#### 5.6.4 Stage 2: 三级路由策略
路由问题的几何描述
Stage 2 要解决的问题是: 给定 query , 从  个候选子空间  中选出  个最相关的.

"最相关"的几何含义是: 子空间  与  之间的**主角**最小. 不同级别的路由对应不同的"主角聚合方式":

![图片](assets/45cca119a3ef.png)

**Level 1**: 加性聚合  (投影能量) — 最快, 默认;

**Level 2**: 乘性聚合  (外积) — 强对齐检测, 用作辅助;

**Level 3**: Plucker-LSH 哈希 — 仅  时启用.

**规范不变性检查**: 三种路由分数都通过  或  表达, 在  () 下不变, 满足前文中的规范约束.
Level 1: 投影能量路由 (默认)

公式:

梯度:

后者需投影到 Stiefel 切空间: .

```
Algorithm 6: Level-1 Projection-Energy RoutingInput:  query q in R^d, 子空间集 {U_i in R^{d x k}}_{i=1}^mOutput: 路由分数 s^proj in R^m1: # 批处理: 一次大 GEMM 处理所有块2: U_stack <- stack({U_i}, dim=0)        # m x d x k3: proj    <- U_stack^T · q              # m x k       (per-block U^T q)4: s^proj  <- sum_l proj[:, l]^2         # m5: return s^proj
```

**复杂度**:  per query. 这是路由阶段的主导开销, 当  时为 , 满足亚线性.

Level 2: 外积路由 (Plucker-Enhanced)

**动机**: Level 1 仅用主角的"加性聚合", 一个大  即可贡献高分, 无法区分"部分对齐"和"完全对齐". 当任务需要严格的子空间包含检测 (如检索特定事实), 应使用乘性聚合.

**关键技巧**: 将 query 通过可学习投影扩展为  维矩阵.

可学习

外积路由分数:

仅当**所有主角**都小时分数才高, 实现"全方向同时对齐"的检测.

与前面 Neumann 思路一致, 对  做 **Modified Gram-Schmidt (MGS)** 正交化: , ,  上三角. 则:

对角元乘积

由于  均为 Stiefel 矩阵,  特征值 , 设  (特征值 ), 则:

截断误差

取 , 当  时截断误差 , 满足路由排序精度需求 (路由只需单调性). 全程无 LU、无 solve, 纯 GEMM + trace 操作.

```
Algorithm 7: Level-2 Outer-Product Routing (GEMM-only)Input:  query q in R^d, 子空间集 {U_i} (U_i^T U_i = I_k), 投影 W_1, ..., W_{k-1}        Neumann 截断阶 J (默认 8)Output: 路由分数 s^wedge in R^m1: # 构造原始 k 维 query 矩阵2: cols <- [q, W_1 q, ..., W_{k-1} q]               # k 个 d 维列向量3: # Modified Gram-Schmidt: Q = Q_hat · R (无 LU/QR 分解)4: Q_hat[:, 0] <- cols[0] / ||cols[0]||5: r_diag[0]   <- ||cols[0]||6: for j = 1..k-1 do7:    v        <- cols[j] - Q_hat[:, 0..j-1] · (Q_hat[:, 0..j-1]^T · cols[j])8:    r_diag[j]<- ||v||9:    Q_hat[:, j] <- v / ||v||10: end for11: log_det_Q <- sum log |r_diag|                    # log|det Q| = sum log|r_ii|, O(k)12: # 批量 M̂_i = Q_hat^T U_i,  一次 GEMM13: M_hat <- Q_hat^T · {U_i}                         # m x k x k14: # Gram 矩阵: G_i = M̂_i^T M̂_i (对称半正定, 特征值 in [0, 1])15: G   <- {M̂_i^T M̂_i}                              # m x k x k16: eps <- G - I_k + reg·I_k                         # reg=1e-4 保证 ||eps||<117: # Neumann log-det: tr(log(I + eps)) ≈ Σ_{j=1}^J (-1)^{j+1}/j · tr(eps^j)18: eps_pow <- eps19: log_det_G <- tr(eps)                             # j=1, coeff=+120: for j = 2..J do                                  # J-1=7 次批量 GEMM21:    eps_pow  <- eps_pow · eps                      # m x k x k22:    log_det_G <- log_det_G + (-1)^{j+1}/j · tr(eps_pow)23: end for24: return exp(log_det_G + 2 · log_det_Q)            # |det(Q^T U_i)|^2, 全 GEMM
```

**复杂度**: MGS  (per query, 一次) + 批 GEMM  + Gram  + Neumann . 总 , 与原方案同阶.

混合路由分数

混合 Level 1 和 Level 2:

投影能量归一化外积对齐归一化

其中  通过 sigmoid 参数化的可学习标量, 初始化为 0 (退化为纯 Level 1).  是各路分数的 batch-normalize 因子, 保证两路尺度可比.

```
Algorithm 8: Hybrid Routing ScoreInput:  query q, 子空间集 {U_i}, 可学习 lambda_logitOutput: 混合分数 s in R^m1: s_proj  <- ProjectionEnergyRouting(q, {U_i})         # Algorithm 62: s_wedge <- OuterProductRouting(q, {U_i})             # Algorithm 73: # 归一化到相近尺度4: s_proj  <- s_proj  / (mean(s_proj) + eps)5: s_wedge <- s_wedge / (mean(s_wedge) + eps)6: # 混合7: lambda <- sigmoid(lambda_logit)                       # 可学习8: s      <- (1 - lambda) · s_proj + lambda · s_wedge9: return s
```

**自适应行为**: 训练初期  (依赖快速 Level 1); 当任务需要严格对齐时, 反向传播将  推向较大值; 否则保持低值以节省外积路由的计算.

Level 3: Plucker-LSH 路由 (超长上下文)

当  时 (例如  长度 +  块大小), 即使 Level 1 的  也成为瓶颈. Plucker-LSH 用预计算的指纹 + 哈希索引将 query 路由复杂度降至 , 其中  是候选集大小.

![图片](assets/13e3e40018ee.png)

LSH 索引构建 (一次性, prefill)

```
Algorithm 9a: Plucker-LSH Index ConstructionInput:  子空间集 {U_i}_{i=1}^m, 哈希参数 (s, L, B)        s: 指纹维度 (默认 64)        L: 哈希表数 (默认 8)        B: 每表位数 (默认 16, 总 2^16 = 64K 桶)Output: 哈希表 hash_tables[1..L]: dict(bucket_id -> list of block indices)        指纹 {f_i}_{i=1}^m1: 预生成 s 组行索引 {I_1, ..., I_s} (与 Algorithm 4 一致)2: 预生成 L 组随机投影 {a_l^(j)}_{l=1..L, j=1..B}, a_l^(j) ~ N(0, I_s)3: for i = 1..m do in parallel:4:     f_i <- PluckerFingerprint(U_i)              # Algorithm 4, in R^s5: end for6: for l = 1..L do:7:     hash_tables[l] <- {}8:     for i = 1..m do:9:         bits <- [sign(<a_l^(j), f_i>) for j = 1..B]   # B-bit hash10:        bucket <- bits_to_int(bits)11:        hash_tables[l].setdefault(bucket, []).append(i)12:    end for13: end for14: return hash_tables, {f_i}
```

LSH Query (per query)

```
Algorithm 9b: Plucker-LSH QueryInput:  query q, 哈希表 hash_tables, 指纹 {f_i}        子空间投影 W_1..W_{k-1} (用于构造 Q-子空间)Output: 候选集 C ⊆ {1..m}, |C| ≈ c' (典型 100-500)1: # 把 query 转化为 k 维 Q-子空间, 用于求其指纹2: Q <- [q | W_1 q | ... | W_{k-1} q]3: Q_ortho <- NewtonSchulzOrtho(Q, n_iter=3)        # 投影到 St(k, d)4: f_q <- PluckerFingerprint(Q_ortho)               # in R^s5:6: # 在所有 L 张哈希表中收集候选块7: C <- empty set8: for l = 1..L do:9:     bits   <- [sign(<a_l^(j), f_q>) for j = 1..B]10:    bucket <- bits_to_int(bits)11:    C <- C ∪ hash_tables[l][bucket]12: end for13:14: # 精排: 仅对候选集计算精确路由分数15: for i in C do:16:    s_i <- (1-lambda) · ||U_i^T q||^2 + lambda · det(Q^T U_i)^217: end for18: return C, {s_i : i in C}
```

**碰撞概率** :

与 Binet-Cauchy 距离单调相关. **False negative 兜底**: 与 local window 注意力互补, 保证近距离信息不漏.

无论使用哪一级路由, 最终输出离散 top-r 选择. 训练阶段的可微分由下一节 (Plucker-DPP + Gumbel-Softmax) 统一处理.
Plucker-DPP 选择机制
**为什么需要 DPP: 从 top-r 到多样性**: 朴素 top-r argmax 选择的失败模式: 当 query 与某个语义簇高度相关时, 若该簇内多个块的子空间几乎重合, top-r 可能选出  个**冗余的**子空间, 浪费选择预算. 这等价于在 Grassmann 流形上做最近邻时未考虑覆盖度.

![图片](assets/ca8239efce8a.png)

**行列式点过程 (DPP)** 提供了同时考虑相关性 (quality) 和多样性 (diversity) 的概率框架:

其中  是  矩阵的  主子矩阵. 行列式天然鼓励**正交**(多样)的选择: 当两个子空间几乎平行时 ; 完全正交时  最大化.
Plucker-DPP 核构造
DPP 核:

其中:

**品质因子**, 由路由分数  经温度  缩放;

**Plucker 核** — Binet-Cauchy 核的指纹近似;

当  时 ; 当两者正交时 .

其实计算也很简单:

```
def build_dpp_kernel(scores, fingerprints, tau):    # scores: (m,), fingerprints: (m, s)    quality = torch.exp(scores / tau)                # (m,)    sim     = (fingerprints @ fingerprints.T) ** 2   # (m, m), Plucker similarity    L = quality.unsqueeze(0) * quality.unsqueeze(1) * sim    return L                                          # (m, m), 半正定
```

**核矩阵的半正定性**: ,  是 Gram 矩阵故半正定, 共轭后仍半正定. 这确保 DPP 良定义.

精确 DPP 采样需要对  做特征分解 (违反无分解约束), 因此采用以下两种近似方案.
方案 A: 贪心 DPP + Gumbel-Softmax (推荐)
**思想**: 逐步选择, 每步选边际增益最大的元素. 边际增益由 Schur 补给出 (DPP 的标准结果):

注意  仅是  矩阵,  极小 (4-16).

为保持训练时的可微性, 用 Gumbel-Softmax 替代每步的 argmax.

![图片](assets/dfd6ad49f138.png)

```
Algorithm 10: Greedy Plucker-DPP with Gumbel-SoftmaxInput:  路由分数 s in R^m, 指纹 {f_i}, 选中预算 r, 温度 tau, 训练标志 trainOutput: 软选择权重 alpha in R^{m x r} (训练) 或 硬选择 S = {i_1,...,i_r} (推理)        若训练: 同时返回 soft 选择掩码 mask in [0,1]^m1: # 构造 DPP 核2: q       <- exp(s / tau)                          # 品质因子3: K_sim   <- (F F^T)^2,  F = stack({f_i})          # m x m Plucker 相似4: L       <- q.unsqueeze(0) · K_sim · q.unsqueeze(1)5:6: # 贪心选择7: S       <- []8: alpha   <- zeros(m, r)9: marg_lo <- log(diag(L))                          # 初始边际增益 = log L_ii10: for step = 1..r do:11:    if train:12:        # Gumbel-Softmax 软选择13:        g     <- Gumbel(0, 1, size=m)14:        probs <- softmax((marg_lo + g) / tau_step)15:        # straight-through: 前向 argmax, 反向 softmax 梯度16:        i*    <- argmax(probs)17:        alpha[:, step] <- probs + (one_hot(i*) - probs).detach()18:    else:19:        i*    <- argmax(marg_lo)                # 硬 argmax20:    end if21:    S.append(i*)22:    # 更新边际增益: 用 Schur 补23:    L_SS    <- L[S, S]                          # |S| x |S|24:    L_SS_inv<- solve(L_SS, I_{|S|})25:    for i in {1..m} \ S do:26:        L_iS  <- L[i, S]27:        marg_lo[i] <- log(L[i,i] - L_iS · L_SS_inv · L_iS^T + eps)28:    end for29:    marg_lo[S] <- -inf                          # 已选不再选30: end for31: return (S, alpha) if train else S
```

**复杂度分析** (paper.md §4.4): 每步 Schur 补  求逆 +  边际更新, 总  (因为  是常数). 远小于 Stage 1/2.

**关键设计点**:

**Gumbel-Softmax + Straight-Through**: 前向用 argmax (硬选择, 推理一致), 反向用 softmax 梯度. PyTorch 写法: `alpha = soft + (hard - soft).detach()`.

**温度退火**:  从 1.0 退火到 0.1, 训练初期探索, 后期接近离散.

**数值稳定**: Schur 补的  必须  (保证半正定), 加  兜底.
方案 B: 软 DPP 正则项
**思想**: 不改变 top-r 选择机制 (仍用纯路由分数 argmax), 而是把多样性作为损失正则项, 反向传播塑造子空间分布.

其中  是选中子空间的 Plucker 指纹 Gram 矩阵 (). 当选中子空间两两正交时 , ; 出现重叠时  增大.

```
Algorithm 11: Soft DPP RegularizationInput:  选中索引 S = {s_1, ..., s_r}, 指纹 {f_i}Output: 多样性正则损失 L_div (标量)1: F_S <- stack({f_{s_a} : a = 1..r})          # r x s2: G_S <- F_S F_S^T                            # r x r3: L_div <- -log det(G_S + epsilon I_r)        # epsilon 兜底数值4: return L_div
```

加入总损失: , .
两方案对比
![图片](assets/960dab52245d.png)
离散 top-r 的 Gumbel-Softmax 松弛 (无 DPP 时)
若不启用 DPP (即纯 top-r), 仍需对离散选择做可微分松弛. 标准 Gumbel-Softmax 给出软概率, 但 top-r (而非 top-1) 需要扩展版本.

**Subset Gumbel-Top-r 算法** (Xie & Ermon, 2019):

```
Algorithm 12: Subset Gumbel Top-r RelaxationInput:  路由分数 s in R^m, 选择预算 r, 温度 tauOutput: 软掩码 alpha in [0,1]^m, sum(alpha) ≈ r (训练)        硬掩码 hard_mask in {0,1}^m, sum(hard_mask) = r (推理)1: # 加 Gumbel 噪声 (训练)2: g     <- Gumbel(0, 1, size=m)3: logit <- (s + g) / tau4:5: # 找 top-r 阈值6: vals_sorted <- sort(logit, descending=True)7: threshold   <- vals_sorted[r-1]                # 第 r 个最大值8:9: # 软掩码: sigmoid 形式, 值越大越接近 110: alpha <- sigmoid((logit - threshold) / tau_inner)11:12: # straight-through: 前向用硬, 反向用软13: hard  <- (logit >= threshold).float()14: alpha_st <- alpha + (hard - alpha).detach()15: return alpha_st
```

#### 5.6.5 Stage 3: 块内注意力计算 (纤维内注意力)

给定 query  和选中块集 , 在选中块的  个 token 上做 softmax 注意力, 得到输出 .

**朴素方案**: 提取选中块的原始 KV , 在  中做标准 attention. 复杂度  per query.

**纤维丛方案 (本节)**: 在  维子空间坐标中做 attention. 复杂度  per query, 当  大时主导项为 , 加速比 .
KV 坐标的回顾与规范变换
回忆 Stage 1 的产物:

不同块用不同标架 , 它们的坐标无法直接比较 — 这正是规范不变性问题. 必须先把所有选中块的坐标"传输"到统一的输出标架 .

**规范变换检查**: 若  (), 则 . 整个 Stage 3 的计算必须对此变换不变.

建议输出标架由路由阶段最高分块给出:

这一选择有两点理由: (1) 最相关的块的子空间最可能"覆盖" query 关心的方向; (2)  自动成立, 自身块无需平行移动开销.
转移矩阵构建 (Grassmann 平行移动的代数化)
无分解的转移矩阵公式:

![图片](assets/0fa80cb14e45.png)

```
Algorithm 13: Transition Matrix ConstructionInput:  输出标架 U_out in R^{d x k}, 选中块标架 {U_j}_{j in S}Output: 转移矩阵堆叠 T in R^{r x k x k}1: # 一次大 GEMM 计算所有 A_{out,j}2: U_S    <- stack({U_j : j in S}, dim=0)        # r x d x k3: A_stack<- U_out^T · U_S                        # r x k x k        (A_{out,j} = U_out^T U_j)4:5: # 计算 T = A (4 A^T A - 3 I)6: I_k    <- identity(k)7: AtA    <- A_stack^T · A_stack                  # batched: r x k x k8: M      <- 4 · AtA - 3 · I_k                    # r x k x k9: T      <- A_stack · M                          # r x k x k10: return T
```

**复杂度**:  计算 ;  计算 . 总 , 当  是常数 (如 4) 时为 , 一次性 per query.

**简化版本**: 当子空间配置平滑时 ( 小), 可直接用 , 进一步省去  那步, 复杂度仅 .
核心: 纤维内注意力计算
将所有计算搬到  维坐标空间中. 这是 Grassmann 注意力相对标准注意力的核心加速来源.

![图片](assets/b6b8ed418b27.png)

```
Algorithm 14: Fiber-Bundle Attention (Core)Input:  query q in R^d        输出标架 U_out, 选中块标架 {U_j}_{j in S}        块内 KV 坐标 {C_j^K, C_j^V}_{j in S}, 各 in R^{k x b}        soft 选择权重 alpha (训练时, in R^r) 或 1 (推理)Output: 注意力输出 h in R^d1: # ---- 投影 query 到输出标架 ----2: q_tilde <- U_out^T q                                 # in R^k3: # ---- 转移矩阵 (Algorithm 13) ----4: T_stack <- TransitionMatrices(U_out, {U_j})           # r x k x k5: # ---- 在 k 维空间传输每块的 KV 坐标 ----6: for j in S do in parallel:7:     k_tilde_j <- T_stack[j] · C_j^K                   # k x b8:     v_tilde_j <- T_stack[j] · C_j^V                   # k x b9: end for10: # ---- 拼接所有块的传输后坐标 ----11: K_tilde <- concat({k_tilde_j : j in S}, dim=1)       # k x (r b)12: V_tilde <- concat({v_tilde_j : j in S}, dim=1)       # k x (r b)13: # ---- 标准 softmax attention (在 R^k 中!) ----14: scores  <- q_tilde · K_tilde / sqrt(k)               # in R^{rb}15: # 将块级 alpha 广播到 token 级 (训练时使用)16: alpha_tok <- alpha.repeat_interleave(b)              # in R^{rb}17: scores  <- scores + log(alpha_tok + eps)             # 软选择 = log-mask18: a       <- softmax(scores)                            # in R^{rb}19: # ---- 加权聚合 + 提升到 R^d ----20: h_local <- V_tilde · a                                # in R^k21: h       <- U_out · h_local                            # in R^d22: return h
```
联络池化 (规范等变聚合)
当需要跨块平均 (而不是注意力加权) 时, 必须用联络池化, 而不是朴素均值.

**朴素均值的错误**:  直接相加不同标架下的坐标, 几何上无意义 (类似不同坐标系下的向量分量直接相加).

```
Algorithm 15: Connection-Aware PoolingInput:  各块聚合坐标 {c_j in R^k}_{j in S}, 参考标架 U_ref, 块标架 {U_j}Output: 池化输出 h_pool in R^d1: T_stack <- TransitionMatrices(U_ref, {U_j})           # r x k x k2: c_transported <- {T_stack[j] · c_j}_{j in S}          # 各 in R^k3: c_avg   <- (1 / r) sum_{j} c_transported              # in R^k4: h_pool  <- U_ref · c_avg                              # in R^d5: return h_pool
```

注: Algorithm 14 已经隐式包含了"传输再聚合"的逻辑 (步骤 7-8 + 19-21), 故纤维内注意力本身就是规范等变的, 无需额外池化层.
复杂度分析
![图片](assets/9fbcb5bcddeb.png)

**总复杂度**:  per query.

**与标准 attention 对比**: 标准 . 加速比  当 .

**5M Context时的算力对比** :

, , ;

标准 attention:  FLOPs;

纤维内注意力:  FLOPs;

实际加速比 , 相当于将块内 attention 这一项贡献的延迟也压低了一个数量级.
可选: Yang-Mills 调制
当子空间配置不一致时 (高规范场强), 可以用  调制注意力权重:

```
def yang_mills_modulate(scores, T_out_j, T_j_out, beta):    F = T_out_j - T_j_out.transpose(-1, -2)         # (r, k, k)    field_norm_sq = (F ** 2).sum(dim=(-1, -2))      # (r,)    return scores - beta * field_norm_sq.repeat_interleave(scores.shape[-1] // T_out_j.shape[0])
```

**复杂度**: 额外 , 完全可忽略. 物理直觉: "弯曲空间中远距离传输的信息可靠性降低, 自动降低其权重."

#### 5.6.6 完整的GSA算法

完整的GSA算法描述如下:

```
Algorithm 16: GrassmannAttention (End-to-End)Input:  序列 X in R^{n x d_model}, 注意力掩码 mask (causal/padding)        超参数: 块大小 b, 子空间维度 k, 选中块数 r                头数 H, 头维度 d = d_model / H                启用 LSH? L1+L2 混合? DPP 多样性? 训练?可学习参数:        W_Q, W_K, W_V    in R^{d_model x d_model}        SubspaceBuilder phi (per head 或 共享)        OuterProj W_1, ..., W_{k-1}  per head        混合系数 lambda_logit, 路由温度 tau_route        DPP 温度 tau_dppOutput: H in R^{n x d_model}============== Stage 0: QKV 投影 ==============1: Q <- X W_Q, K <- X W_K, V <- X W_V                    # (n, d_model)2: 分头: Q, K, V in R^{H x n x d}                        # H 头并行处理============== Stage 1: 子空间构建 (per head, 一次性) ==============3: m <- ceil(n / b)4: for h = 1..H do in parallel:5:     for i = 1..m do in parallel:                       # batched GEMM6:         B_i^h <- K^h[(i-1)b : ib, :]7:         U_i^h <- BuildSubspace_phi(B_i^h)              # Algorithm 5C8:         C_i^{K,h} <- (U_i^h)^T (K^h[(i-1)b:ib, :])^T   # k x b9:         C_i^{V,h} <- (U_i^h)^T (V^h[(i-1)b:ib, :])^T   # k x b10:        f_i^h    <- PluckerFingerprint(U_i^h)           # Algorithm 411:    end for12: end for============== 主循环: 每个 query (per head) ==============13: for h = 1..H do in parallel:14:    for q_idx in 1..n do in parallel:15:        q <- Q^h[q_idx, :]                             # in R^d           # --- Stage 2 路由 ---16:        if use_lsh and m > 10000:17:            C <- PluckerLSHQuery(q, hash_tables)        # Algorithm 9b, |C| ~ c'18:            候选下标 idx_set <- C19:        else:20:            idx_set <- {1..m}21:        end if22:        s_proj^h <- ProjectionEnergyRouting(q, {U_i^h}_{i in idx_set})  # Alg 623:        s_wedge^h <- OuterProductRouting(q, {U_i^h}_{i in idx_set})     # Alg 724:        lambda    <- sigmoid(lambda_logit)25:        s^h       <- (1-lambda) · normalize(s_proj^h) + lambda · normalize(s_wedge^h)           # --- Causal mask: 仅允许 query 看到自己之前的块 ---26:        i_q <- floor(q_idx / b)27:        s^h[i > i_q] <- -inf           # --- Plucker-DPP 选择 ---28:        if training:29:            (S, alpha) <- GreedyDPP_Gumbel(s^h, {f_i^h}, r, tau_dpp)    # Alg 1030:        else:31:            S <- GreedyDPP_Hard(s^h, {f_i^h}, r); alpha <- ones(r)32:        end if           # --- Stage 3 纤维内注意力 ---33:        j_star <- argmax_{j in S} s^h[j]34:        U_out  <- U_{j_star}^h35:        h^h_q  <- FiberAttention(q, U_out, {U_j^h}_{j in S},                                    {C_j^{K,h}, C_j^{V,h}}_{j in S}, alpha)  # Alg 1436:    end for37: end for============== Stage 4: 头合并 + 输出投影 ==============38: H_concat <- concat({h^h}_{h=1..H}, dim=-1)            # n x d_model39: H        <- H_concat W_O                              # 输出投影40: return H
```

#### 5.6.7 KVCache处理

推理时 token 逐个生成, 必须支持 O(1) per-token 的增量更新. 关键是块大小满了之后开启新块. 我们看看KVCache的状态如下:

```
class GrassmannKVCache:    """    分层混合存储 (paper.md §6.4):        - 热区: 最近 w 个 token 的原始 KV (w = b)        - 温区: 中距离块的 (U_i, C_i^K, C_i^V)        - 冷区: 远距离块仅保留 U_i 和指纹 f_i    """    def __init__(self, n_layers, n_heads, d, k, b, n_max_blocks):        self.b, self.k, self.d = b, k, d        # 子空间基: (L, H, m_max, d, k)        self.U     = []        # KV 坐标: (L, H, m_max, k, b)        self.C_K   = []        self.C_V   = []        # 指纹: (L, H, m_max, s)        self.fingerprints = []        # 热区原始 KV: (L, H, b, d)        self.hot_K = torch.zeros(n_layers, n_heads, b, d)        self.hot_V = torch.zeros_like(self.hot_K)        self.hot_count = 0           # 热区当前 token 数
```
每 token 增量更新 (Oja-style)
![图片](assets/86f6be8b1629.png)

```
Algorithm 17: Incremental KV Cache UpdateInput:  新 token 的 K_t, V_t in R^d (per head per layer)        当前热区 hot_K, hot_V (in R^{b x d}), hot_count        最近未固化块的 U_curr (per layer per head)1: # 写入热区2: hot_K[hot_count] <- K_t3: hot_V[hot_count] <- V_t4: hot_count <- hot_count + 15: # Oja's Rule 增量更新当前块的子空间 (Sec 3.3)6: U_curr <- U_curr + eta · K_t · (K_t^T U_curr)        # rank-1 update7: 如 hot_count mod P == 0:  U_curr <- NewtonSchulzOrtho(U_curr, n_iter=3)8: # 当热区满时, 固化为新块9: if hot_count == b:10:    U_curr <- NewtonSchulzOrtho(U_curr, n_iter=5)    # 最终正交化11:    C_K_new <- U_curr^T · hot_K^T                     # k x b12:    C_V_new <- U_curr^T · hot_V^T13:    f_new   <- PluckerFingerprint(U_curr)14:    cache.append(U_curr, C_K_new, C_V_new, f_new)15:    hot_count <- 016:    U_curr <- random_orthogonal(d, k)                 # 重置当前子空间17: end if
```
长上下文淘汰策略
当块数  达到上限  时, 触发淘汰.推荐 LFU + Plucker 多样性保护.

```
Algorithm 18: Plucker-Aware LFU EvictionInput:  当前块集 {U_i, C_i^K, C_i^V, f_i, freq_i}, 上限 COutput: 淘汰一个块的索引 i_evict1: # 候选: 访问频率最低的若干块2: candidates <- argmin_{|S|=10} freq_i3: # 计算每个候选块淘汰后的剩余 Gram 行列式4: F_all <- stack({f_j})                              # m x s5: for i in candidates do:6:     mask_no_i <- {1..m} \ {i}7:     G_i <- F_all[mask_no_i] · F_all[mask_no_i]^T  # (m-1) x (m-1)8:     # 用截断 Gram 行列式作为多样性度量9:     div_i <- log det(G_i + epsilon I)10: end for11: # 选择"淘汰后多样性最大保留"的块 (即 div_i 最大)12: i_evict <- argmax_{i in candidates} div_i13: return i_evict
```

**理由**: 行列式越大, 对应剩余子空间越正交、覆盖越均匀. 选择能保留最大行列式的淘汰对象, 等价于"宁愿淘汰冗余块".
分层混合存储管理
![图片](assets/287ae65f7ab4.png)

**冷热迁移规则**:

块固化 → 进入温区

长时间 (例如  tokens) 未被路由命中 → 进入冷区, 丢弃 

冷区块再被命中 → 因为  已丢, 输出仅用  投影后的均值代替 (有损)

![图片](assets/2fa34424f43a.png)

### 5.7  Grassmann注意力与范畴论的统一
5.7.1 核心范畴和流形结构的对应
![图片](assets/654ba3d2450d.png)
Token vs 子空间: 信息载体的粒度跃迁

**范畴论侧**: 语言范畴  中的对象是单个 token , 每个 token 是一个零维实体(点).

**Grassmann侧**: Grassmann流形  上的点是  维子空间 , 由正交基  表示.

**统一解释**: 两者构成信息载体的粒度层级. Token是原子级语义单元; 子空间是块级语义摘要, 将  个 token 的统计结构压缩为  维主方向. 从范畴论角度, 这等价于一个 **商函子**: 将精细范畴(token级)映射到粗糙范畴(块级), 每个等价类(块)对应一个子空间.

![图片](assets/283b1ebd7bc3.png)

形式化:

态射 vs 平行移动/转移矩阵: 信息传输的结构化编码

**范畴论侧**: PM-FuncAttn中的态射矩阵  编码了从 token  到 token  的语义变换关系. 它不仅表达"是否关注", 还表达"如何变换信息".

**Grassmann侧**: 转移矩阵  由对称空间平行移动导出:

它编码了从块  的纤维坐标到块  的纤维坐标的保距传输.

![图片](assets/d96a4581948e.png)

**统一解释**: 两者都实现了"信息传输不仅仅是标量加权, 而是结构化的线性变换". 关键区别在于:

: 无约束空间, 需要通过损失函数  软约束一致性

: 严格正交约束, 由几何结构硬编码保距性

Nerve构造 vs 纤维丛: 局部-全局关系的拓扑编码

**范畴论侧**: Nerve构造  将范畴编码为单纯复形:

0-单纯形 = 对象(token)

1-单纯形 = 态射(成对关系)

-单纯形 = -阶可复合态射链(高阶组合结构)

**Grassmann侧**: 纤维丛  提供了:

底空间 = 块划分(粗粒度位置)

纤维 = 子空间内标架(局部坐标自由度)

联络 = 跨块信息传输规则

**统一解释**: Nerve编码了对象间关系的**组合拓扑**, 纤维丛编码了空间上的**微分几何**. 两者的共同目标是描述**局部信息如何一致地拼装为全局结构**.

具体对应:

Nerve的 -单纯形存在性 ↔ 纤维丛截面的存在性(障碍理论)

Nerve的连通性() ↔ 纤维丛的Holonomy障碍 

Nerve的高阶洞() ↔ 纤维丛的非平凡特征类

函子性 vs 规范不变性: 结构一致性的公理化

**范畴论侧**: 函子  保持复合: . PM-FuncAttn的  软约束函子性.

**Grassmann侧**: 规范不变性要求所有可观测量对  右作用不变:

**统一解释**: 两者都是**结构保持**的数学表达:

函子性 = 计算过程与态射复合的顺序无关

规范不变性 = 计算结果与坐标系选择无关

两者通过以下等式统一:

复合保持函子性路径无关平坦联络规范场曲率为零

Kan扩张 vs 子空间路由: 信息外推的最优策略

**范畴论侧**: 左Kan扩张的逐点公式:

BS-KanAttn实现: 从锚点集  的已知值, 通过余极限权重  推断非锚点的值.

**Grassmann侧**: GSA路由:

从选中块集合  的纤维坐标, 通过联络平行移动聚合到输出标架.

**统一解释**: 余极限的本质是"从所有能到达目标的源头, 选择最一致的聚合方式". 在Grassmann流形上:

"能到达" = 路由分数  (弦距离邻域)

"一致的聚合" = 联络平行移动保证的等变性

余极限权重 = softmax(路由分数)

![图片](assets/ddd28fb603a6.png)

伴随函子 vs 投影-提升对偶

**范畴论侧**: 伴随  意味着 . MF-AdjAttn中:

正向注意力 : 解码器 query 到编码器 key/value 空间

逆向注意力 : 编码器 query 到解码器 key/value 空间

**Grassmann侧**:

投影:  (从  到  的纤维坐标)

提升:  (从  回到  的环境空间)

**统一解释**: 投影和提升满足:

这正是伴随关系的内积形式. 在GSA中:

子空间构建(投影) = 左伴随(自由函子, 信息压缩)

纤维丛提升(重建) = 右伴随(遗忘函子, 信息还原)

联络保证了投影-提升的信息守恒, 对应伴随一致性约束

5.7.2  Grassmann注意力范畴  的形式化定义
**定义 1** (Grassmann注意力范畴): 定义范畴  如下:

**对象**: , 即  中的  维子空间

**态射**: , 即由平行移动导出的转移矩阵

**复合**:  (矩阵乘法)

**恒等**: 

**定理 1**: GSA的转移矩阵系统  定义了一个函子:

其中  是底空间范畴(块索引 + 邻近关系).

**定义 2** (注意力2-范畴 ):

**0-胞(对象)**: 信息载体 — token(标准注意力) / 子空间(GSA)

**1-态射**: 标量注意力权重  — 编码"关注程度"

**2-态射**: 态射矩阵 (PM-FuncAttn) / 转移矩阵 (GSA) — 编码"如何变换"

**解释**: 标准注意力只利用了1-态射层面的信息. PM-FuncAttn和GSA通过引入2-态射, 将注意力从"标量路由"提升为"结构化信息变换". 2-范畴结构自然地编码了:

水平复合:  (传输的级联)

垂直复合: 跨层子空间传递

**定理 2**: DPP选择等价于在Nerve复形中寻找最大体积的k-单纯形.

**证明**:

DPP核矩阵的构造为:

其中 .

DPP采样概率: .  的几何含义:

对角项 : 编码子空间  与 query 的相关性(品质)

非对角项 : 编码子空间对之间的体积重叠

: 选中子空间在Plücker空间中张成的平行多面体的体积

在Nerve构造中:

选中的  个子空间  构成一个 -单纯形

单纯形的"体积" 

DPP最大化  ↔ 在Nerve复形中选择**最大体积的截面**

具体来说:

其中  是纯多样性核.  精确地度量了选中子空间在Plücker嵌入空间中的"正交体积", 即它们张成的高阶单纯形的体积. 

**推论**: DPP的贪心算法(Schur complement边际增益)对应于在Nerve复形中逐步添加能最大化体积增量的顶点 — 这是单纯形体积的贪心最大化.

#### 5.7.3 TC-SimAttn ↔ 纤维内注意力 (FBA) 的等价性

两者都实现了"在低维空间中计算高阶交互":

**TC-SimAttn**: 将 -单纯形交互分解为  维张量收缩级联, 

**FBA**: 在  维纤维坐标空间中计算注意力, 

当 TC-SimAttn 的窗口  对齐块大小 , 且  时, TC-SimAttn的1-单纯形计算在计算图上等价于FBA.

![图片](assets/d58e2de4d39c.png)

#### 5.7.4 BS-KanAttn ↔ GSA三级路由的等价性

两者的本质都是"锚点识别 + 从锚点推断", 但采用不同的数学语言:

![图片](assets/2ab6fcfa9ea9.png)

*GSA路由是BS-KanAttn的正则化版本* : GSA的三级路由是BS-KanAttn在以下正则化条件下的特殊情况:

锚点集  固定为块中心(消除动态选择的不规则性)

可达性由Grassmann度量定义(连续可微, 替代硬阈值)

余极限权重由DPP正则化(多样性保证)

#### 5.7.5 PM-FuncAttn ↔ 联络 + Yang-Mills正则化

*PM-FuncAttn = 纤维丛联络在GL中的松弛*: PM-FuncAttn的态射矩阵系统  是GSA转移矩阵系统  在以下松弛条件下的推广:

将  松弛为  (去除正交约束)

将  允许  (允许不同维度)

将几何约束松弛为软损失 

![图片](assets/af4f43c09037.png)

![图片](assets/8d6216fde833.png)

**关键insight**: 正交约束将优化从"高维非凸空间中的无约束搜索"转化为"紧致流形上的几何优化". 代价是表达力受限(只能表达旋转/反射, 无法表达缩放), 收益是梯度稳定性和优化景观的改善.

#### 5.7.6 ST-HomAttn ↔ Holonomy自适应

![图片](assets/044e0984ec1e.png)

### 5.7.7 CatGrassFormer

CatFormer和GSA各自都有三阶段结构, 其对应关系揭示了统一的可能:

![图片](assets/69a32ac2dcd7.png)

**关键insight**: CatFormer沿**模型深度**展开三阶段(跨层), GSA在**单层内**展开三阶段(层内). 统一架构应将两者**嵌套**: 外层是CatFormer的深度沙漏, 内层每一层使用GSA的三阶段注意力.

![图片](assets/b0301c68d5e6.png)