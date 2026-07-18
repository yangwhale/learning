# 科学智能-AI4S-2: 变分法和有限元方法

> 作者: zartbot  
> 日期: 2024年4月17日 14:51  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489445&idx=1&sn=f2c08311757691474e55b2dcd7dd69f2&chksm=f9960767cee18e71271b34e4021f6a298d74a08fbe7d74761fe874eba285c28b40744a76493b#rd

---

FDM有限差分方法解偏微分方程，解的结果就是方程的在网格节点上的近似解，是一种点近似的解法。而工程上更希望是一种函数近似的方法，也就是今天我们会谈到的有限元方法(Finite Elements Methods,FEM)，有一个介绍有限元的简短的视频

      
     
       
         
           
             
                                

                 
                   
已关注
                   **                 
             
             
               关注
           
           
                            **               重播                                         **               分享                                                      **               赞                                     
         
                   
         
                   
         
       
     
     

关闭**

**观看更多**

更多**

**

**

**

*退出全屏*

[**]()

**

   
         
     
       [         视频详情       ]()     
   
 

熟肉可以参考《10分钟理解有限元法》[1]

有限元法基于变分原理，所以我们先从变分法谈起。最后还搬运了一个基于FeniCS的实战视频。

## 1. 变分法

人们通过微分方程描述自然界的一些物理现象，并且直接由偏微分方程差分方法来离散化求解。但是在很多情况下，描述一个物理现象或者过程，有不同的形式，例如从物理守恒定律出发导出变分原理。虽然这两种方法某种意义上是等价的，但是计算方法并不是等效的，由变分原理出发能更真实的反应物理现实。有限元法就是基于变分原理的一种离散计算方法。

### 1.1 从最速降线谈起

在很多实际应用中，需要考虑过程优化的问题，从而归结出以下数学问题

为函数空间，也被称为容许集，为函数的函数，称为泛函，通常是一个积分,我们想要通过选择被积的函数来最大化或者最小化的值。 例如最速降线问题，设曲线为,且满足

![图片](assets/83aa74424bd9.png)

对于所有的这样的曲线的集合,我们要寻找到一个

设物体质量为,则有

即,s表示当前位置到起始点A的长度，而t表示相应的滑行时间，故有

从A到B的滑行时间为

问题就被转化为

其中

### 1.2 变分问题解的必要条件

`变分学基本引理` 设,如果

则

对于任意,定义

显然, 因此

问题转化为研究关于未知变量的目标函数的优化问题，即

直接计算：

令,有

即：

由变分学基本引理可得Euler-Lagrange方程

这是满足变分问题的解的必要条件

### 1.3   二次函数的极值

我们以二次函数的极值问题来介绍变分原理的基本概念和方法。在维欧式空间中记向量和矩阵的记号如下：

定义的内积为

`定理1` 假设矩阵对称正定，则下列两个问题等价：

求使得

求下列方程组的解

定理1表明，如果 是极值问题的解，那么它也是方程组的解。是定义在上的二次函数，称为上的“二次泛函”，许多数学物理问题的直接数学形式就是求解“二次泛函”的极小值, 并且对解做了某些“光滑性”假设后，才归结到微分方程。

另一方面对于微分方程数值解，我们可以通过变分法把它转化为一个“二次泛函”的极小值问题，因为这样的问题建立数值解法往往更方便。

`Example`我们以一根长为l的弦，其固定在A(0,0)和B(l,0)两点，设有强度为的外部负荷,表示在负荷下弦的平衡位置，根据力平衡条件，为弦上张力，满足

边界条件为

考虑力学中的“极小位能原理"

弦平衡位置为，但是我们对在哪一个函数类里取极小值，也就是说要给出属于哪一类函数空间，为了使积分有意义，又必须对做必要限制。

### 1.4 Sobolev空间概述

对于数学函数的光滑性有很多种。最基本的要求可能就是函数要连续，更进一步的要求是可微，再强一些的概念是导数的连续性(),在研究微分方程的过程中， 人们发现函数空间不是研究微分方程的解的恰当的空间。

而Sobolev空间正是空间的替代品，用于研究偏微分方程的解。本节对Sobolev空间做一些简单的介绍，简单的来说，Sobolev空间定义的函数连同它直到m阶导数都属于可积函数

设,称是的对的广义偏导数,,

更一般的，对于多重指标,用记号

`广义导数定义`： 称是的阶广义导数，如果,

`Sobolev空间`: 设,令

规定范数：

特别的来说，时,我们定义1阶Sobolev空间如下：

设, 表示定义在I内的平方可积函数组成的线性空间，线性空间,其中是的广义导数, 内积和范数如下：

这样的空间构成1阶Sobolev空间, 同样可以定义m阶Sobolev空间, , ,导数均为广义导数，内积和范数如下：

下面我们来讲述一个重要的Sobolev空间嵌入定理，这对PDE理论研究非常重要。

设区域是中的开集，边界充分光滑，从到的嵌入算子是有界算子，即存在常数M，使得

并且嵌入算子是完全连续的。另一方面，若,N是区域所属空间维数，则嵌入,且存在常数M，使得

且嵌入算子是完全连续的

特别的说，嵌入,即指若一个函数只要它具有直到k阶的广义导数,那么我们一定可以修改这个函数在零测集上的函数值，使它成为上的连续函数。

最后，若有非负整数满足,则嵌入,且存在常数M，使得

嵌入算子是完全连续算子

### 1.5 变分法求解微分方程

我们将换为一般的线性算子,换成函数,换成函数,换成Sobolev空间，则问题转化为对方程,其对应的泛函定义为

我们需要证明线性算子是对称正定的，然后就可以转换为变分问题了。

当时，Poisson方程的Dirichlet问题，

与变分问题

其中

以及其变分形式

三者互相等价

记, 则(1.5.3)可以写成如下形式

变分形式(1.5.5)为

在(1.5.1)定解问题中要求其解，而变分问题(1.5.2~1.5.4)或者变分形式(1.5.5)中的解只需要具有一阶广义导数。变分问题在力学上相应于“最小势能原理”，而变分形式在力学上相应于“虚功原理”。

对于二维Poisson方程的其它边值问题的变分原理类似。

### 1.6 变分问题近似解

#### 1.6.1 Ritz方法

求解变分问题

其中区域,为Hilbert空间，且假定,泛函

而

Ritz方法的基本思想是：不把泛函的极值放在容许函数空间中考虑，因为是无穷维函数空间，而是在内找一个有限维子空间,其维数为,选取上一组基函数,则可以表示为这组基函数所生成的空间，记为：

对任意有

其中是任意实常数，在上求解变分问题的近似解，即求,使得

记矩阵和列向量分别为：

则

这样就把泛函的极小值问题，转化为求以为自变量的二次函数的极小值问题。

矩阵是正定的，则极小值一定存在，若在时，达到极小值，则

上述代数方程组有唯一解，则

#### 1.6.2 Galerkin方法

求解,为Hilbert空间，使得

或写成更一般的形式

选取中的一个有限维子空间

是N个线性无关的基函数，在中寻求变分形式的近似解,即使得

取并设变分形式的近似解

为确定系数，把v_N,u_N$带入

同理可得方程组

## 2. 有限元方法

有限元方法是经典变分方法(Ritz-Galerkin方法)与分片多项式插值结合的产物。由于选择了特殊的基函数，使它能适应一般的区域。有限元方法基于变分原理又具有差分方法的一些特点，并且适合较复杂的区域和大小不同的网格，因此有着非常广泛的应用。

本节以二维Poisson方程Dirichlet边值问题为例

用有限元方法求相应的变分问题

### 2.1 区域剖分

对于一个二维区域，通常是进行三角形剖分(Triangulation),如下图所示：即分成有限个互相不重叠的三角形元素(Elements),剖分时尽量不要出现大钝角三角形，这样会影响计算精度，同时不允许把一个三角形的顶点作为另一个三角形的内点。在预计中未知函数变化剧烈的部分，可以把网格取得更密集一些，当边界为曲线时，用折线代替。

![图片](assets/c91bdf30bd9e.png)

剖分后的每一个三角形称为一个`单元`，所有单元的顶点称为`节点`，在内部的节点称为`内节点`,在边界上的节点称为`边界节点`.所有单元的全体记为,边界为,是一条封闭折线，区域剖分后，对单元和节点编号，设共计个单元，记为,有个节点，记为坐标为,每个单元的是那个定点按照逆时针顺序编号

![图片](assets/a04247a2f190.png)

### 2.2 线性插值函基函数

我们可以构造一个函数空间,对于,满足
a) 在上连续，且b) 在每个单元上,是的一次多项式

具有这样性质的函数是分片线性函数，并且可以验证是一个线性空间，三角形的顶点依次为 顶点顺序排列保证了三角形面积可以表示为

在单元上构造线性插值函数，即在单元上令

则可以构造单元上的线性插值基函数

其中

其几何图像构成三角锥，如图所示，称为形函数

![图片](assets/1512df586384.png)

在每一个节点处，用它为公共顶点的单元上的形函数确定一个高度为1，而其它单元取值为0的“角锥”函数（也称为屋顶函数）,如下图所示

![图片](assets/997b678e7102.png)

这样的函数取为函数空间的一个基函数，每个基函数对应一个节点，如果点是边界节点，则“角锥”在边界上的某些平面将是铅直的，对语边界条件，只需要在每一个内节点建立相应的基函数即可，谁内节点总数为N，有

且线性无关，

即有

其中为在点处的值，我们称为试探函数空间，它是的一个有限维子空间。

### 2.3 有限元方程的形成

在建立试探函数空间后，我们在有限维子空间上求解变分问题近似解,使得

记

则有限元方程为,在力学上把A称为刚度矩阵，f称为载荷矩阵。在具体计算时，先采用单元分析，在每个单元上算出,得到单元刚度矩阵，然后在对应叠加构成总刚度矩阵，同演的每个单元计算出单元载荷向量在合成。

### 2.4 FeniCS实战

搬运了一个来自machine-learning-and-simulation[2]的视频

      
     
       
         
           
             
                                

                 
                   
已关注
                   **                 
             
             
               关注
           
           
                            **               重播                                         **               分享                                                      **               赞                                     
         
                   
         
                   
         
       
     
     

关闭**

**观看更多**

更多**

**

**

**

*退出全屏*

[**]()

**

   
         
     
       [         视频详情       ]()     
   
 

这是一个FeniCS库基于利用有限元方法，求解一个悬臂梁一端被夹紧的情况受自身重力载荷发生的挠曲

![图片](assets/367bc46dd84e.png)

```
"""Cauchy Momentum Equation:           − ∇⋅σ = fConstitutive Stress-Strain:         σ = λ tr(ε) I₃ + 2 μ εDisplacement-Strain:                ε = 1/2 (∇u + (∇u)ᵀ)σ  : Cauchy Stress (3x3 matrix)f  : Forcing right hand side (3D vector)λ  : Lambda Lame parameter (scalar)μ  : Mu Lame parameter (scalar)ε  : Engineering Strain (3x3 matrix)I₃ : 3x3 Identity tensor (=matrix)u  : Displacement (3D vector)∇⋅ : The divergence operator (here contracts matrix to vector)tr : The trace operator (sum of elements on main diagonal)∇  : The gradient operator (here expands vector to matrix)ᵀ  : The transpose operatore -------Scenario:A cantilever beam is clamped at one end               .+------------------------+             .' |                      .'|            +---+--------------------+'  |      ↓ gravity   clamped  |   |                    |   |            |  ,+--------------------+---+            |.'                      | .'            +------------------------+'It is subject to the load due to its own weight and willdeflect accordingly. Under an assumpation of smalldeformation the material follows linear elasticity.------Solution strategy.:Define by "v" a test function from the vector function spaceon u.Weak Form:    <σ(u), ∇v> = <f, v> + <T, v>with T being the traction vector to prescribe Neumann BC (here =0)Alternative Weak Form (more commonly used):    <σ(u), ε(v)> = <f, v> + <T, v>(valid because σ(u) will always be symmetric and the inner productof a symmetric matrix with a non-symmetric matrix vanishes)------Once the displacement vector field u is obtained, we can compute thevon Mises stress (a scalar stress measure) by1. Evaluating the deviatoric stress tensor    s = σ − 1/3 tr(σ) I₃2. Computing the von Mises stress    σ_M = √(3/2 s : s)"""import fenics as feCANTILEVER_LENGTH = 1.0CANTILEVER_WIDTH = 0.2N_POINTS_LENGTH = 10N_POINTS_WIDTH = 3LAME_MU = 1.0LAME_LAMBDA = 1.25DENSITY = 1.0ACCELERATION_DUE_TO_GRAVITY = 0.016def main():    # Mesh and Vector Function Space    mesh = fe.BoxMesh(        fe.Point(0.0, 0.0, 0.0),        fe.Point(CANTILEVER_LENGTH, CANTILEVER_WIDTH, CANTILEVER_WIDTH),        N_POINTS_LENGTH,        N_POINTS_WIDTH,        N_POINTS_WIDTH,    )    lagrange_vector_space_first_order = fe.VectorFunctionSpace(        mesh,        "Lagrange",        1,    )        # Boundary Conditions    def clamped_boundary(x, on_boundary):        return on_boundary and x[0] < fe.DOLFIN_EPS    dirichlet_clamped_boundary = fe.DirichletBC(        lagrange_vector_space_first_order,        fe.Constant((0.0, 0.0, 0.0)),        clamped_boundary,    )    # Define strain and stress    def epsilon(u):        engineering_strain = 0.5 * (fe.nabla_grad(u) + fe.nabla_grad(u).T)        return engineering_strain        def sigma(u):        cauchy_stress = (            LAME_LAMBDA * fe.tr(epsilon(u)) * fe.Identity(3)            +            2 * LAME_MU * epsilon(u)        )        return cauchy_stress        # Define weak form    u_trial = fe.TrialFunction(lagrange_vector_space_first_order)    v_test = fe.TestFunction(lagrange_vector_space_first_order)    forcing = fe.Constant((0.0, 0.0, - DENSITY * ACCELERATION_DUE_TO_GRAVITY))    traction = fe.Constant((0.0, 0.0, 0.0))    weak_form_lhs = fe.inner(sigma(u_trial), epsilon(v_test)) * fe.dx  # Crucial to use inner and not dot    weak_form_rhs = (        fe.dot(forcing, v_test) * fe.dx        +        fe.dot(traction, v_test) * fe.ds    )    # Compute solution    u_solution = fe.Function(lagrange_vector_space_first_order)    fe.solve(        weak_form_lhs == weak_form_rhs,        u_solution,        dirichlet_clamped_boundary,    )    # Compute the von Mises stress    deviatoric_stress_tensor = (        sigma(u_solution)        -        1/3 * fe.tr(sigma(u_solution)) * fe.Identity(3)    )    von_Mises_stress = fe.sqrt(3/2 * fe.inner(deviatoric_stress_tensor, deviatoric_stress_tensor))    lagrange_scalar_space_first_order = fe.FunctionSpace(        mesh,        "Lagrange",        1,    )    von_Mises_stress = fe.project(von_Mises_stress, lagrange_scalar_space_first_order)    # Write out fields for visualization with Paraview    u_solution.rename("Displacement Vector", "")    von_Mises_stress.rename("von Mises stress", "")    beam_deflection_file = fe.XDMFFile("beam_deflection.xdmf")    beam_deflection_file.parameters["flush_output"] = True    beam_deflection_file.parameters["functions_share_mesh"] = True    beam_deflection_file.write(u_solution, 0.0)    beam_deflection_file.write(von_Mises_stress, 0.0)if __name__ == "__main__":    main()
```

参考资料

[1] 
10分钟理解有限元法: https://www.bilibili.com/video/BV1tq4y1j7f1
[2] 
machine-learning-and-simulation: https://github.com/Ceyron/machine-learning-and-simulation/