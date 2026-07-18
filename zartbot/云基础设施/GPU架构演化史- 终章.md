# GPU架构演化史: 终章

> 作者: zartbot  
> 日期: 2022年9月8日 16:01  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488403&idx=1&sn=bcbfc7613fa6c236967c7ddae93249a7&chksm=f9960351cee18a47ce634a561b9de16f25789e9b2887668d402825fe6c98303de8b1aa863164#rd

---

这个GPU的系列,大概花了20天的时间整理,起因是今年Intel重返独立显卡的第一年,再加上国内一大堆GPU厂商流片回来开始发布. 似乎想起1997年前后群魔乱舞的年代,加上当年的AGP和如今的CXL,几分感慨: "1997年过去了,我很怀念它", 另一方面是云渲染和云超算业务的需求。

[**GPU架构演化史1：3D渲染算法概述**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487916&idx=1&sn=51c249f3b17c1930c08f75b08aa7e1d0&chksm=f996016ecee18878f99b7740f435c51a8cd45f321b1e0b6f14a7335c555b3be298f89550b910&scene=21#wechat_redirect)

[**GPU架构演化史2: 1980-1993 SGI时代**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487954&idx=2&sn=038983b8328c6e6c56fe25188b16b640&chksm=f9960110cee188065d7c7c7cba3ae149e20f8d54f7c8f2a38a7363e22ade1b4f3b69be9e64a2&scene=21#wechat_redirect)

[**GPU架构演化史3: 1994-2000 群魔乱舞**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487954&idx=1&sn=a7a8a61b1d1fd179dc525d251bdfdbd1&chksm=f9960110cee18806664f3728109483eee36e2d2f12b2b8c3f0f2624fdf211a7eafb2ef7701eb&scene=21#wechat_redirect)

[**GPU架构演化史4: 2000-2006 AN争霸可编程**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487976&idx=1&sn=d258826f5829b2225b93183248f3f893&chksm=f996012acee1883cd8a41eb8d57d6b6fee57357d5ffd184883a89a5b4524c796aec27e7840dc&scene=21#wechat_redirect)

[**GPU架构演化史5: 2006-2010 统一着色器带来初代CUDA**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488000&idx=1&sn=95ff38f9e2d1a6f4df3e2adc223bd8af&chksm=f99602c2cee18bd4cb7781cc1e3af0f1fc1aad3ad455920cd7c993515e7d2baa304bb71355b7&scene=21#wechat_redirect)

[**GPU架构演化史6: Telsa CUDA架构详解**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488023&idx=1&sn=2a3d6f808b9d52396edb230ca8ce6e09&chksm=f99602d5cee18bc357812b1f682bb3f45c404da328688d825071de81ed8b69d7c92e1e48e592&scene=21#wechat_redirect)

[**GPU架构演化史7: Fermi架构详解**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488081&idx=1&sn=efeda52bd9d5233eda36c9c2ec75fdbf&chksm=f9960293cee18b85e264d0fe112218cd57e956d820d856e6ae835dcf1b21263edccf3d2c1db9&scene=21#wechat_redirect)

[**GPU架构演化史8: Kepler架构详解**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488134&idx=1&sn=5991ce5c7378ba6cba5c13b0c8eb226c&chksm=f9960244cee18b52c00f564a11bdfb92f2341409882501336b7461207e53d4cc9d66d7f56b18&scene=21#wechat_redirect)

[**GPU架构演化史9: Maxwell架构详解**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488157&idx=1&sn=3ad1fb6266e7ecaaf7385bc742f43d1d&chksm=f996025fcee18b49396182202469fbad0186c827710ef8e5a9eb240f879f46572172b840ee4b&scene=21#wechat_redirect)

[**GPU架构演化史10: Pascal架构详解**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488195&idx=1&sn=ad3cc222fac42fefc4dc9362f6f3b93b&chksm=f9960201cee18b17e20a0fc7b505b97998527ac6cad65998597c5c7cfb4d353c9faaca4da10c&scene=21#wechat_redirect)

[**GPU架构演化史11: Volta架构详解**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488243&idx=1&sn=25b7eacd13e0daf339462ba5fc6ffdba&chksm=f9960231cee18b27ec3c7cfe384214396e56af4fc79c35996760c3e9f30944236a43c8011caa&scene=21#wechat_redirect)

[**GPU架构演化史12: Turing架构详解**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488280&idx=1&sn=55bf2338621782b6996e69267aecdd2e&chksm=f99603dacee18acc50b6ae07753bf62fe43253d87b66439e3ff5462e77383ee29493fb7aa101&scene=21#wechat_redirect)

[**GPU架构演化史13: Ampere架构详解**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488322&idx=1&sn=b4dda78f7e82f4f17c86338df545a736&chksm=f9960380cee18a967807559a4f67a77a82763ee78446635811f97b92986b93f87694834c2bd7&scene=21#wechat_redirect)

[**GPU架构演化史14: Hopper架构详解**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488380&idx=1&sn=bf83d9150f629adbd46016c0a1ba7062&chksm=f99603becee18aa841afc71cabbfc2e5a003aad98ccc89f5dd2188bec30ff83f109e7a53dc22&scene=21#wechat_redirect)

今年六月有个投资机构叫渣去跟某GPU公司交流一下，发现知识欠缺，很怂不敢去。同时前段时间去参加一个Live VideoStack的展会,因为Ruta的项目认识了国内一些做RTC的同学,看到前同事代表字节做的一个8K VR的demo.联想到最近李健、刘德华这些在抖音或者微信视频号上的演唱会，再加上VR眼镜的价格已经降到2000以下，感觉有戏了...

字节的某个同学还给我了一个Pico，玩了玩感觉也逐渐成熟了，而另一个字节同学在玩VR游戏串流的时候还需要在家里专门配一个PC，所以心里有一个梦想，一方面如何构建一个能够支持数十万人的VR演唱会，在云端能够协同渲染，而另一方面如何能够更低成本的在边缘计算节点完成VR游戏的云渲染，和视频直播不同的是，由于每个人的FOV的不同，渲染如何构建，如何协同，渲染管线如何定义.

对于云渲染的好奇心成了这一系列文章的开端. 另外作为一个做DPU的小工程师,号称数据中心第三个Socket, 却不了解其它两个Socket的能力,有点感觉自己不太敬业, 同时过去两年研究Ruta和NetDAM对于AI模型的发展少了很多关注, 也想借此机会补齐, 于是就有了这一个系列的读书笔记,参考资料有几本书和接近1000个PDF文件,比较杂乱,因为是笔记而不是论文,所以引用资料也没有很详细的标出.只是本着像费曼学习法那样,讲懂读者也就让自己搞懂的初衷,写下这些笔记.当然里面肯定还有很多错误，如果误导了大家还请海涵.

因为工作重心的不同以及精力有限,其实还有很多课需要继续补的,只是很粗浅的学了一些OpenGL的Shader和一些算法,而例如Games101~Games204,图形学的虎书龙书,RT Rendering,PBRT都得脚踏实地的去读, 而在参考资料的时候,读到一些浅墨的文字,以及一本他审校的CUDA的书,再看看那本Realtime Rendering 4th edition,有些悲凉和惋惜，但愿他翻译的遗作能够早日面世吧....

而这一路走过来，从80年代初期的图形卡到现在的GPU，在HPC、图形学、AI三条线中穿插游走，渐渐的读懂了Nvidia这样的公司伟大之处，当然这个行业的竞争非常激烈，无论是图形学或者HPC或者AI，新的技术出现都是碾压性的效果，图形学从光栅化到光追，从第一代Direct3D到现在的Direct12，渲染流水线发生了天翻地覆的变换。HPC这些对性能的极致追求，使得它更容易靠近底层去优化，基本上HPC的网络架构、处理器架构、异构架构每一代都有重大的变革。

而正是这些背景促进了整个行业的发展，老黄也有那种无法割舍的危机感， 也正是这些危机感使得每一代Nvidia都可以去颠覆自己以往的设计，特别细赏这样的勇气。当然在这整个历史长河里， 伟大的公司不止Nvidia，还有那个一直跟它贴身肉搏的ATI，以及消失在历史长河里的SGI，昙花一现的3Dfx.还有二十多年来卧薪尝胆几次尝试的Intel.

恰逢读书期间，A100、H100以及MI200都被禁了，因此更有必要让自己对这个领域有更多的了解，暂时手边还没有太多的测试环境，以后有机会可能会在GEMM上再做一些研究，以及多读一些Tensor Category的书，从张量范畴的角度来看待编译器优化和模型设计的问题也是一个非常好的方向.

在这个领域，想到曾经的天河和神威，再看如今的榜单，蛮多感慨的..榜单的前三名也没有NV的身影，而AMD MI200也是一个非常值得去学习的产品，同样Fugaku的Tofu-D互联和Slingshot-11都是值得去关注的

![图片](assets/17d1ddf3dab7.png)

当然从架构上来看，和这一系列文章的主线来看，可能我们很多架构师还是没有去仔细思考GPGPU在Gaming(Graphic)和Compute(HPC)和AI这三个场景的差异，因此设计出来的架构虽然某些指标高，但总觉得缺乏一个顶级架构师那样内功深厚的平衡，或者遇到各个领域问题的妥协.其实AMD也把这图点破了

![图片](assets/76db9a60abec.png)

但是回想起国内很多做AI加速芯片的厂商过去几年犯过的错，多了几分感慨。同样AMD MI200上的计算单元对于矩阵运算的规模，到16x16x4就好了，只是他们比nVidia更倾向于去做HPC的生意，FP64的支持更好，而少了一些AI训练的FP8. 所以这次被禁不要简单的认为是限制国内AI发展，更多的是在高端制造和科学研究领域的FP64算力

![图片](assets/28c10595e73b.png)

当然对于Infinity vs NVLink这样的话题，毕竟我自己也没用过，更没评价的权利了，只是觉得这样的架构比GraceHopper也有更美妙的地方

![图片](assets/4b70e2e6686c.png)

体系架构而已，不知道是不是以前很多年做VLIW的架构带来的影响， AMD的图例还是画着vGPR和vALU，然后整个CU一个Program Counter，我不确定他们会不会有nVdia那样的协作组的概念,可能后面有精力要去仔细看看AMD的ISA Guide了

![图片](assets/f88e07a3f990.png)

但感觉是这一块做的有些烦的...

![图片](assets/e3b965258416.png)

当然AMD的很多chiplet design还是很好玩的， 我一直期待一个Zen4+CDNA2的处理器，再加上收购的Pensando的DPU或者再封一块Xilinx的FPGA，似乎架构上可以干出一些惊天动地的事情来，但是总觉得从片上网络来看有些说不出来的问题

而另一方面在软件生态结构上也不只是有CUDA， 值得我们关注的是Intel的Ponte Vecchio，同时支持SIMT和SIMD，再加上OneAPI似乎有点意思

![图片](assets/b7f5960dc05d.png)

从大的体系架构来看，三家其实基本上都大同小异了。Vector Core、Matrix Core

![图片](assets/086160ed7666.png)

或者是这次发布的BIren

![图片](assets/4bf8475b203f.png)

然后层次化的内存

![图片](assets/120559c0a95f.png)

昇腾、nvidia都有的TMA，在Biren上叫TDA

![图片](assets/bc9673c14612.png)

当然CUDA的SIMT乃至coop-group也在biren上有

![图片](assets/407ea9572188.png)

在芯片的封装上，还有一些值得挖掘的地方

![图片](assets/5bc5155d14d2.png)

![图片](assets/bca79d9e5038.png)

![图片](assets/df8162b95d5a.png)

当然在超大规模互联上，个人喜好来看更喜欢DataFlow Architecture，一方面是自己在做分布式AI架构的时候写过一个DataFlow engine，从软件上来看挺有趣的，另一方面就是对NetDAM的研究和Tenstorrent的一些scale-out的架构，片上网络和片间直接Ethernet做一些事情很有趣，关于Tenstorrent可以看一下这篇文章：

[**再谈云计算范式的变革**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487651&idx=1&sn=0bac7f43f530d3b599a1a53912f5b485&chksm=f9960061cee189777e72ca5e499ef87d7990e0f866299d30ec648c45ee9fcfc769f522c3d595&scene=21#wechat_redirect)

[**Tenstorrent&NetDAM**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487564&idx=1&sn=3c1144e0c45111ae6018c0166315c12c&chksm=f996008ecee18998c130d08bc45df12370d75d37e677ad62f358ac5e05e34b7367adb4de0980&scene=21#wechat_redirect)

![图片](assets/b30a9db35e01.jpg)

![图片](assets/63ec52a94596.jpg)

BUDA相对于CUDA Kernel拆分成kernel和pipe这两部分，个人来看似乎还是比cooperative group那样更干净的方式。

当然像Google的Pathway来看，局部的处理器阵列加上DataFlow似乎真是一条正路，中间的Branching、Routing的功能更多，与其这样密集的SIMT，还不如干脆丢出去，所以在一系列新的MoE模型和Transformer一类的基础模型上，CUDA或者Nvidia并不一定是一条正确的路，它还受到很多图形业务的羁绊。正好被禁了，也没有什么生态兼容的说法了。

谈到Tenstorrent了，另一个值得关注的问题就是CU的指令集干脆用RISCV统一了好了，也有一些RISC-V的GPU开源项目了，例如Vortex好像国内清华最近也有一些消息，没有具体的去关注。

当然最后还有一个值得关注的问题就是MLIR，似乎这也是会带来整个Infra变革的东西，但里面还有很多工作要去做

![图片](assets/5e128bfc9386.png)

不过，反正已经开学了，有小朋友的神兽已经归笼，大家也可以开始学习了，毕竟学习强国嘛：）