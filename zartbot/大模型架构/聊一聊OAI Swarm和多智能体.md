# 聊一聊OAI Swarm和多智能体

> 作者: zartbot  
> 日期: 2024年10月13日 08:10  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492483&idx=1&sn=4b9ad405329cc57fc7acef37587bb05e&chksm=f995f341cee27a5782daa2dd147ed0d54a4fd8fdf31b90fabeada4dd4c8e068ab44888a766ca#rd

---

最近OAI开源了一个叫SWARM[1]的多智能体项目, 挺有意思的. 作者还有一篇博客介绍《Orchestrating Agents: Routines and Handoffs》[2]

对于多智能体(Multi-Agent)的协同编排是一个非常有趣的话题, 然后Routines和Handoffs又有点操作系统进程间切换的感觉, 似乎构成了一个新的LLMOS的雏形了? 有了输入输出,多任务编排调度, 进程切换等? Tokens as File? POSIX interface是什么呢?

![图片](assets/632e29740bdf.png)

Multi-Agent并不是什么新东西, 这不还有Swarm Corportation的创始人Kye Gomez发了一大堆推文控诉OAI剽窃.

![图片](assets/269e3eb2fd1d.png)

其实吧, 多智能体的框架叫Swarm的二十年前就有了, 渣B 20年前就在用Swarm多智能体做股市仿真交易的研究.

![图片](assets/19be59d7b975.jpg)

当时桌上的这本书就是

![图片](assets/50f5da3b6675.png)

通过多智能体的博弈来获得一个GroundTruth蛮有趣的, 例如当时渣B把投资者分为:专业投资者, 投机者, 跟风韭菜等多个角色, 每个投资者有一个可以动态根据行情更新的状态转移矩阵来决定自己买卖股票的概率, 然后构建了百万级的Agents,并通过一个符合幂律分布的随机图来进行消息传递, 最后对市场上的ETF300的股票进行仿真交易来预测市场在极端宏观因素影响下的走势.

而现在, 智能体的算法已经由单个状态转移矩阵变成了LLM, 这会是一件非常有趣的事情, 例如仿真十亿个智能体, 然后灌入通过大模型产生一些关于市场的评论让它们根据外在信息在仿真市场内撮合交易,特别有趣.

话说最近收到国外一个高频交易机构的邀请, 感叹现在勾引人不说自己有个很大的GPU Farm都不好意思开口....

### 谈谈GPU架构和智算中心

话说恰逢NV GeForce 256发布25周年, 然后回过头去看看 《GPU架构演化史3: 1994-2000 群魔乱舞》

那个年代, 图像处理的流水线相对固定后

![图片](assets/4ec855d63ae5.png)

3Dfx的成功在于廉价的EDO内存和纹理处理的Offload来降低CPU的内存带宽需求, 而GeForce256的成功则是将定点处理和几何变换处理的算力Offload到GPU

![图片](assets/302e78695e28.png)

逐渐蚕食CPU的路径非常明显, 那么未来呢? GPU本身在Compute和Memory Bound的时候该怎么做呢? GPGPU的故事讲了也快20年了,而NV随着光追开启在图形处理上又开始DSA化, 在计算上也逐渐的越来越难编程, 未来呢? 来个民科暴论:“GPU也会逐渐被一种新的体系结构的东西蚕食掉, 但很慢大概也需要8~10年” 而在这些快速的变革中, 投资回报率如何计算? 听说有的厂商已经开始卖卡了不高Pre-Train了. 而看到SemiAnalysis的一个报告, H100租金降低到2美元/时, 低的1.5都有. 随着B200的逐渐上市, 这些大量的H100的小型“智算中心”如何变现? 特别是国内疯狂建设的一些智算中心?

参考资料

[1] 
swarm: https://github.com/openai/swarm
[2] 
Orchestrating Agents: Routines and Handoffs: https://cookbook.openai.com/examples/orchestrating_agents