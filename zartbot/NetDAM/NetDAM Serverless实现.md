# NetDAM Serverless实现

> 作者: zartbot  
> 日期: 2021年10月17日 16:03  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486752&idx=2&sn=bb38a695843ee52c21f74f0d64ba03ca&chksm=f9961de2cee194f46317983606d1177d38e758c569d43abf6f00f0ef83869aa317b8de464371#rd

---

当NetDAM支持加解密卸载、可靠传输、共享内存池后，接下来做Serverless就水到渠成了，照旧先给不懂的人普及一下无服务，然后讲NetDAM可以做什么

### Serverless简介

Serverless（无服务器)最早由AWS在2014年提供的Lambda服务实现，无服务器是一种云原生开发模型，可使开发人员专注构建和运行应用，而无需管理服务器。无服务器方案中仍然有服务器，但它们已从应用开发中抽离了出来。云提供商负责置备、维护和扩展服务器基础架构等例行工作。开发人员可以简单地将代码打包到容器中进行部署。部署之后，无服务器应用即可响应需求，并根据需要自动扩容。公共云提供商的无服务器产品通常通过一种事件驱动执行模型来按需计量。因此，当无服务器功能闲置时，不会产生费用。

Serverless有很多案例：具体可以参考AWS的介绍[1]

### NetDAM Serverless

本质上NetDAM有了板载内存和多块NetDAM共享内存池后，我们就可以通过Serverless控制来根据用户请求分配临时地址空间，然后由客户直接写入NetDAM，在写入完成后，调用一条NetDAM解密指令，接下来将解密后的CryptoOP放入CompleteQueue，由主CPU批量Poll后决定执行。

![图片](assets/6748fd2a4bc6.png)

整个过程中，从计算域来看，CPU、GPU、FPGA都是在访问NetDAM内存，而抛开了所有的TCP、Kernel、加解密的消耗，几乎做到了完全的零损耗，实现了inline加解密和TCP协议栈**的完全卸载。这些都得益于NetDAM可编程指令集的架构优势，甚至我们还可以构建真正的大规模资源池，通过对共享的mempool读写执行计算完成更多更大规模的操作，最终片上网络和数据中心网络通过CXL. NetDAM和Ruta实现了大融合，而几个月前提过的这个图，Data-Centric Computing ISA就是NetDAM的完整实现。至于存储和分布式数据库，其实也就那么回事了

![图片](assets/366b4fd612a9.png)

#### Reference

[1]
AWS Lambda: https://aws.amazon.com/cn/lambda/?c=ser&sec=srv