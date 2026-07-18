# RDMA这十年的反思3：AWS HPC为什么不用Infiniband

> 作者: zartbot  
> 日期: 2024年4月12日 03:07  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489300&idx=1&sn=3ad44db6269dcf74c885d33f04885019&chksm=f99607d6cee18ec0738aafcad22e8cccc3abfcf20bef0f8f64f9979c52b1dcb575cc2785ffd4#rd

---

最近看到一个视频介绍《How EFA works and why we don't use infiniband in the cloud.》[1]，
视频搬运如下：

      
     
       
         
           
             
                                

                 
                   
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
   
 

顺便打个广告，阿里云的第八代云服务器通过eRDMA技术可以全面支持HPC等高性能应用，并且和标准的RDMA Verbs生态完全兼容，用户无需任何特殊配制就可以把ANSYS FLUENT, Star-CCM+, OpenFOAM等多种HPC应用在云上部署。

视频介绍了AWS在HPC场景不使用Infiniband以及开发改造EFA的原因。访谈对象是AWS的Senior Principal Engineer, Brian Barrett，也是EFA的开发者之一。在加入AWS之前Brain在Los Alamos和Sandia两个美国的国家实验室工作过，开发了大量的MPI相关的组件

![图片](assets/f3f1acb1f17b.png)

HPC的低延迟需求来自于很多应用都会通过网格刨分来进行并行运算，然后网格间有复杂而频繁的通信数据交互，Brain将其称为“Ghost Cell Exchange”。
![图片](assets/32c3e5dd57c4.png)

因此很多HPC系统将单个报文的延迟(Single packet latency)放在第一位，这也是Infiniband/RoCEv1/RoCEv2非常在意报文大小和HPE Cray构建HPC Ethernet的原因。

在AWS EFA的实践来看，单个报文的延迟并不是问题，而更重要的是网络中的拥塞冲突带来的长尾延迟。通过SRD来解决了几个问题：

多路径降低拥塞冲突概率

多路径解决链路失效等问题

MPI的很多操作不需要Reliable Connection的通信语义严格保序

解决QP数量多的爆炸问题

关于不兼容RC语义的原因：从Brain的履历也能大概看出来，由于Brain大量的OpenMPI的开发经历，所以在构建SRD时选择了不和标准的RC语义兼容，这也给后续的生态带来了一些问题。

### 1. 不使用Infiniband的原因

访谈中Brain介绍了一些原因: "云数据中心很多时候是要满足资源调度和共享等一系列弹性部署的需求，专用的Infiniband网络构建的集群如同在汪洋大海中的孤岛" 并且国外HPC需求较国内高的原因在访谈中也介绍了：国外并没有太多的线下机房，通常一些HPC任务需要在一些超算集群排队数周，如果有一个性能差不多的云上环境，对客户而言很有吸引力。

### 2. 应用性能

从应用性能来看，Brain的观点是单个报文的延迟(Single packet latency)并没有那么的重要，更重要的是实现长尾延迟的避免,例如Star-CCM+的测试报告《EFA-enabled C5n instances to scale Simcenter STAR-CCM+》[2]在3000核时加速比都还非常好

![图片](assets/9d6b2489a4f4.png)

ANSYS Fluent性能也非常好

![图片](assets/aa82f24556e4.png)

访谈中Brain还提到高性能存储是影响HPC应用的另一个关键因素，因此构建了FSx for Lustre的支持

![图片](assets/a53bca5f9094.png)

### 3. 一些缺点和争议

AWS通过Reliable Datagram实现了多路径的支持能力，但是似乎国内很多人把这个事情搞混了，虽然传输语义上实现了可交换，但是基于Reliable Connection语义Verbs兼容的情况下依旧可以实现多路径的处理，而且这个技术在2002年IETF提出iWARP时构建的Direct Data Placement(DDP)就已经讨论的很清楚了

![图片](assets/ad444118749a.png)

另外在HPC这个领域，特别是在国内部门间的通信壁垒非常高，很多从业者材料/物理/机械这些专业毕业的，对于HPC软件和相应的求解器只会使用，而IT等部门通常也只是使用商用软件测试招标，相应的算法和通信等优化的团队较少，并且企业通常因为软件授权价格等问题停留在较老的软件版本上。针对这些商用软件生态兼容使得RD这样的语义带来了很多负担。

参考资料

[1] 
How EFA works and why we don't use infiniband in the cloud.: https://www.youtube.com/watch?v=IgPWzhIHX68
[2] 
EFA-enabled C5n instances to scale Simcenter STAR-CCM+: https://aws.amazon.com/cn/blogs/compute/efa-enabled-c5n-instances-to-scale-simcenter-star-ccm/