# MPI Note[8]: 分布式机器学习AllReduce

> 作者: zartbot  
> 日期: 2021年4月27日 16:59  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485803&idx=1&sn=8fea4ef0df7f89c3c0ff281b8a5cbd8e&chksm=f99619a9cee190bfe8a038d31b6c44803dd90113e01da78ca52fe71876ffe5c337c4932f2b6b#rd

---

❝
最近华为发布了2000亿参数的模型盘古，阿里也有推出了270亿参数的PLUG模型.竞争越来越激烈。另一方面则是中美之间关于E级超算的竞争。
❞
关于分布式机器学习，以前有一篇文章讲过一些内容[1].神经网络计算通常是一些`矩阵-向量`乘法是指， 其中是一个维的矩阵，  是一个 维向量，列向量  可以按如下方式计算:

并行算法相对简单，我们可以为每个进程分配  的  行进行计算即可. 或者对矩阵进行一些特殊的划分减小通信量:

![图片](assets/f9cbd88c72f8.png)

当然矩阵的并行计算我们还是在下一个章节再来谈，今天先来看另一个重要的问题，训练参数的同步，也就是经常会听到的一个词`AllReduce`.而且随着系统规模扩大，它对整个训练速度的影响越来越关键..

![图片](assets/3cf3489376f8.png)

### 分布式机器学习 

由于`数据规模`和`模型规模`的扩大必须利用计算机`集群`来构建`分布式`机器学习框架，并行处理的方式主要是在两个维度。数据规模比较好理解随着IOT和数据采集，可供于训练的数据集通常可以达到数十TB，
而模型规模主要是业界竞争的重点，例如`世界第一`华为盘古2000亿, `全球领先`GPT-3参数已经达到1700亿，而常见的各厂家模型也到了数十亿级(nVidia MegatronLM 83亿，微软Turing-NLG 170亿,阿里PLUG 270亿).

`数据并行`很容易解释，主要是如何存储训练样本，并且在多机器之间传递混淆样本，基本上大家大同小异的都在采用SSD、分布式存储解决这些问题.另一个问题便是`模型并行`,当单个工作节点无法存储时，就需要对模型本身进行分割，而如何分割这个模型又是一个最优化的过程...
![图片](assets/3b6340cf95e7.png)

模型分割了，那么就存在一个非常关键的问题了，参数的同步:
![图片](assets/ffb5cd2a5199.png)

而通常的做法是采用Model Average(MA)需要将每个子模型的参数平均，那么需要在多个节点中将参数求和的处理方式：
![图片](assets/601a228cf14f.jpg)

这个过程就是AllReduce操作.关于AllReduce可以参考腾讯机智团队分享--AllReduce算法的前世今生[2] 讲的蛮细的，而今天我们专门来用MPI实现一下RingAllreduce[3]

`RingAllreduce`并不是需要将Underlay网络构造成一个环，普通的CLOS交换网(当然要保证无阻塞)就可以了，通过环状通信的方式，完全利用了所有的带宽，并且非常聪明的避开了`Incast`的影响.

### Ring AllReduce 

#### 准备数据

我们还是采用一个MPI集群来做这个实验，首先我们来构造一些数据，因为没有那么多张GPU和服务器集群，所以伪造几百万个便于验证的浮点数作为参数集，同样的参数构造两组是为了对比ringAllReduce和腾讯提出的层次化ringAllReduce

```
    int num_ele_per_node = atoi(argv[2]);        float *local_param = (float *)malloc(sizeof(float) * num_ele_per_node);    float *local_param2 = (float *)malloc(sizeof(float) * num_ele_per_node);    for (int i = 0; i < num_ele_per_node; i++)    {        local_param[i] = world_rank;        local_param2[i] = world_rank;    }
```

#### 标准库实现

标准的MPI Allreduce实现如下,稍后我们会用global_sum来验证我们自己写的ringAllreduce结果.

```
    float *global_sum = (float *)malloc(sizeof(float) * num_ele_per_node);    MPI_Allreduce(local_param, global_sum, num_ele_per_node, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
```

当然这一步也可以采用如下方式计时对比后面的ringAllreduce算法.

```
    //start time    MPI_Barrier(MPI_COMM_WORLD);    mpi_start_time = MPI_Wtime();    MPI_Allreduce(local_param, global_sum, num_ele_per_node, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);        //endtime    MPI_Barrier(MPI_COMM_WORLD);    mpi_end_time = MPI_Wtime();    if (world_rank == 0)        printf("mpi  time:%15.0fus\n", (mpi_end_time - mpi_start_time) * 1e6);
```

#### RingAllreduce

接下来我们来构造这个函数, Reduce Operator我们hardcode成reduceSUM了.

```
void reduceSUM(float *dst, float *src, int size){    for (int i = 0; i < size; i++)        dst[i] += src[i];}void ringAllReduce(float *data, int count, MPI_Datatype datatype, MPI_Comm communicator)
```

首先我们需要将数据划分成N个Segment,如果不能整除，那么剩下的数每个segment多一个,尽量使得segment划分均匀，然后再划分的同时，我们记录下每段的开始元素位置.

![图片](assets/27e35b1737cf.png)

```
  int comm_rank;    int comm_size;    MPI_Comm_rank(communicator, &comm_rank);    MPI_Comm_size(communicator, &comm_size);    //split dataset    int segment_size = count / comm_size;    int residual = count % comm_size;    int *segment_sizes = (int *)malloc(sizeof(int) * comm_size);    int *segment_start_ptr = (int *)malloc(sizeof(int) * comm_size);    int segment_ptr = 0;    for (int i = 0; i < comm_size; i++)    {        segment_start_ptr[i] = segment_ptr;        segment_sizes[i] = segment_size;        if (i < residual)            segment_sizes[i]++;        segment_ptr += segment_sizes[i];    }        //verify    if (segment_start_ptr[comm_size - 1] + segment_sizes[comm_size - 1] != count)    {        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_COUNT);    }
```

接下来我们做N-1轮迭代，每一轮中，我们将数据发送到下一个节点，然后由下一个节点接收完成后，执行reduceSUM

![图片](assets/eac4b84f167f.png)

你可以注意到每个节点的上一个节点和下一个节点为

```
int next(int rank, int size){    return ((rank + 1) % size);}int prev(int rank, int size){    return ((size + rank - 1) % size);}
```

而每一轮迭代需要传输的segment为

```
   int recv_chunk = (comm_rank - iter - 1 + comm_size) % comm_size;   int send_chunk = (comm_rank - iter + comm_size) % comm_size;
```

因此RingAllreduce计算如下：

```
    MPI_Status recv_status;    MPI_Request recv_req;    float *buffer = (float *)malloc(sizeof(float) * segment_sizes[0]);    for (int iter = 0; iter < comm_size - 1; iter++)    {        int recv_chunk = (comm_rank - iter - 1 + comm_size) % comm_size;        int send_chunk = (comm_rank - iter + comm_size) % comm_size;        float *sending_segment = &(data[segment_start_ptr[send_chunk]]);        MPI_Irecv(buffer, segment_sizes[recv_chunk], datatype, prev(comm_rank, comm_size), 0, communicator, &recv_req);        MPI_Send(sending_segment, segment_sizes[send_chunk], datatype, next(comm_rank, comm_size), 0, communicator);        float *updating_segment = &(data[segment_start_ptr[recv_chunk]]);        MPI_Wait(&recv_req, &recv_status);        //after send recieve finshed, execute reduce        reduceSUM(updating_segment, buffer, segment_sizes[recv_chunk]);    }
```

![图片](assets/4b88e2b990bc.png)

紧接着第二三四轮... 

![图片](assets/ae4374f2c310.png)

N-1轮处理完了然后执行AllGather操作，将每个完成计算的Segment数据分发给其它节点. 

![图片](assets/3f4ee0f02d49.png)

```
    for (int iter = 0; iter < comm_size - 1; iter++)    {        int recv_chunk = (comm_rank - iter + comm_size) % comm_size;        int send_chunk = (comm_rank - iter + 1 + comm_size) % comm_size;        float *sending_segment = &(data[segment_start_ptr[send_chunk]]);        float *updating_segment = &(data[segment_start_ptr[recv_chunk]]);        MPI_Sendrecv(sending_segment, segment_sizes[send_chunk], datatype, next(comm_rank, comm_size), 0, updating_segment, segment_sizes[recv_chunk], datatype, prev(comm_rank, comm_size), 0, communicator, &recv_status);    }
```

后续流程如下：

![图片](assets/9955a7c3912a.png)

当然我们在计算时还可以考虑NUMA结构或者物理机器多核结构，采用层次化的方式处理,例如腾讯利用GPU的NVLink带宽，组内直接Reduce，然后组间在RingAllReduce，做完后再组内BCast
![图片](assets/8ea50f864bdf.jpg)

我们来看一下MPI节点分割的技巧, 通过取模染色`color`和`MPI_Comm_split`即可分为不同的subgroup，然后将每个subgroup中找一个leader出来作为maingroup.

```
 // Split subgroup by host and create SUBGRP_COMM    int color = world_rank / node_per_host;    MPI_Comm SUBGRP_COMM;    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &SUBGRP_COMM);    int subgrp_rank, subgrp_size;    MPI_Comm_rank(SUBGRP_COMM, &subgrp_rank);    MPI_Comm_size(SUBGRP_COMM, &subgrp_size);    // Create main group    MPI_Group main_grp, world_grp;    MPI_Comm MAINGRP_COMM;    MPI_Comm_group(MPI_COMM_WORLD, &world_grp);    int host_num = world_size / node_per_host;    int *maingrp_ranks = (int *)malloc(sizeof(int) * host_num);    for (int i = 0; i < host_num; i++)    {        maingrp_ranks[i] = i * node_per_host;    }    MPI_Group_incl(world_grp, host_num, maingrp_ranks, &main_grp);    MPI_Comm_create_group(MPI_COMM_WORLD, main_grp, 0, &MAINGRP_COMM);
```

后面我们就用到了两种算法来对比,第一个是层次化的算法

```
    ringAllReduce(local_param, num_ele_per_node, MPI_FLOAT, SUBGRP_COMM);    if (MAINGRP_COMM != MPI_COMM_NULL)    {        ringAllReduce(local_param, num_ele_per_node, MPI_FLOAT, MAINGRP_COMM);    }    MPI_Bcast(local_param, num_ele_per_node, MPI_FLOAT, 0, SUBGRP_COMM);
```

第二个是直接对MPI_COMM_WORLD加载ringAllReduce

```
ringAllReduce(local_param2, num_ele_per_node, MPI_FLOAT, MPI_COMM_WORLD);
```

计算结果

```
kevin@netdev:~/Desktop/mpi/07_ringallreduce$ mpicc collective.c main.c -o fookevin@netdev:~/Desktop/mpi/07_ringallreduce$ mpiexec -np 16 ./foo 4 6000000build-in mpi  time:         146276ushierachy-ring time:         163840usallnode-ring  time:         101049us
```

源代码
collective.c

```
#include <stdio.h>#include <stdlib.h>#include <stdint.h>#include "mpi.h"#include "collective.h"int next(int rank, int size){    return ((rank + 1) % size);}int prev(int rank, int size){    return ((size + rank - 1) % size);}void reduceSUM(float *dst, float *src, int size){    for (int i = 0; i < size; i++)        dst[i] += src[i];}void ringAllReduce(float *data, int count, MPI_Datatype datatype, MPI_Comm communicator){    int comm_rank;    int comm_size;    MPI_Comm_rank(communicator, &comm_rank);    MPI_Comm_size(communicator, &comm_size);    //split dataset    int segment_size = count / comm_size;    int residual = count % comm_size;    int *segment_sizes = (int *)malloc(sizeof(int) * comm_size);    int *segment_start_ptr = (int *)malloc(sizeof(int) * comm_size);    int segment_ptr = 0;    for (int i = 0; i < comm_size; i++)    {        segment_start_ptr[i] = segment_ptr;        segment_sizes[i] = segment_size;        if (i < residual)            segment_sizes[i]++;        segment_ptr += segment_sizes[i];    }    if (segment_start_ptr[comm_size - 1] + segment_sizes[comm_size - 1] != count)    {        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_COUNT);    }    MPI_Status recv_status;    MPI_Request recv_req;    float *buffer = (float *)malloc(sizeof(float) * segment_sizes[0]);    for (int iter = 0; iter < comm_size - 1; iter++)    {        int recv_chunk = (comm_rank - iter - 1 + comm_size) % comm_size;        int send_chunk = (comm_rank - iter + comm_size) % comm_size;        float *sending_segment = &(data[segment_start_ptr[send_chunk]]);        MPI_Irecv(buffer, segment_sizes[recv_chunk], datatype, prev(comm_rank, comm_size), 0, communicator, &recv_req);        MPI_Send(sending_segment, segment_sizes[send_chunk], datatype, next(comm_rank, comm_size), 0, communicator);        float *updating_segment = &(data[segment_start_ptr[recv_chunk]]);        MPI_Wait(&recv_req, &recv_status);        //after send recieve finshed, execute reduce        reduceSUM(updating_segment, buffer, segment_sizes[recv_chunk]);    }    MPI_Barrier(communicator);    //allGather    for (int iter = 0; iter < comm_size - 1; iter++)    {        int recv_chunk = (comm_rank - iter + comm_size) % comm_size;        int send_chunk = (comm_rank - iter + 1 + comm_size) % comm_size;        float *sending_segment = &(data[segment_start_ptr[send_chunk]]);        float *updating_segment = &(data[segment_start_ptr[recv_chunk]]);        MPI_Sendrecv(sending_segment, segment_sizes[send_chunk], datatype, next(comm_rank, comm_size), 0, updating_segment, segment_sizes[recv_chunk], datatype, prev(comm_rank, comm_size), 0, communicator, &recv_status);    }    free(buffer);}
```

collective.h

```
void ringAllReduce(float *data, int count,  MPI_Datatype datatype, MPI_Comm communicator);
```

main.c

```
#include <stdio.h>#include <stdlib.h>#include <stdint.h>#include "mpi.h"#include "collective.h"int main(int argc, char **argv){    if (argc != 3)    {        fprintf(stderr, "Usage: allreduce node_per_host num_ele_per_node\n");        exit(1);    }    int node_per_host = atoi(argv[1]);    int num_ele_per_node = atoi(argv[2]);    double mpi_start_time, mpi_end_time;    int world_rank, world_size;    MPI_Init(&argc, &argv);    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);    MPI_Comm_size(MPI_COMM_WORLD, &world_size);    if ( world_size % node_per_host!=0 ) {        printf("Invalid configuration...");        MPI_Abort(MPI_COMM_WORLD,MPI_ERR_BASE);    }    // Prepare Data    float *local_param = (float *)malloc(sizeof(float) * num_ele_per_node);    float *local_param2 = (float *)malloc(sizeof(float) * num_ele_per_node);    for (int i = 0; i < num_ele_per_node; i++)    {        local_param[i] = world_rank;        local_param2[i] = world_rank;    }    //Build-in Allreduce as a reference    float *global_sum = (float *)malloc(sizeof(float) * num_ele_per_node);    MPI_Barrier(MPI_COMM_WORLD);    mpi_start_time = MPI_Wtime();    MPI_Allreduce(local_param, global_sum, num_ele_per_node, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);    MPI_Barrier(MPI_COMM_WORLD);    mpi_end_time = MPI_Wtime();    if (world_rank == 0)        printf("build-in mpi  time:%15.0fus\n", (mpi_end_time - mpi_start_time) * 1e6);#ifdef DEBUG    if (world_rank == 0)    {        for (int i = 0; i < num_ele_per_node; i++)            printf("global_sum[%d]: %f,  avg: %f\n", i, global_sum[i], global_sum[i] / world_size);    }#endif    MPI_Barrier(MPI_COMM_WORLD);    mpi_start_time = MPI_Wtime();    // Split subgroup by host and create SUBGRP_COMM    int color = world_rank / node_per_host;    MPI_Comm SUBGRP_COMM;    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &SUBGRP_COMM);    int subgrp_rank, subgrp_size;    MPI_Comm_rank(SUBGRP_COMM, &subgrp_rank);    MPI_Comm_size(SUBGRP_COMM, &subgrp_size);    // Create main group    MPI_Group main_grp, world_grp;    MPI_Comm MAINGRP_COMM;    MPI_Comm_group(MPI_COMM_WORLD, &world_grp);    int host_num = world_size / node_per_host;    int *maingrp_ranks = (int *)malloc(sizeof(int) * host_num);    for (int i = 0; i < host_num; i++)    {        maingrp_ranks[i] = i * node_per_host;    }    MPI_Group_incl(world_grp, host_num, maingrp_ranks, &main_grp);    MPI_Comm_create_group(MPI_COMM_WORLD, main_grp, 0, &MAINGRP_COMM);#ifdef DEBUG    //Validate COMM_GROUP    if (MAINGRP_COMM != MPI_COMM_NULL)    {        printf("WORLD RANK:%d Local Rank:%d , in MAIN_GROUP\n", world_rank, subgrp_rank);    }    else    {        printf("WORLD RANK:%d Local Rank:%d \n", world_rank, subgrp_rank);    }#endif    ringAllReduce(local_param, num_ele_per_node, MPI_FLOAT, SUBGRP_COMM);    if (MAINGRP_COMM != MPI_COMM_NULL)    {        ringAllReduce(local_param, num_ele_per_node, MPI_FLOAT, MAINGRP_COMM);    }    MPI_Bcast(local_param, num_ele_per_node, MPI_FLOAT, 0, SUBGRP_COMM);    MPI_Barrier(MPI_COMM_WORLD);    mpi_end_time = MPI_Wtime();    if (world_rank == 0)    {        printf("hierachy-ring time:%15.0fus\n", (mpi_end_time - mpi_start_time) * 1e6);    }    //validate    for (int i = 0; i < num_ele_per_node; i++)        if (local_param[i] - global_sum[i])        {            printf("Node[%d]allreduce Error: Local[%f] Global[%f]\n", world_rank, local_param[i], global_sum[i]);            MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ASSERT);        }    MPI_Barrier(MPI_COMM_WORLD);    mpi_start_time = MPI_Wtime();    ringAllReduce(local_param2, num_ele_per_node, MPI_FLOAT, MPI_COMM_WORLD);    MPI_Barrier(MPI_COMM_WORLD);    mpi_end_time = MPI_Wtime();    if (world_rank == 0)    {        printf("allnode-ring  time:%15.0fus\n", (mpi_end_time - mpi_start_time) * 1e6);    }    free(maingrp_ranks);    free(global_sum);    free(local_param);    free(local_param2);    MPI_Finalize();}
```

#### Reference

[1]
Ruta for AI：分布式机器学习的网络优化: https://mp.weixin.qq.com/s/bCX4Rbyb21NbDrgNg5Utlw
[2]
腾讯机智团队分享--AllReduce算法的前世今生: https://zhuanlan.zhihu.com/p/79030485
[3]
浅谈Tensorflow分布式架构：ring all-reduce算法: https://zhuanlan.zhihu.com/p/69797852