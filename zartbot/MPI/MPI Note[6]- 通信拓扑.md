# MPI Note[6]: 通信拓扑

> 作者: zartbot  
> 日期: 2021年4月25日 15:30  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485760&idx=2&sn=0cc41b35dc5528266e45d19ad8ae20fe&chksm=f9961982cee190949750000968aa800d4c7533e0f2fe379e892171d382e89ce15f7a9833fa21#rd

---

### 通信拓扑 

在很多超算的网络拓扑结构中并没有像我们传统的数据中心网络那样选择FatTree的架构，而通常为了更好的扩展性和分区资源调度的特性，选择了*D-Torus的结构。最具代表的一个是Google TPU的集群构成的2D-Torus，另一个是Fugaku**超算构建的6D-Torus.

今年应该是中美E级超算**都要问世的一年，倒是非常期待它们能在体系结构上带来一些新的思路。

### 环形拓扑通信 

在分布式深度学习平台中经常会听到一种叫Ring AllReduce的算法，本质上就是一个环形通信，而2D-Torus一类的也就是在东西向环一次，然后再南北向环一圈。

我们需要采用如下的方法环形结构中，前导和后继的节点，如下:

```
int next(){     int rank,size;    MPI_Comm_rank(MPI_COMM_WORLD, &rank);    MPI_Comm_size(MPI_COMM_WORLD, &size);    return ( (rank+1) % size);}int prev(){     int rank,size;    MPI_Comm_rank(MPI_COMM_WORLD, &rank);    MPI_Comm_size(MPI_COMM_WORLD, &size);    return ( (size + rank - 1) % size);}
```

然后就是简单的阻塞通信构造一个RingBcast就好：

```
int main(int argc, char * argv[]) {    int world_rank, world_size, value;        value = 12345;    MPI_Status status;    MPI_Init(&argc,&argv);    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);    MPI_Comm_size(MPI_COMM_WORLD, &world_size);    if (world_rank == 0){        MPI_Send(&value, 1 , MPI_INT, next(),0,MPI_COMM_WORLD);    } else {        MPI_Recv(&value, 1 , MPI_INT, prev(),0,MPI_COMM_WORLD, &status);        printf("proc %d recv %d\n",world_rank,value);        if (world_rank < world_size - 1) {            MPI_Send(&value, 1 , MPI_INT, next(),0,MPI_COMM_WORLD);        }    }    MPI_Finalize();}
```

### Ring Scatter 

```
#include <mpi.h>#include <stdio.h>int main (int argc, char * argv[]) {    int world_rank, world_size, value;            MPI_Request req;    MPI_Status status;    MPI_Init(&argc,&argv);    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);    MPI_Comm_size(MPI_COMM_WORLD, &world_size);    if (world_rank == 0) {        //准备原始数据        int values[world_size-1];        for (int i=0; i<world_size;i++) {            values[i]=i*i;        }       //采用非阻塞逆序发送        for (int i=1; i < world_size;i++) {            printf("[Init]node 0 sending value %d to node %d, intended for node %d\n",values[world_size-1-i],1,world_size-i);            MPI_Isend(&values[world_size-1-i],1,MPI_INT,1,0,MPI_COMM_WORLD,&req);        }    } else {        int recv_value;        int transfer_value;        //处理中继转发的变量，依旧采用非阻塞        for (int i = world_rank;i<world_size-1;i++ ) {            MPI_Recv(&transfer_value,1,MPI_INT,world_rank-1,0,MPI_COMM_WORLD,&status);            printf("[Trans]node %d recv value %d for node %d, transfers node %d\n",world_rank,transfer_value,world_size-1-i+world_rank,world_rank+1);            MPI_Isend(&transfer_value,1,MPI_INT, world_rank+1,0,MPI_COMM_WORLD,&req);        }        //处理节点应当接受的数据        MPI_Recv(&recv_value,1,MPI_INT,world_rank-1,0,MPI_COMM_WORLD,&status);        printf("[Final]node %d recieved value %d from node %d\n",world_rank,recv_value,world_rank-1);    }    MPI_Finalize();    return 0;}
```

```
kevin@netdev:~/Desktop/mpi/topo$ mpicc ringscatter.c -o rskevin@netdev:~/Desktop/mpi/topo$ mpirun -np 6 ./rs[Init]node 0 sending value 16 to node 1, intended for node 5[Init]node 0 sending value 9 to node 1, intended for node 4[Init]node 0 sending value 4 to node 1, intended for node 3[Init]node 0 sending value 1 to node 1, intended for node 2[Init]node 0 sending value 0 to node 1, intended for node 1[Trans]node 1 recv value 16 for node 5, transfers node 2[Trans]node 1 recv value 9 for node 4, transfers node 2[Trans]node 2 recv value 16 for node 5, transfers node 3[Trans]node 2 recv value 9 for node 4, transfers node 3[Trans]node 3 recv value 16 for node 5, transfers node 4[Trans]node 4 recv value 16 for node 5, transfers node 5[Trans]node 3 recv value 9 for node 4, transfers node 4[Final]node 5 recieved value 16 from node 4[Final]node 4 recieved value 9 from node 3[Trans]node 1 recv value 4 for node 3, transfers node 2[Trans]node 1 recv value 1 for node 2, transfers node 2[Final]node 1 recieved value 0 from node 0[Trans]node 2 recv value 4 for node 3, transfers node 3[Final]node 2 recieved value 1 from node 1[Final]node 3 recieved value 4 from node 2
```