# MPI Note[3]: 基本通信续

> 作者: zartbot  
> 日期: 2021年4月16日 12:49  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485692&idx=1&sn=d0193c6584b14581d5a08473aaed2d1f&chksm=f996183ecee191289c1508261c4de7dbb26b514996a1aac9285311fca99b160e83ec36248588#rd

---

补充一些来自于MPITutorial的内容

### MPI_Status 

Send、Recv示例中的消息数量是固定的预先知道的，如果有动态的消息传输则需要使用MPI_Status处理

MPI_Status包含了如下一些信息：

`MPI_SOURCE`: 发送端的Rank.

`MPI_Tag`: 消息的Tag

`Count`: 使用`MPI_Get_count`获取消息的长度。

```
#include <stdio.h>#include "mpi.h"#include <stdlib.h>#include <time.h>int main(int argc, char *argv[]){    int world_rank, world_size;    MPI_Init(&argc, &argv);    MPI_Comm_size(MPI_COMM_WORLD, &world_size);    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);        const int MAX_NUMBERS = 100;    int numbers[MAX_NUMBERS];    int number_amount;    if (world_rank == 0)    {        // 取得一个随机数作为发送数据个数        srand(time(NULL));        number_amount = (rand() / (float)RAND_MAX) * MAX_NUMBERS;        // 发送        MPI_Send(numbers, number_amount, MPI_INT, 1, 0, MPI_COMM_WORLD);        printf("0 sent %d numbers to 1\n", number_amount);    }    else if (world_rank == 1)    {        MPI_Status status;                // 按照MAX_NUMERS接收消息        MPI_Recv(numbers, MAX_NUMBERS, MPI_INT, 0, 0, MPI_COMM_WORLD,                 &status);        // 接收消息后，获取消息长度        MPI_Get_count(&status, MPI_INT, &number_amount);        // 打印输出        printf("1 received %d numbers from 0. Message source = %d, "               "tag = %d\n",               number_amount, status.MPI_SOURCE, status.MPI_TAG);    }    MPI_Finalize();}
```

### MPI_Probe 

相对于前述示例，可以使用MPI_Probe在Recv前获取接收状态，并分配接收缓冲区：

```
// Author: Wes Kendall// Copyright 2011 www.mpitutorial.com// This code is provided freely with the tutorials on mpitutorial.com. Feel// free to modify it for your own use. Any distribution of the code must// either provide a link to www.mpitutorial.com or keep this header intact.//// Example of using MPI_Probe to dynamically allocated received messages//#include <mpi.h>#include <stdio.h>#include <stdlib.h>#include <time.h>int main(int argc, char** argv) {  MPI_Init(NULL, NULL);  int world_size;  MPI_Comm_size(MPI_COMM_WORLD, &world_size);  if (world_size != 2) {    fprintf(stderr, "Must use two processes for this example\n");    MPI_Abort(MPI_COMM_WORLD, 1);  }  int world_rank;  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);  int number_amount;  if (world_rank == 0) {    const int MAX_NUMBERS = 100;    int numbers[MAX_NUMBERS];    // Pick a random amont of integers to send to process one    srand(time(NULL));    number_amount = (rand() / (float)RAND_MAX) * MAX_NUMBERS;        // Send the amount of integers to process one    MPI_Send(numbers, number_amount, MPI_INT, 1, 0, MPI_COMM_WORLD);    printf("0 sent %d numbers to 1\n", number_amount);  } else if (world_rank == 1) {    MPI_Status status;    // Probe 获取接收消息的状态    MPI_Probe(0, 0, MPI_COMM_WORLD, &status);    MPI_Get_count(&status, MPI_INT, &number_amount);    // 动态分配接收缓冲区    int* number_buf = (int*)malloc(sizeof(int) * number_amount);        // 真正执行阻塞接收的地方    MPI_Recv(number_buf, number_amount, MPI_INT, 0, 0, MPI_COMM_WORLD,             MPI_STATUS_IGNORE);    printf("1 dynamically received %d numbers from 0.\n",           number_amount);    free(number_buf);  }  MPI_Finalize();}
```

### 节点分组 

很多分布式任务需要将节点分组，MPI里默认会创建一个`MPI_COMM_WORLD`的`MPI_Group`并将所有的节点加入其中。

例如我们可以采用如下方式把节点分配到不同的组中

```
#include <mpi.h>#include <stdio.h>int main(int argc, char *argv[]){    const int sub_group_ranks_1[4] = {0, 1, 2, 3};    const int sub_group_ranks_2[4] = {4, 5, 6, 7};    int world_rank, world_size;    MPI_Group world_group, sub_group_1, sub_group_2;    MPI_Comm sub_comm_1, sub_comm_2;    MPI_Init(&argc, &argv);    //Retrieve the node rank in the world    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);    MPI_Comm_size(MPI_COMM_WORLD, &world_size);    MPI_Comm_group(MPI_COMM_WORLD, &world_group);    //Create group    MPI_Group_incl(world_group, 4, sub_group_ranks_1, &sub_group_1);    MPI_Group_incl(world_group, 4, sub_group_ranks_2, &sub_group_2);    //create new communicator    MPI_Comm_create_group(MPI_COMM_WORLD, sub_group_1, 0, &sub_comm_1);    MPI_Comm_create_group(MPI_COMM_WORLD, sub_group_2, 0, &sub_comm_2);    if (sub_comm_1 != MPI_COMM_NULL)    {        int sub_rank;        MPI_Comm_rank(sub_comm_1, &sub_rank);        printf("WORLD RANK:%d Belongs Sub Group 1,Local Rank:%d\n", world_rank, sub_rank);    }    if (sub_comm_2 != MPI_COMM_NULL)    {        int sub_rank;        MPI_Comm_rank(sub_comm_2, &sub_rank);        printf("WORLD RANK:%d Belongs Sub Group 2,Local Rank:%d\n", world_rank, sub_rank);    }    MPI_Group_free(&world_group);    MPI_Group_free(&sub_group_1);    MPI_Group_free(&sub_group_2);    if (sub_comm_1 != MPI_COMM_NULL)    {        MPI_Comm_free(&sub_comm_1);    }    if (sub_comm_2 != MPI_COMM_NULL)    {        MPI_Comm_free(&sub_comm_2);    }    MPI_Finalize();}
```

执行结果

```
zartbot@mpi1:~/mpi/study/group$ mpicc subgroup.c -o subgroupzartbot@mpi1:~/mpi/study/group$ mpirun -np 8 subgroupWORLD RANK:4 Belongs Sub Group 2,Local Rank:0WORLD RANK:5 Belongs Sub Group 2,Local Rank:1WORLD RANK:6 Belongs Sub Group 2,Local Rank:2WORLD RANK:7 Belongs Sub Group 2,Local Rank:3WORLD RANK:0 Belongs Sub Group 1,Local Rank:0WORLD RANK:1 Belongs Sub Group 1,Local Rank:1WORLD RANK:2 Belongs Sub Group 1,Local Rank:2WORLD RANK:3 Belongs Sub Group 1,Local Rank:3
```

### MPI_Comm_Split 

一种简单的分组方式

```
#include <stdlib.h>#include <stdio.h>#include <mpi.h>int main(int argc, char **argv) {  MPI_Init(NULL, NULL);  // Get the rank and size in the original communicator  int world_rank, world_size;  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);  MPI_Comm_size(MPI_COMM_WORLD, &world_size);  int color = world_rank / 4; // Determine color based on row  // Split the communicator based on the color and use the original rank for ordering  MPI_Comm row_comm;  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &row_comm);  int row_rank, row_size;  MPI_Comm_rank(row_comm, &row_rank);  MPI_Comm_size(row_comm, &row_size);  printf("WORLD RANK/SIZE: %d/%d --- ROW RANK/SIZE: %d/%d\n",    world_rank, world_size, row_rank, row_size);  MPI_Comm_free(&row_comm);  MPI_Finalize();}
```

```
zartbot@mpi1:~/mpi/study/group$ mpirun  -np 8 ./splitWORLD RANK/SIZE: 0/8 --- ROW RANK/SIZE: 0/4WORLD RANK/SIZE: 1/8 --- ROW RANK/SIZE: 1/4WORLD RANK/SIZE: 2/8 --- ROW RANK/SIZE: 2/4WORLD RANK/SIZE: 4/8 --- ROW RANK/SIZE: 0/4WORLD RANK/SIZE: 5/8 --- ROW RANK/SIZE: 1/4WORLD RANK/SIZE: 6/8 --- ROW RANK/SIZE: 2/4WORLD RANK/SIZE: 3/8 --- ROW RANK/SIZE: 3/4WORLD RANK/SIZE: 7/8 --- ROW RANK/SIZE: 3/4
```

### 同步屏障 

有些时候需要进行一些粗颗粒度的并行计算，例如每个local节点进行大批量的模型训练，需要在某一个地方等待一下，大家步调一致再执行下一步。
或者有些需要Benchmark测量的地方，需要停下来统计一下时间：

```
#include <mpi.h>#include <stdio.h>int main(int argc, char *argv[]){    int world_rank, world_size;    double start, end;    MPI_Init(&argc, &argv);    //Retrieve the node rank in the world    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);    MPI_Comm_size(MPI_COMM_WORLD, &world_size);    //start barrier    MPI_Barrier(MPI_COMM_WORLD);    start = MPI_Wtime();    /* some local computations here */    int j = 0;    for(int i=0;i<1000000;i++) {        j++;    }    //end barrier        MPI_Barrier(MPI_COMM_WORLD);     end = MPI_Wtime();            MPI_Finalize();    if (world_rank == 0)    {                                        printf("time %f\n",end-start);    }}
```

```
zartbot@mpi1:~/mpi/study/group$ mpicc barrier.c -o barrierzartbot@mpi1:~/mpi/study/group$ mpirun -np 8 barriertime 0.002982
```

### BSP 

Bulk Synchrononous Parallel,BSP模型计算步骤如下:

`concurrent computation step`: processes locally and asynchronously compute, and those local computation can overlap with communications,

`communication step`: processes exchange data between themselves,

`synchronization barrier step`: when a process reaches a synchronization barrier,it waits for all the other processes to reach this barrier before proceeding another super-step.

### 并行矩阵乘法 

可以用最简单的阻塞通信实现一个并行矩阵乘法

```
#include <mpi.h>#include <stdio.h>int main(int argc, char *argv[]){    int A[4][4], b[4], c[4], line[4], temp[4], local_value, myid;    MPI_Init(&argc, &argv);    MPI_Comm_rank(MPI_COMM_WORLD, &myid);        /* 初始化整个矩阵 */    if (myid == 0)    {         for (int i = 0; i < 4; i++)        {            for (int j = 0; j < 4; j++) {                A[i][j] = i + j;                printf("a[%d][%d]=%d ", i , j , A[i][j]);            }            b[i] = i;            printf("b[%d]=%d\n",i,b[i]);        }                line[0] = A[0][0];        line[1] = A[0][1];        line[2] = A[0][2];        line[3] = A[0][3];    }    if (myid == 0)    {        for (int i = 1; i < 4; i++)        {             //构建temp向量，并将temp和b一起发送给其它节点            temp[0] = A[i][0];            temp[1] = A[i][1];            temp[2] = A[i][2];            temp[3] = A[i][3];            MPI_Send(temp, 4, MPI_INT, i, i,MPI_COMM_WORLD);            MPI_Send(b, 4, MPI_INT, i, i, MPI_COMM_WORLD);        }    }    else    {        //其它节点接收temp和b，并将temp存放到本地的line向量中        MPI_Recv(line, 4, MPI_INT, 0, myid, MPI_COMM_WORLD, MPI_STATUS_IGNORE);        MPI_Recv(b, 4, MPI_INT, 0, myid, MPI_COMM_WORLD, MPI_STATUS_IGNORE);    }    //分布式执行乘法    c[myid] = line[0] * b[0] + line[1] * b[1] + line[2] * b[2] + line[3] * b[3];    if (myid != 0)    {        //发送结果到master节点        MPI_Send(&c[myid], 1, MPI_INT, 0, myid,MPI_COMM_WORLD);    }    else    {        //master打印自己的结果        printf("c[0]=%d\n", c[0]);        for (int i = 1; i < 4; i++)        {            //master打印其它的结果            MPI_Recv(&c[i], 1, MPI_INT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);            printf("c[%d]=%d\n", i, c[i]);        }    }    MPI_Finalize();    return 0;}
```

执行结果：

```
zartbot@mpi1:~/mpi/study/matrix$ mpicc vector.c -o vectorzartbot@mpi1:~/mpi/study/matrix$ mpirun -np 4 vectora[0][0]=0 a[0][1]=1 a[0][2]=2 a[0][3]=3 b[0]=0a[1][0]=1 a[1][1]=2 a[1][2]=3 a[1][3]=4 b[1]=1a[2][0]=2 a[2][1]=3 a[2][2]=4 a[2][3]=5 b[2]=2a[3][0]=3 a[3][1]=4 a[3][2]=5 a[3][3]=6 b[3]=3c[0]=14c[1]=20c[2]=26c[3]=32
```

### Reduce Example 

Reduce函数如下：

```
int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
```

我们以一个简单的阶乘为例， sendbuf为number变量，recvbuf为globalFact，count=1，MPI_Op为MPI_Prod,根节点为0.

中间我们加了一些屏障用于输出中间过程。

```
#include <stdio.h>#include "mpi.h"int main(int argc, char *argv[]){    int i, me, nprocs;    int number, globalFact = -1, localFact;    MPI_Init(&argc, &argv);    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);    MPI_Comm_rank(MPI_COMM_WORLD, &me);    number = me + 1;    MPI_Barrier(MPI_COMM_WORLD);    printf("[before]rank: %d local number:%d GlobalFact: %d\n", me, number, globalFact);    MPI_Reduce(&number, &globalFact, 1, MPI_INT, MPI_PROD, 0, MPI_COMM_WORLD);    MPI_Barrier(MPI_COMM_WORLD);    printf("[after]rank: %d local number:%d GlobalFact: %d\n", me, number, globalFact);    MPI_Barrier(MPI_COMM_WORLD);    if (me == 0)    {        printf("Computing the factorial in MPI: %d processus = %d\n", nprocs, globalFact);        localFact = 1;        for (i = 0; i < nprocs; i++)        {            localFact *= (i + 1);        }        printf("Versus local factorial: %d\n", localFact);    }    MPI_Finalize();}
```

执行结果:

```
zartbot@mpi1:~/mpi/study/matrix$ mpirun -np 8 reduce[before]rank: 0 local number:1 GlobalFact: -1[before]rank: 1 local number:2 GlobalFact: -1[before]rank: 2 local number:3 GlobalFact: -1[before]rank: 3 local number:4 GlobalFact: -1[before]rank: 4 local number:5 GlobalFact: -1[before]rank: 6 local number:7 GlobalFact: -1[before]rank: 7 local number:8 GlobalFact: -1[before]rank: 5 local number:6 GlobalFact: -1[after]rank: 0 local number:1 GlobalFact: 40320[after]rank: 2 local number:3 GlobalFact: -1[after]rank: 3 local number:4 GlobalFact: -1[after]rank: 7 local number:8 GlobalFact: -1[after]rank: 4 local number:5 GlobalFact: -1[after]rank: 6 local number:7 GlobalFact: -1[after]rank: 5 local number:6 GlobalFact: -1[after]rank: 1 local number:2 GlobalFact: -1Computing the factorial in MPI: 8 processus = 40320Versus local factorial: 40320
```