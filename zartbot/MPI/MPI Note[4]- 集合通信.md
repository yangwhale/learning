# MPI Note[4]: 集合通信

> 作者: zartbot  
> 日期: 2021年4月17日 13:07  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485705&idx=2&sn=5a88e3108a4cf283192dfe740962beed&chksm=f99619cbcee190dd816d5277c4688edb1deb7cf47d9755d620611657151bda229b0faeb91278#rd

---

前面讲完了点到点通信，下面来看另一个话题，集体(Collective)通信.
点到点的时候已经介绍了一些关于MPI_Barrier的内容:

![图片](assets/583b6ba1a8f0.png)

接下来介绍一些BCast、Scatter、Gather的操作

### MPI_Bcast 

半开玩笑的说,应该叫组播更合适，因为MPI_Bcast有明确的`MPI_Comm`组，和组播源`root`的概念

```
MPI_Bcast(void* data, int count, MPI_Datatype datatype, int root, MPI_Comm communicator)
```

这种模式用于将相同的数据分发到其它节点，如下图所示：

![图片](assets/b86fa1cf47b0.png)

当然一开始，我们就可以用单播的方式，一个个的发就好

```
void my_bcast(void* data, int count, MPI_Datatype datatype, int root, MPI_Comm communicator) {  int world_rank;  int world_size;  MPI_Comm_rank(communicator, &world_rank);  MPI_Comm_size(communicator, &world_size);  if (world_rank == root) {    //作为根节点逐个单播发送    int i;    for (i = 0; i < world_size; i++) {      if (i != world_rank) {        MPI_Send(data, count, datatype, i, 0, communicator);      }    }  } else {    //其它节点阻塞接收    MPI_Recv(data, count, datatype, root, 0, communicator, MPI_STATUS_IGNORE);  }}
```

但是系统自带的更加高效的一种树状分发结构：

![图片](assets/4d058de86875.png)

但是这样的分发效率并不是很高，初期泛洪速度有些慢，而思科ASR1000组播也是类似的，只是一开始就复制N份出去，第二轮N*N...N是几不能说

```
// Author: Wes Kendall// Copyright 2011 www.mpitutorial.com// This code is provided freely with the tutorials on mpitutorial.com. Feel// free to modify it for your own use. Any distribution of the code must// either provide a link to www.mpitutorial.com or keep this header intact.//// Comparison of MPI_Bcast with the my_bcast function//#include <stdio.h>#include <stdlib.h>#include <mpi.h>#include <assert.h>int main(int argc, char** argv) {  if (argc != 3) {    fprintf(stderr, "Usage: compare_bcast num_elements num_trials\n");    exit(1);  }  int num_elements = atoi(argv[1]);  int num_trials = atoi(argv[2]);  MPI_Init(NULL, NULL);  int world_rank;  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);  double total_my_bcast_time = 0.0;  double total_mpi_bcast_time = 0.0;  int i;  int* data = (int*)malloc(sizeof(int) * num_elements);  assert(data != NULL);  for (i = 0; i < num_trials; i++) {    //标记起始时间    MPI_Barrier(MPI_COMM_WORLD);    total_my_bcast_time -= MPI_Wtime();        //开始测试    my_bcast(data, num_elements, MPI_INT, 0, MPI_COMM_WORLD);        //结束时间    MPI_Barrier(MPI_COMM_WORLD);    total_my_bcast_time += MPI_Wtime();        MPI_Barrier(MPI_COMM_WORLD);    total_mpi_bcast_time -= MPI_Wtime();        MPI_Bcast(data, num_elements, MPI_INT, 0, MPI_COMM_WORLD);        MPI_Barrier(MPI_COMM_WORLD);    total_mpi_bcast_time += MPI_Wtime();  }  // Print off timing information  if (world_rank == 0) {    printf("Data size = %d, Trials = %d\n", num_elements * (int)sizeof(int),           num_trials);    printf("Avg my_bcast time = %lf\n", total_my_bcast_time / num_trials);    printf("Avg MPI_Bcast time = %lf\n", total_mpi_bcast_time / num_trials);  }  free(data);  MPI_Finalize();}
```

### MPI_Scatter 

MPI_Scatter 和MPI_Bcast的不同点在于，Scatter针对每个非root节点分发的数据都不同。

![图片](assets/2ed3a7c1368d.png)

```
MPI_Scatter(    void* send_data,    int send_count,    MPI_Datatype send_datatype,    void* recv_data,    int recv_count,    MPI_Datatype recv_datatype,    int root,    MPI_Comm communicator)
```

具体实例如下,首先我们在Root节点构造了一个array，然后每个node构造了用于接受的sub_array，并且在MPI_Scatter中定义好了发送和接收的数量。

```
#include <stdio.h>#include <stdlib.h>#include <mpi.h>int main(int argc, char **argv){    if (argc != 2)    {        fprintf(stderr, "Usage: scatter num_ele_per_node\n");        exit(1);    }    int num_ele_per_node = atoi(argv[1]);    MPI_Init(NULL, NULL);    int world_rank;    int world_size;    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);    MPI_Comm_size(MPI_COMM_WORLD, &world_size);    float *array = NULL;    if (world_rank == 0)    {        array = (float *)malloc(sizeof(float) * world_size * num_ele_per_node);        for (int i = 0; i < world_size * num_ele_per_node; i++)        {            array[i] = i;        }    }    float *sub_array = (float *)malloc(sizeof(float) * num_ele_per_node);    MPI_Scatter(array, num_ele_per_node, MPI_FLOAT, sub_array, num_ele_per_node, MPI_FLOAT, 0, MPI_COMM_WORLD);    MPI_Barrier(MPI_COMM_WORLD);    for (int i = 0; i < num_ele_per_node; i++)        printf("proc %d sub_array[%d]: %f\n", world_rank, i, sub_array[i]);    if (world_rank == 0)    {        free(array);    }    free(sub_array);    MPI_Barrier(MPI_COMM_WORLD);    MPI_Finalize();}
```

执行

```
zartbot@mpi1:~/mpi/study/collective$ mpicc scatter.c -o scatterzartbot@mpi1:~/mpi/study/collective$ mpiexec -np 8 ./scatter 2proc 0 sub_array[0]: 0.000000proc 0 sub_array[1]: 1.000000proc 1 sub_array[0]: 2.000000proc 1 sub_array[1]: 3.000000proc 2 sub_array[0]: 4.000000proc 2 sub_array[1]: 5.000000proc 3 sub_array[0]: 6.000000proc 3 sub_array[1]: 7.000000proc 4 sub_array[0]: 8.000000proc 4 sub_array[1]: 9.000000proc 5 sub_array[0]: 10.000000proc 5 sub_array[1]: 11.000000proc 6 sub_array[0]: 12.000000proc 6 sub_array[1]: 13.000000proc 7 sub_array[0]: 14.000000proc 7 sub_array[1]: 15.000000zartbot@mpi1:~/mpi/study/collective$ mpiexec -np 8 ./scatter 3proc 0 sub_array[0]: 0.000000proc 0 sub_array[1]: 1.000000proc 0 sub_array[2]: 2.000000proc 1 sub_array[0]: 3.000000proc 1 sub_array[1]: 4.000000proc 1 sub_array[2]: 5.000000proc 2 sub_array[0]: 6.000000proc 2 sub_array[1]: 7.000000proc 2 sub_array[2]: 8.000000proc 3 sub_array[0]: 9.000000proc 3 sub_array[1]: 10.000000proc 3 sub_array[2]: 11.000000proc 4 sub_array[0]: 12.000000proc 4 sub_array[1]: 13.000000proc 4 sub_array[2]: 14.000000proc 5 sub_array[0]: 15.000000proc 5 sub_array[1]: 16.000000proc 5 sub_array[2]: 17.000000proc 6 sub_array[0]: 18.000000proc 6 sub_array[1]: 19.000000proc 6 sub_array[2]: 20.000000proc 7 sub_array[0]: 21.000000proc 7 sub_array[1]: 22.000000proc 7 sub_array[2]: 23.000000
```

### MPI_Gather 

Gather和Scatter刚好相反，如下图所示：

![图片](assets/22c6bf0e4fd6.png)

```
MPI_Gather(    void* send_data,    int send_count,    MPI_Datatype send_datatype,    void* recv_data,    int recv_count,    MPI_Datatype recv_datatype,    int root,    MPI_Comm communicator)
```

例如我们来做一个示例

```
#include <stdio.h>#include <stdlib.h>#include <mpi.h>int main(int argc, char **argv){    if (argc != 2)    {        fprintf(stderr, "Usage: scatter num_ele_per_node\n");        exit(1);    }    int num_ele_per_node = atoi(argv[1]);    MPI_Init(NULL, NULL);    int world_rank;    int world_size;    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);    MPI_Comm_size(MPI_COMM_WORLD, &world_size);    float *all_array = NULL;    float *array = (float *)malloc(sizeof(float) * num_ele_per_node);    for (int i = 0; i < num_ele_per_node; i++)    {        array[i] = world_rank;    }    if (world_rank == 0)        all_array = (float *)malloc(sizeof(float) * world_size * num_ele_per_node);    MPI_Gather(array, num_ele_per_node, MPI_FLOAT, all_array, num_ele_per_node, MPI_FLOAT, 0, MPI_COMM_WORLD);    MPI_Barrier(MPI_COMM_WORLD);    if (world_rank == 0)    {        for (int i = 0; i < world_size * num_ele_per_node; i++)            printf("all_array[%d]: %f\n", i, all_array[i]);        free(all_array);    }    free(array);    MPI_Barrier(MPI_COMM_WORLD);    MPI_Finalize();}
```

执行：

```
zartbot@mpi1:~/mpi/study/collective$ mpicc gather.c -o gatherzartbot@mpi1:~/mpi/study/collective$ mpiexec -np 8 ./gather 3all_array[0]: 0.000000all_array[1]: 0.000000all_array[2]: 0.000000all_array[3]: 1.000000all_array[4]: 1.000000all_array[5]: 1.000000all_array[6]: 2.000000all_array[7]: 2.000000all_array[8]: 2.000000all_array[9]: 3.000000all_array[10]: 3.000000all_array[11]: 3.000000all_array[12]: 4.000000all_array[13]: 4.000000all_array[14]: 4.000000all_array[15]: 5.000000all_array[16]: 5.000000all_array[17]: 5.000000all_array[18]: 6.000000all_array[19]: 6.000000all_array[20]: 6.000000all_array[21]: 7.000000all_array[22]: 7.000000all_array[23]: 7.000000
```

### 一个计算平均数的示例 

```
#include <stdio.h>#include <stdlib.h>#include <mpi.h>#include <assert.h>float compute_avg(float *array, int num_elements){    float sum = 0.f;    int i;    for (i = 0; i < num_elements; i++)    {        sum += array[i];    }    return sum / num_elements;}int main(int argc, char **argv){    if (argc != 2)    {        fprintf(stderr, "Usage: scatter num_ele_per_node\n");        exit(1);    }    int num_ele_per_node = atoi(argv[1]);    MPI_Init(NULL, NULL);    int world_rank;    int world_size;    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);    MPI_Comm_size(MPI_COMM_WORLD, &world_size);    float *array = NULL;    if (world_rank == 0)    {        array = (float *)malloc(sizeof(float) * world_size * num_ele_per_node);        for (int i = 0; i < world_size * num_ele_per_node; i++)        {            array[i] = i;        }    }    float *sub_array = (float *)malloc(sizeof(float) * num_ele_per_node);    MPI_Scatter(array, num_ele_per_node, MPI_FLOAT, sub_array, num_ele_per_node, MPI_FLOAT, 0, MPI_COMM_WORLD);    for (int i = 0; i < num_ele_per_node; i++)        printf("proc %d sub_array[%d]: %f\n", world_rank, i, sub_array[i]);    float sub_avg = compute_avg(sub_array, num_ele_per_node);    MPI_Barrier(MPI_COMM_WORLD);    printf("proc %d sub_average: %f\n",world_rank,sub_avg);    float *sub_avgs = NULL;    if (world_rank == 0)    {        sub_avgs = (float *)malloc(sizeof(float) * world_size);        assert(sub_avgs != NULL);    }    MPI_Gather(&sub_avg, 1, MPI_FLOAT, sub_avgs, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);    if (world_rank == 0)    {        float avg = compute_avg(sub_avgs, world_size);        printf("Avg of all elements is %f\n", avg);        free(array);    }    free(sub_array);    MPI_Barrier(MPI_COMM_WORLD);    MPI_Finalize();}
```

```
zartbot@mpi1:~/mpi/study/collective$ mpicc avg.c -o avgzartbot@mpi1:~/mpi/study/collective$ mpiexec -np 8 ./avg 3proc 0 sub_array[0]: 0.000000proc 0 sub_array[1]: 1.000000proc 0 sub_array[2]: 2.000000proc 1 sub_array[0]: 3.000000proc 1 sub_array[1]: 4.000000proc 1 sub_array[2]: 5.000000proc 2 sub_array[0]: 6.000000proc 2 sub_array[1]: 7.000000proc 2 sub_array[2]: 8.000000proc 3 sub_array[0]: 9.000000proc 3 sub_array[1]: 10.000000proc 3 sub_array[2]: 11.000000proc 4 sub_array[0]: 12.000000proc 4 sub_array[1]: 13.000000proc 4 sub_array[2]: 14.000000proc 5 sub_array[0]: 15.000000proc 5 sub_array[1]: 16.000000proc 5 sub_array[2]: 17.000000proc 6 sub_array[0]: 18.000000proc 6 sub_array[1]: 19.000000proc 6 sub_array[2]: 20.000000proc 7 sub_array[0]: 21.000000proc 7 sub_array[1]: 22.000000proc 7 sub_array[2]: 23.000000proc 0 sub_average: 1.000000proc 1 sub_average: 4.000000proc 2 sub_average: 7.000000proc 3 sub_average: 10.000000proc 4 sub_average: 13.000000proc 5 sub_average: 16.000000proc 7 sub_average: 22.000000proc 6 sub_average: 19.000000Avg of all elements is 11.500000zartbot@mpi1:~/mpi/study/collective$
```

## MPI_Allgather

等于MPI_Gather后再加一个Bcast

![图片](assets/22e52da32ccf.png)

例如我们稍微修改一下前述代码

```
    MPI_Barrier(MPI_COMM_WORLD);    printf("proc %d sub_average: %f\n", world_rank, sub_avg);    float *sub_avgs = (float *)malloc(sizeof(float) * world_size);    MPI_Allgather(&sub_avg, 1, MPI_FLOAT, sub_avgs, 1, MPI_FLOAT, MPI_COMM_WORLD);    float avg = compute_avg(sub_avgs, world_size);    printf("Proc %d Avg of all elements is %f\n", world_rank,avg);    if (world_rank == 0)    {        free(array);    }
```

执行结果

```
Proc 7 Avg of all elements is 11.500000Proc 1 Avg of all elements is 11.500000Proc 4 Avg of all elements is 11.500000Proc 2 Avg of all elements is 11.500000Proc 0 Avg of all elements is 11.500000Proc 5 Avg of all elements is 11.500000Proc 3 Avg of all elements is 11.500000Proc 6 Avg of all elements is 11.500000
```