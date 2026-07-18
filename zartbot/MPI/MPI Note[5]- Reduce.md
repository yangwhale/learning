# MPI Note[5]: Reduce

> 作者: zartbot  
> 日期: 2021年4月18日 11:03  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485721&idx=2&sn=d98c28006693b8db79141043d22af39c&chksm=f99619dbcee190cd7d323f1d37ee1cc68bd67222be5852cb6bc67ea05fb9a587622d46732bb5#rd

---

### MPI_Reduce 

Reduce 是函数式编程中的一个非常经典的概念， 主要是加载一个可交换的二元算子进行规约操作。最直观的就是分布式算如下两个东西：

由于是可交换的二元算子，因此谁先做谁后做都无所谓，那么就有了分布式的处理方法，每个节点自己做一份，然后最终来聚合:

![图片](assets/8b4814b1d0a4.png)

当然也可以按照向量的方式同时处理多个值

![图片](assets/eb5e742f3804.png)

我们注意到这个可交换的约束， 因此通常支持的算符通常如下:

`MPI_MAX` - 返回最大元素。

`MPI_MIN` - 返回最小元素。

`MPI_SUM` - 对元素求和。

`MPI_PROD` - 将所有元素相乘。

`MPI_LAND` - 对元素执行逻辑与运算。

`MPI_LOR` - 对元素执行逻辑或运算。

`MPI_BAND` - 对元素的各个位按位与执行。

`MPI_BOR` - 对元素的位执行按位或运算。

`MPI_MAXLOC` - 返回最大值和所在的进程的秩。

`MPI_MINLOC` - 返回最小值和所在的进程的秩。

```
#include <stdio.h>#include <stdlib.h>#include <mpi.h>int main(int argc, char **argv){    if (argc != 2)    {        fprintf(stderr, "Usage: scatter num_ele_per_node\n");        exit(1);    }    int num_ele_per_node = atoi(argv[1]);    MPI_Init(NULL, NULL);    int world_rank;    int world_size;    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);    MPI_Comm_size(MPI_COMM_WORLD, &world_size);    //准备本地数据    float *array = (float *)malloc(sizeof(float) * num_ele_per_node);    for (int i = 0; i < num_ele_per_node; i++)    {        array[i] = world_rank + i;    }    float *global_sum = (float *)malloc(sizeof(float) * num_ele_per_node);    MPI_Reduce(array, global_sum, num_ele_per_node, MPI_FLOAT,MPI_SUM, 0, MPI_COMM_WORLD);    MPI_Barrier(MPI_COMM_WORLD);    if (world_rank == 0)    {        for (int i = 0; i < num_ele_per_node; i++)            printf("global_sum[%d]: %f,  avg: %f\n", i, global_sum[i], global_sum[i]/world_size);          }    free(global_sum);    free(array);    MPI_Barrier(MPI_COMM_WORLD);    MPI_Finalize();}
```

结果

```
kevin@netdev:~/Desktop/mpi/05_reduce$ mpirun -np 16 ./reduce 10global_sum[0]: 120.000000,  avg: 7.500000global_sum[1]: 136.000000,  avg: 8.500000global_sum[2]: 152.000000,  avg: 9.500000global_sum[3]: 168.000000,  avg: 10.500000global_sum[4]: 184.000000,  avg: 11.500000global_sum[5]: 200.000000,  avg: 12.500000global_sum[6]: 216.000000,  avg: 13.500000global_sum[7]: 232.000000,  avg: 14.500000global_sum[8]: 248.000000,  avg: 15.500000global_sum[9]: 264.000000,  avg: 16.500000
```

### MPI_Allreduce 

估计这个词因为很多分布式机器学习模型参数同步而更加容易的被人熟知。本质上就是Reduce的基础上加了一个MPI_Bcast

![图片](assets/2ea75c06d3ae.png)

```
MPI_Allreduce(    void* send_data,    void* recv_data,    int count,    MPI_Datatype datatype,    MPI_Op op,    MPI_Comm communicator)
```

例如我们来写一个计算标准差的程序

```
#include <stdio.h>#include <stdlib.h>#include <mpi.h>#include <math.h>int main(int argc, char **argv){    if (argc != 2)    {        fprintf(stderr, "Usage: scatter num_ele_per_node\n");        exit(1);    }    int num_ele_per_node = atoi(argv[1]);    MPI_Init(NULL, NULL);    int world_rank;    int world_size;    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);    MPI_Comm_size(MPI_COMM_WORLD, &world_size);    //prepare local data    float *array = (float *)malloc(sizeof(float) * num_ele_per_node);    for (int i = 0; i < num_ele_per_node; i++)    {        array[i] = world_rank + i * world_rank;    }    float *global_sum = (float *)malloc(sizeof(float) * num_ele_per_node);    MPI_Allreduce(array, global_sum, num_ele_per_node, MPI_FLOAT,MPI_SUM, MPI_COMM_WORLD);        float *global_mean = (float *)malloc(sizeof(float) * num_ele_per_node);    for (int i = 0; i < num_ele_per_node; i++)        global_mean[i] = global_sum[i]/world_size;    float *local_var = (float *)malloc(sizeof(float) * num_ele_per_node);    float *global_var = (float *)malloc(sizeof(float) * num_ele_per_node);    for (int i = 0; i < num_ele_per_node; i++)    {        local_var[i] = (array[i] - global_mean[i]) * (array[i] - global_mean[i]);    }    MPI_Reduce(local_var, global_var, num_ele_per_node, MPI_FLOAT,MPI_SUM, 0, MPI_COMM_WORLD);    if (world_rank == 0) {    for (int i = 0; i < num_ele_per_node; i++)       printf("Stddev[%d]: %f \n",i , sqrt(global_var[i]/world_size));            }            free(local_var);    free(global_var);    free(global_sum);    free(global_mean);    free(array);    MPI_Barrier(MPI_COMM_WORLD);    MPI_Finalize();}
```

编译和运行

```
kevin@netdev:~/Desktop/mpi/05_reduce$ mpicc std.c -o std -lmkevin@netdev:~/Desktop/mpi/05_reduce$ mpirun -np 16 ./std 6Stddev[0]: 4.609772 Stddev[1]: 9.219544 Stddev[2]: 13.829317 Stddev[3]: 18.439089 Stddev[4]: 23.048861 Stddev[5]: 27.658633 
```