# 基于eRDMA实测DeepSeek开源的3FS

> 作者: zartbot  
> 日期: 2025年3月1日 07:06  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493320&idx=1&sn=70f97436be6617d75940d8bee66cb1df&chksm=f995f60acee27f1c7bbbe49837f4d280ec8ecb68a1036560c51105308c186eeaf6f0d581b79b#rd

---

`本文仅代表个人观点, 与作者任职的机构无关.`

DeepSeek昨天开源了3FS分布式文件系统, 通过180个存储节点提供了 6.6TiB/s的存储性能, 全面支持大模型的训练和推理的KVCache转存以及向量数据库等能力, 每个客户端节点支持40+GB/s峰值吞吐用于KVCache查找.

发布后, 我们在阿里云ECS上进行了快速的复现, 并进行了性能测试, ECS在第八代实例中全地域全可用区部署了高性能的eRDMA通信能力, 解决了RDMA超大规模组网的问题, 并且ECS可选的存储介质有： ESSD、EED、本地盘等多种类型. 

值得一提的是, 在RDMA大规模组网时通常需要设计基于多路径转发的拥塞控制协议, 例如AWS SRD和UEC, 但是这些协议为了应对多路径转发时的乱序处理, 均不支持标准的RDMA Reliable Connection传输, 因此在适配3FS时会有大量的工作, 而eRDMA实现了高性能多路径转发及拥塞控制,并且完全兼容标准RDMA Reliable Connection传输, 无需修改任何3FS代码就可以直接运行.

为了对标3FS的官方部署指南, 我们这次测试中选择了本地盘实例构建了5个存储节点, 并且通过5个client进行了测试, 经过测试所有节点都能够打满实例产品规格的带宽(单机100Gbps).

![图片](assets/db43091ad50d.png)

ECS 9代实例将普遍标配400Gbps CIPU 2.0, 带宽和DeepSeek 3FS线下部署规格一致, 后续可以根据用户的需求提供新的实例规格满足业务需求.

对于缺少RDMA和相关存储测试环境的研究者和开源生态的贡献者, 可以通过如下文档在阿里云上基于eRDMA构建3FS并进行后续的测试和开发. 后续我们将对3fs进行更多的分析.

本文结构如下

```
1. DeepSeek 3FS分布式存储概述2. 安装和编译3fs2.1 构建编译环境2.2 编译3fs2.3 制作镜像3. 部署3FS3.1 安装ClickHouse和FoundationDB3.2 配置监控服务3.3 配置Admin Client3.4 配置Mgmtd Service3.5 配置Meta Service3.6 配置Storage Service3.7 配置3FS3.8 配置FUSE Client4. 性能测试
```

## 1. DeepSeek 3FS分布式存储概述

对3FS的关注大概是在2019年幻方有一篇文章介绍3FS时就在关注它, 当时只有一个record格式的git[1]. 作为量化交易的同行, 我在2014年搭建自己的私募量化平台时也在做一些分布式内存数据库的实现, 主要用途就是模型需要快速的从大量tick数据里抓取数据, 另外一些回测框架也需要极高的I/O处理能力, 类似于今天开源的另一个小项目smallpond[2]:

```
df = sp.partial_sql("SELECT ticker, min(price), max(price) FROM {0} GROUP BY ticker", df)df.write_parquet("output/")print(df.to_pandas())
```

正是这些原来在幻方量化使用的高性能分布式文件系统, 这一次用在了DeepSeek大模型的训练和推理上.下图展示了在 3FS 集群上进行的读压力测试的吞吐量。该集群由 180 个存储节点组成，每个节点配备 2×200Gbps InfiniBand 网卡和十六块 14TiB NVMe SSD。大约 500+个客户端节点被用于读压力测试，每个客户端节点配置了 1x200Gbps InfiniBand 网卡。最终的累计读吞吐量达到约 6.6 TiB/s，包括来自训练作业的背景流量。

![图片](assets/2fe0374ee853.jpg)

它采用了基于CRAQ的链式replication机制实现了数据的强一致性, 使应用能够以本地无关的方式访问分布在数百个服务器上的数千个SSD的存储资源. 然后还集成了用于LLM推理优化的KVCache服务, 客户端的峰值吞吐高达40GB/s

![图片](assets/b9dd275fa4b3.jpg)

详细的3FS架构和实现分析我们将在后续的文章中进行分析, 这一篇主要讲解如何基于云服务和eRDMA安装部署并进行性能测试.

## 2. 安装和编译3fs

首先我们在阿里云上申请一个`ecs.g8i.4xlarge`实例作为编译环境使用. 注意在创建实例的时候,选择`unbuntu 22.04`, 并勾选eRDMA驱动安装

![图片](assets/a53695d2b56b.png)

同时在弹性网卡中勾选`弹性RDMA接口`

![图片](assets/a65e2a174803.png)

### 2.1 构建编译环境

首先安装编译需要的package

```
# for Ubuntu 22.04.apt install cmake libuv1-dev liblz4-dev liblzma-dev libdouble-conversion-dev libprocps-dev libdwarf-dev libunwind-dev \  libaio-dev libgflags-dev libgoogle-glog-dev libgtest-dev libgmock-dev clang-format-14 clang-14 clang-tidy-14 lld-14 \  libgoogle-perftools-dev google-perftools libssl-dev ccache gcc-12 g++-12 libboost-all-dev
```

然后安装libfuse, 需要注意使用fuse3.16以上的版本

```
wget https://github.com/libfuse/libfuse/releases/download/fuse-3.16.1/fuse-3.16.1.tar.gztar vzxf fuse-3.16.1.tar.gzcd fuse-3.16.1/mkdir build; cd buildapt install mesonmeson setup ..ninja ; ninja install
```

安装rust工具链

```
#rust toolchainscurl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

安装foundation db

```
Download at https://github.com/apple/foundationdb/releases/tag/7.3.63wget https://github.com/apple/foundationdb/releases/download/7.3.63/foundationdb-clients_7.3.63-1_amd64.debwget https://github.com/apple/foundationdb/releases/download/7.3.63/foundationdb-server_7.3.63-1_amd64.debdpkg -i foundationdb-clients_7.3.63-1_amd64.debdpkg -i foundationdb-server_7.3.63-1_amd64.deb
```

### 2.2 编译3fs

按照如下方式下载和编译3fs

```
git clone https://github.com/deepseek-ai/3fscd 3fsgit submodule update --init --recursive./patches/apply.shcmake -S . -B build -DCMAKE_CXX_COMPILER=clang++-14 -DCMAKE_C_COMPILER=clang-14 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ONcmake --build build -j 32
```

检查编译输出的binary

```
root@3fs-1:~/3fs# ls -lrt build/bin/total 2289660-rwxr-xr-x 1 root root 148429888 Feb 28 17:41 hf3fs-admin-rwxr-xr-x 1 root root 105192704 Feb 28 17:41 monitor_collector_main-rwxr-xr-x 1 root root 178870448 Feb 28 17:42 mgmtd_main-rwxr-xr-x 1 root root 172303184 Feb 28 17:47 migration_main-rwxr-xr-x 1 root root 363821208 Feb 28 17:47 admin_cli-rwxr-xr-x 1 root root 174688488 Feb 28 17:47 simple_example_main-rwxr-xr-x 1 root root 284427704 Feb 28 17:48 meta_main-rwxr-xr-x 1 root root 395983072 Feb 28 17:48 storage_bench-rwxr-xr-x 1 root root 311693016 Feb 28 17:48 storage_main-rwxr-xr-x 1 root root 209211768 Feb 28 17:48 hf3fs_fuse_main
```

### 2.3 制作镜像

编译完成后对该台机器制作一个镜像用于后续部署

![图片](assets/8f827d7314de.png)

## 3. 部署3FS

我们采用官方文档推荐的方式, 在阿里云上创建1个`ecs.g8i.48xlarge`作为meta服务器和5个`ecs.i4.32xlarge`实例作为存储服务器.

Node

实例类型

IP

Memory

SSD

RDMA

meta

ecs.g8i.48xlarge

10.99.0.1

256GB

-

eRDMA

storage1

ecs.i4.32xlarge

10.99.0.2

1024GB

4TB × 8

eRDMA

storage2

ecs.i4.32xlarge

10.99.0.3

1024GB

4TB × 8

eRDMA

storage3

ecs.i4.32xlarge

10.99.0.4

1024GB

4TB × 8

eRDMA

storage4

ecs.i4.32xlarge

10.99.0.5

1024GB

4TB × 8

eRDMA

storage5

ecs.i4.32xlarge

10.99.0.6

1024GB

4TB × 8

eRDMA

fuseclient1

ecs.ebmg8i.48xlarge

10.99.0.101

1024GB

-

eRDMA

fuseclient2

ecs.ebmg8i.48xlarge

10.99.0.102

1024GB

-

eRDMA

fuseclient3

ecs.ebmg8i.48xlarge

10.99.0.103

1024GB

-

eRDMA

fuseclient4

ecs.ebmg8i.48xlarge

10.99.0.104

1024GB

-

eRDMA

fuseclient5

ecs.ebmg8i.48xlarge

10.99.0.105

1024GB

-

eRDMA

启动后, 将所有的eRDMA模式调成compatmode=1

```
rmmod erdmamodprobe erdma compat_mode=1
```

修改配置文件中的max_sge

```
cd ~/3fs/configssed -i 's/max_sge = 16/max_sge = 1/g' `grep -rl max_sge`
```

另外由于3FS使用了mellanox网卡的ibdev2netdev,在执行3fs命令时会调用, 因此我们在eRDMA环境,我们需要构造一个命令输出.采用如下方式添加脚本

```
vim  /usr/sbin/ibdev2netdev 添加如下内容#!/bin/bashecho "erdma_0 port 1 ==> eth0 (Up)"
```

保存退出后, 修改为可执行

```
chmod +x /usr/sbin/ibdev2netdev
```

然后将meta对应的ip填入每个节点的`/etc/hosts`

```
vim /etc/hosts#添加10.99.0.1 meta
```

每个节点的服务和相应的配置文件和官方建议相同,如下所示:

Service

Binary

Config files

NodeID

Node

monitor

monitor_collector_main
monitor_collector_main.toml
-

meta

admin_cli

admin_cli
admin_cli.toml

fdb.cluster

-

meta
storage1
storage2
storage3
storage4
storage5

mgmtd

mgmtd_main
mgmtd_main_launcher.toml
mgmtd_main.toml
mgmtd_main_app.toml

fdb.cluster

1

meta

meta

meta_main
meta_main_launcher.toml
meta_main.toml
meta_main_app.toml

fdb.cluster

100

meta

storage

storage_main
storage_main_launcher.toml
storage_main.toml
storage_main_app.toml
10001~10005

storage1
storage2
storage3
storage4
storage5

client

hf3fs_fuse_main
hf3fs_fuse_main_launcher.toml
hf3fs_fuse_main.toml
-

meta

### 3.1 安装ClickHouse和FoundationDB

由于复用了编译环境的镜像已经安装了FoundationDB, 因此仅需在`meta`节点安装ClickHouse

安装clickhouse, 可以参考`https://clickhouse.com/docs/install`

```
sudo apt-get install -y apt-transport-https ca-certificates curl gnupgcurl -fsSL 'https://packages.clickhouse.com/rpm/lts/repodata/repomd.xml.key' | sudo gpg --dearmor -o /usr/share/keyrings/clickhouse-keyring.gpgARCH=$(dpkg --print-architecture)echo "deb [signed-by=/usr/share/keyrings/clickhouse-keyring.gpg arch=${ARCH}] https://packages.clickhouse.com/deb stable main" | sudo tee /etc/apt/sources.list.d/clickhouse.listsudo apt-get updatesudo apt-get install -y clickhouse-server clickhouse-client#在安装的时候会要求输入密码, 此时我们统一输入`eRDMA123!!`
```

使用如下方式开启clickhouse服务

```
 sudo clickhouse start  
```

然后使用安装时的密码验证登陆

```
root@3fs-meta:~# clickhouse-client --password 'eRDMA123!!'ClickHouse client version 25.2.1.3085 (official build).Connecting to localhost:9000 as user default.Connected to ClickHouse server version 25.2.1.Warnings: * Delay accounting is not enabled, OSIOWaitMicroseconds will not be gathered. You can enable it using `echo 1 > /proc/sys/kernel/task_delayacct` or by using sysctl.3fs-meta :)
```

然后退出, 并采用如下命令创建Metric table

```
clickhouse-client --password 'eRDMA123!!' -n  < ~/3fs/deploy/sql/3fs-monitor.sql
```

### 3.2 配置监控服务

仅在`meta`节点配置安装`monitor_collector`服务.

```
mkdir -p /opt/3fs/{bin,etc}mkdir -p /var/log/3fscp ~/3fs/build/bin/monitor_collector_main /opt/3fs/bincp ~/3fs/configs/monitor_collector_main.toml /opt/3fs/etc
```

修改`monitor_collector_main.toml`如下所示

```
vim /opt/3fs/etc/monitor_collector_main.toml#最后一段修改为Host IP, 预配置密码, 用户名默认为default, 端口号默认为9000[server.monitor_collector.reporter.clickhouse]db = '3fs'host = '127.0.0.1'passwd = 'eRDMA123!!'port = '9000'user = 'default'
```

启动monitor_collector服务如下

```
cp ~/3fs/deploy/systemd/monitor_collector_main.service /usr/lib/systemd/systemsystemctl start monitor_collector_main
```

检查服务状态

```
root@3fs-meta:/opt/3fs/etc# systemctl status monitor_collector_main● monitor_collector_main.service - monitor_collector_main Server     Loaded: loaded (/lib/systemd/system/monitor_collector_main.service; disabled; vendor preset: enabled)     Active: active (running) since Fri 2025-02-28 21:09:06 CST; 15s ago   Main PID: 14401 (monitor_collect)      Tasks: 58 (limit: 629145)     Memory: 258.4M        CPU: 113ms     CGroup: /system.slice/monitor_collector_main.service             └─14401 /opt/3fs/bin/monitor_collector_main --cfg /opt/3fs/etc/monitor_collector_main.toml
```

### 3.3 配置Admin Client

在`所有`节点安装admin_cli

```
mkdir -p /opt/3fs/{bin,etc}rsync -avz meta:~/3fs/build/bin/admin_cli /opt/3fs/binrsync -avz meta:~/3fs/configs/admin_cli.toml /opt/3fs/etcrsync -avz meta:/etc/foundationdb/fdb.cluster /opt/3fs/etc
```

更新`admin_cli.toml`

```
vim /opt/3fs/etc/admin_cli.toml##更新如下内容cluster_id = "stage"[fdb]clusterFile = '/opt/3fs/etc/fdb.cluster'
```

admin_cli的使用帮助文档可以输入

```
root@3fs-meta:/opt/3fs/etc# /opt/3fs/bin/admin_cli -cfg /opt/3fs/etc/admin_cli.toml helpbench                          Usage: bench [--rank VAR] [--timeout VAR] [--coroutines VAR] [--seconds VAR] [--remove] pathcd                             Usage: cd [-L] [--inode] pathchecksum                       Usage: checksum [--list] [--batch VAR] [--md5] [--fillZero] [--output VAR] pathcreate                         Usage: create [--perm VAR] [--chain-table-id VAR] [--chain-table-ver VAR] [--chain-list VAR] [--chunk-size VAR] [--stripe-size VAR] pathcreate-range                   Usage: create-range [--concurrency VAR] prefix inclusive_start exclusive_endcreate-target                  Usage: create-target --node-id VAR --disk-index VAR --target-id VAR --chain-id VAR [--add-chunk-size] [--chunk-size VAR...] [--use-new-chunk-engine]create-targets                 Usage: create-targets --node-id VAR [--disk-index VAR...] [--allow-existing-target] [--add-chunk-size] [--use-new-chunk-engine]current-user                   Usage: current-userdecode-user-token              Usage: decode-user-token tokendrop-user-cache                Usage: drop-user-cache [--uid VAR] [--all]dump-chain-table               Usage: dump-chain-table [--version VAR] tableId csv-file-pathdump-chains                    Usage: dump-chains csv-file-path-prefixdump-chunkmeta                 Usage: dump-chunkmeta [--chain-ids VAR...] [--chunkmeta-dir VAR] [--parquet-format] [--only-head] [--parallel VAR]dump-dentries                  Usage: dump-dentries [--num-dentries-perfile VAR] [--fdb-cluster-file VAR] [--dentry-dir VAR] [--threads VAR]dump-inodes                    Usage: dump-inodes [--num-inodes-perfile VAR] [--fdb-cluster-file VAR] [--inode-dir VAR] [--parquet-format] [--all-inodes] [--threads VAR].....
```

### 3.4 配置Mgmtd Service

`mgmtd`仅在`meta`节点安装.首先拷贝文件

```
cp ~/3fs/build/bin/mgmtd_main /opt/3fs/bincp ~/3fs/configs/{mgmtd_main.toml,mgmtd_main_launcher.toml,mgmtd_main_app.toml} /opt/3fs/etc
```

修改配置文件, 将mgmtd配置文件`mgmtd_main_app.toml`定义`node_id =1`

```
vim /opt/3fs/etc/mgmtd_main_app.toml##修改node_id = 1
```

修改`/opt/3fs/etc/mgmtd_main_launcher.toml`中的cluster_id和clusterFile

```
cluster_id = "stage"[fdb]clusterFile = '/opt/3fs/etc/fdb.cluster'
```

修改`mgmtd_main.toml`将remoteip修改为meta服务器地址

```
[common.monitor.reporters.monitor_collector]remote_ip = "10.99.0.1:10000"
```

配置完成后, 初始化集群

```
/opt/3fs/bin/admin_cli -cfg /opt/3fs/etc/admin_cli.toml "init-cluster --mgmtd /opt/3fs/etc/mgmtd_main.toml 1 1048576 16"Init filesystem, root directory layout: chain table ChainTableId(1), chunksize 1048576, stripesize 16Init config for MGMTD version 1
```

其中参数1代表chainTable ID, 1048576代表chunksize, 16代表file strip size.然后启动服务并验证

```
cp ~/3fs/deploy/systemd/mgmtd_main.service /usr/lib/systemd/systemsystemctl start mgmtd_mainroot@3fs-meta:/opt/3fs/etc# systemctl status mgmtd_main● mgmtd_main.service - mgmtd_main Server     Loaded: loaded (/lib/systemd/system/mgmtd_main.service; disabled; vendor preset: enabled)     Active: active (running) since Fri 2025-02-28 21:33:46 CST; 27s ago   Main PID: 16375 (mgmtd_main)      Tasks: 36 (limit: 629145)     Memory: 192.7M        CPU: 123ms     CGroup: /system.slice/mgmtd_main.service             └─16375 /opt/3fs/bin/mgmtd_main --launcher_cfg /opt/3fs/etc/mgmtd_main_launcher.toml --app-cfg /opt/3fs/etc/mgmtd_main_app.toml
```

然后采用如下命令检查节点:

```
root@3fs-meta:~# /opt/3fs/bin/admin_cli -cfg /opt/3fs/etc/admin_cli.toml --config.mgmtd_client.mgmtd_server_addresses '["RDMA://10.99.0.1:8000"]' "list-nodes"Id  Type   Status         Hostname  Pid    Tags  LastHeartbeatTime  ConfigVersion  ReleaseVersion1   MGMTD  PRIMARY_MGMTD  3fs-meta  17434  []    N/A                1(UPTODATE)    250228-dev-1-999999-824fbf5c
```

### 3.5 配置Meta Service

该服务仅在`meta`服务器安装, 拷贝文件如下所示

```
cp ~/3fs/build/bin/meta_main /opt/3fs/bincp ~/3fs/configs/{meta_main_launcher.toml,meta_main.toml,meta_main_app.toml} /opt/3fs/etc
```

修改`meta_main_app.toml`中的node_id = 100. 修改`meta_main_launcher.toml`中的 cluster_id, clusterFile

```
cluster_id = "stage"[mgmtd_client]mgmtd_server_addresses = ["RDMA://10.99.0.1:8000"]
```

修改`meta_main.toml`如下:

```
[server.mgmtd_client]mgmtd_server_addresses = ["RDMA://10.99.0.1:8000"][common.monitor.reporters.monitor_collector]remote_ip = "10.99.0.1:10000"[server.fdb]clusterFile = '/opt/3fs/etc/fdb.cluster'
```

更新配置如下

```
/opt/3fs/bin/admin_cli -cfg /opt/3fs/etc/admin_cli.toml --config.mgmtd_client.mgmtd_server_addresses '["RDMA://10.99.0.1:8000"]' "set-config --type META --file /opt/3fs/etc/meta_main.toml"
```

启动服务

```
cp ~/3fs/deploy/systemd/meta_main.service /usr/lib/systemd/systemsystemctl start meta_mainroot@3fs-meta:~# systemctl status meta_main● meta_main.service - meta_main Server     Loaded: loaded (/lib/systemd/system/meta_main.service; disabled; vendor preset: enabled)     Active: active (running) since Fri 2025-02-28 22:37:58 CST; 7s ago   Main PID: 17709 (meta_main)      Tasks: 64 (limit: 629145)     Memory: 408.9M        CPU: 250ms     CGroup: /system.slice/meta_main.service             └─17709 /opt/3fs/bin/meta_main --launcher_cfg /opt/3fs/etc/meta_main_launcher.toml --app-cfg /opt/3fs/etc/meta_main_app.toml             
```

检查节点

```
root@3fs-meta:~# /opt/3fs/bin/admin_cli -cfg /opt/3fs/etc/admin_cli.toml --config.mgmtd_client.mgmtd_server_addresses '["RDMA://10.99.0.1:8000"]' "list-nodes"Id   Type   Status               Hostname  Pid    Tags  LastHeartbeatTime    ConfigVersion  ReleaseVersion1    MGMTD  PRIMARY_MGMTD        3fs-meta  17434  []    N/A                  1(UPTODATE)    250228-dev-1-999999-824fbf5c100  META   HEARTBEAT_CONNECTED  3fs-meta  17709  []    2025-02-28 22:38:28  1              250228-dev-1-999999-824fbf5c
```

### 3.6 配置Storage Service

在所有存储节点启用storage服务, 由于我们每个节点只有8块盘, 配置挂载如下:

```
mkdir -p /storage/data{0..7}mkdir -p /var/log/3fsfor i in {0..7};do mkfs.xfs -L data${i} /dev/nvme${i}n1;mount -o noatime,nodiratime -L data${i} /storage/data${i};donemkdir -p /storage/data{0..7}/3fsroot@3fs-storage001:~# df -kh | grep nvme/dev/nvme1n1    3.5T   25G  3.5T   1% /storage/data1/dev/nvme2n1    3.5T   25G  3.5T   1% /storage/data2/dev/nvme3n1    3.5T   25G  3.5T   1% /storage/data3/dev/nvme4n1    3.5T   25G  3.5T   1% /storage/data4/dev/nvme5n1    3.5T   25G  3.5T   1% /storage/data5/dev/nvme6n1    3.5T   25G  3.5T   1% /storage/data6/dev/nvme7n1    3.5T   25G  3.5T   1% /storage/data7/dev/nvme0n1    3.5T   25G  3.5T   1% /storage/data0
```

增加aio请求的最大数

```
sysctl -w fs.aio-max-nr=67108864
```

修改`meta`节点的原始配置文件`~/3fs/configs/storage_main_launcher.toml`中的clusterid和管理地址

```
 vim ~/3fs/configs/storage_main_launcher.toml  cluster_id = "stage"[mgmtd_client]mgmtd_server_addresses = ["RDMA://10.99.0.1:8000"]
```

修改`~/3fs/configs/storage_main.toml`中的IP地址和target path

```
vim ~/3fs/configs/storage_main.toml[server.mgmtd]mgmtd_server_address = ["RDMA://10.99.0.1:8000"][common.monitor.reporters.monitor_collector]remote_ip = "10.99.0.1:10000"[server.targets]target_paths = ["/storage/data0/3fs","/storage/data1/3fs","/storage/data2/3fs","/storage/data3/3fs","/storage/data4/3fs","/storage/data5/3fs","/storage/data6/3fs","/storage/data7/3fs"]
```

从meta节点拷贝执行文件和配置文件

```
rsync -avz meta:~/3fs/build/bin/storage_main /opt/3fs/binrsync -avz meta:~/3fs/configs/{storage_main_launcher.toml,storage_main.toml,storage_main_app.toml} /opt/3fs/etc
```

每个存储节点修改`/opt/3fs/etc/storage_main_app.toml`中的node_id, 五台机器分别为10001~10005

然后每个存储节点更新

```
/opt/3fs/bin/admin_cli -cfg /opt/3fs/etc/admin_cli.toml --config.mgmtd_client.mgmtd_server_addresses '["RDMA://10.99.0.1:8000"]' "set-config --type STORAGE --file /opt/3fs/etc/storage_main.toml"
```

最后启动并验证服务

```
rsync -avz meta:~/3fs/deploy/systemd/storage_main.service /usr/lib/systemd/systemsystemctl start storage_mainroot@3fs-storage001:/opt/3fs/etc# systemctl status storage_main● storage_main.service - storage_main Server     Loaded: loaded (/lib/systemd/system/storage_main.service; disabled; vendor preset: enabled)     Active: active (running) since Fri 2025-02-28 23:02:07 CST; 30s ago   Main PID: 7788 (storage_main)      Tasks: 242 (limit: 629145)     Memory: 9.5G        CPU: 10.017s     CGroup: /system.slice/storage_main.service     
```

检查系统节点:

```
root@3fs-storage001:~# /opt/3fs/bin/admin_cli -cfg /opt/3fs/etc/admin_cli.toml --config.mgmtd_client.mgmtd_server_addresses '["RDMA://10.99.0.1:8000"]' "list-nodes"/root/.profile: line 10: /.cargo/env: No such file or directoryId     Type     Status               Hostname        Pid    Tags  LastHeartbeatTime    ConfigVersion  ReleaseVersion1      MGMTD    PRIMARY_MGMTD        3fs-meta        17434  []    N/A                  1(UPTODATE)    250228-dev-1-999999-824fbf5c100    META     HEARTBEAT_CONNECTED  3fs-meta        17709  []    2025-02-28 23:03:19  2(UPTODATE)    250228-dev-1-999999-824fbf5c10001  STORAGE  HEARTBEAT_CONNECTED  3fs-storage001  7788   []    2025-02-28 23:03:20  5(UPTODATE)    250228-dev-1-999999-824fbf5c10002  STORAGE  HEARTBEAT_CONNECTED  3fs-storage002  9025   []    2025-02-28 23:03:22  5(UPTODATE)    250228-dev-1-999999-824fbf5c10003  STORAGE  HEARTBEAT_CONNECTED  3fs-storage003  6745   []    2025-02-28 23:03:20  5(UPTODATE)    250228-dev-1-999999-824fbf5c10004  STORAGE  HEARTBEAT_CONNECTED  3fs-storage004  7309   []    2025-02-28 23:03:21  5(UPTODATE)    250228-dev-1-999999-824fbf5c10005  STORAGE  HEARTBEAT_CONNECTED  3fs-storage005  6776   []    2025-02-28 23:03:19  5(UPTODATE)    250228-dev-1-999999-824fbf5c
```

### 3.7 配置3FS

创建管理员

```
root@3fs-meta:~/3fs/configs# /opt/3fs/bin/admin_cli -cfg /opt/3fs/etc/admin_cli.toml --config.mgmtd_client.mgmtd_server_addresses '["RDMA://10.99.0.1:8000"]' "user-add --root --admin 0 root"Uid                0Name               rootToken              AADDI7y+8QAUtUR+2wCeuDI5(Expired at N/A)IsRootUser         trueIsAdmin            trueGid                0SupplementaryGids
```

将token保存在`/opt/3fs/etc/token.txt`中.

然后创建chain Table, 首先安装python相关的依赖

```
pip3 install -r ~/3fs/deploy/data_placement/requirements.txt
```

然后执行data_placement计算命令

```
root@3fs-meta# python3 ~/3fs/deploy/data_placement/src/model/data_placement.py \   -ql -relax -type CR --num_nodes 5 --replication_factor 3 --min_targets_per_disk 6   2025-02-28 23:23:06.821 | INFO     | __main__:run:125 - solving model with appsi_highs #0: DataPlacementModel-v=5,b=10,r=6,k=3,λ=2,lb=1,ub=02025-02-28 23:23:06.821 | INFO     | __main__:build_model:182 - self.num_nodes=5 self.num_targets_per_disk=6 self.group_size=3 self.num_groups=10 self.qlinearize=True self.relax_lb=1 self.relax_ub=02025-02-28 23:23:06.821 | INFO     | __main__:build_model:192 - self.sum_recovery_traffic_per_failure=6 self.max_recovery_traffic_on_peer=22025-02-28 23:23:06.821 | INFO     | __main__:build_model:196 - self.all_targets_used=True self.balanced_peer_traffic=False2025-02-28 23:23:06.821 | INFO     | __main__:build_model:197 - self.num_targets_used=30 self.num_targets_total=302025-02-28 23:23:06.839 | INFO     | __main__:build_model:272 - lower bound imposed on peer traffic: self.relax_lb=1 self.qlinearize=True self.all_targets_used=TrueRunning HiGHS 1.8.0 (git hash: eda5cbe): Copyright (c) 2024 HiGHS under MIT licence terms         1       0         1 100.00%   inf             inf                  inf      132     16      7       299     0.0sSolving report  Status            Infeasible  Primal bound      inf  Dual bound        inf  Gap               inf  Solution status   -  Timing            0.02 (total)                    0.00 (presolve)                    0.00 (postsolve)  Nodes             1  LP iterations     299 (total)                    0 (strong br.)                    109 (separation)                    0 (heuristics)        Nodes      |    B&B Tree     |            Objective Bounds              |  Dynamic Constraints |       Work     Proc. InQueue |  Leaves   Expl. | BestBound       BestSol              Gap |   Cuts   InLp Confl. | LpIters     Time         0       0         0   0.00%   1               inf                  inf        0      0      0         0     0.0sObjective function is integral with scale 1Coefficient ranges:  Matrix [1e+00, 1e+00]  Cost   [0e+00, 0e+00]  Bound  [1e+00, 1e+00]  RHS    [1e+00, 6e+00]Presolving model335 rows, 150 cols, 1000 nonzeros  0s325 rows, 150 cols, 900 nonzeros  0sSolving MIP model with:   325 rows   150 cols (150 binary, 0 integer, 0 implied int., 0 continuous)   900 nonzeros         0       0         0   0.00%   1               inf                  inf        0      0      4       190     0.0s2025-02-28 23:23:06.879 | ERROR    | __main__:run:133 - cannot find solution for current params: infeasible:- Status: error  Termination condition: infeasible  Termination message: TerminationCondition.infeasible2025-02-28 23:23:06.879 | INFO     | __main__:run:125 - solving model with appsi_highs #1: DataPlacementModel-v=5,b=10,r=6,k=3,λ=2,lb=1,ub=12025-02-28 23:23:06.879 | INFO     | __main__:build_model:182 - self.num_nodes=5 self.num_targets_per_disk=6 self.group_size=3 self.num_groups=10 self.qlinearize=True self.relax_lb=1 self.relax_ub=12025-02-28 23:23:06.880 | INFO     | __main__:build_model:192 - self.sum_recovery_traffic_per_failure=6 self.max_recovery_traffic_on_peer=22025-02-28 23:23:06.880 | INFO     | __main__:build_model:196 - self.all_targets_used=True self.balanced_peer_traffic=False2025-02-28 23:23:06.880 | INFO     | __main__:build_model:197 - self.num_targets_used=30 self.num_targets_total=302025-02-28 23:23:06.882 | INFO     | __main__:build_model:272 - lower bound imposed on peer traffic: self.relax_lb=1 self.qlinearize=True self.all_targets_used=TrueRunning HiGHS 1.8.0 (git hash: eda5cbe): Copyright (c) 2024 HiGHS under MIT licence terms         1       0         1 100.00%   1               1                  0.00%       57      4      3       194     0.0sSolving report  Status            Optimal  Primal bound      1  Dual bound        1  Gap               0% (tolerance: 0.01%)  Solution status   feasible                    1 (objective)                    0 (bound viol.)                    0 (int. viol.)                    0 (row viol.)         0       0         0   0.00%   1               inf                  inf        0      0      3       181     0.0s         0       0         0   0.00%   1               inf                  inf        0      0      0         0     0.0sObjective function is integral with scale 1Coefficient ranges:  Matrix [1e+00, 1e+00]  Cost   [0e+00, 0e+00]  Bound  [1e+00, 1e+00]  RHS    [1e+00, 6e+00]Presolving model335 rows, 150 cols, 1000 nonzeros  0s325 rows, 150 cols, 900 nonzeros  0sSolving MIP model with:   325 rows   150 cols (150 binary, 0 integer, 0 implied int., 0 continuous)   900 nonzeros        Nodes      |    B&B Tree     |            Objective Bounds              |  Dynamic Constraints |       Work     Proc. InQueue |  Leaves   Expl. | BestBound       BestSol              Gap |   Cuts   InLp Confl. | LpIters     Time  Timing            0.01 (total)                    0.00 (presolve)                    0.00 (postsolve)  Nodes             1  LP iterations     194 (total)                    0 (strong br.)                    13 (separation)                    0 (heuristics)2025-02-28 23:23:06.906 | SUCCESS  | __main__:solve:165 - optimal solution:- Status: ok  Termination condition: optimal  Termination message: TerminationCondition.optimal2025-02-28 23:23:06.907 | DEBUG    | __main__:check_solution:322 - 1,2: 1.52025-02-28 23:23:06.907 | DEBUG    | __main__:check_solution:322 - 1,3: 1.52025-02-28 23:23:06.907 | DEBUG    | __main__:check_solution:322 - 1,4: 1.52025-02-28 23:23:06.907 | DEBUG    | __main__:check_solution:322 - 1,5: 1.52025-02-28 23:23:06.907 | DEBUG    | __main__:check_solution:322 - 2,1: 1.52025-02-28 23:23:06.907 | DEBUG    | __main__:check_solution:322 - 2,3: 1.52025-02-28 23:23:06.907 | DEBUG    | __main__:check_solution:322 - 2,4: 1.52025-02-28 23:23:06.907 | DEBUG    | __main__:check_solution:322 - 2,5: 1.52025-02-28 23:23:06.907 | DEBUG    | __main__:check_solution:322 - 3,1: 1.52025-02-28 23:23:06.907 | DEBUG    | __main__:check_solution:322 - 3,2: 1.52025-02-28 23:23:06.907 | DEBUG    | __main__:check_solution:322 - 3,4: 1.52025-02-28 23:23:06.907 | DEBUG    | __main__:check_solution:322 - 3,5: 1.52025-02-28 23:23:06.907 | DEBUG    | __main__:check_solution:322 - 4,1: 1.52025-02-28 23:23:06.907 | DEBUG    | __main__:check_solution:322 - 4,2: 1.52025-02-28 23:23:06.907 | DEBUG    | __main__:check_solution:322 - 4,3: 1.52025-02-28 23:23:06.907 | DEBUG    | __main__:check_solution:322 - 4,5: 1.52025-02-28 23:23:06.907 | DEBUG    | __main__:check_solution:322 - 5,1: 1.52025-02-28 23:23:06.907 | DEBUG    | __main__:check_solution:322 - 5,2: 1.52025-02-28 23:23:06.907 | DEBUG    | __main__:check_solution:322 - 5,3: 1.52025-02-28 23:23:06.907 | DEBUG    | __main__:check_solution:322 - 5,4: 1.52025-02-28 23:23:06.907 | INFO     | __main__:check_solution:331 - min_peer_traffic=1.5 max_peer_traffic=1.52025-02-28 23:23:06.907 | INFO     | __main__:check_solution:332 - total_traffic=30.0 max_total_traffic=302025-02-28 23:23:07.068 | SUCCESS  | __main__:run:148 - saved solution to: output/DataPlacementModel-v_5-b_10-r_6-k_3-λ_2-lb_1-ub_1
```

然后执行产生chainTable

```
python3 ~/3fs/deploy/data_placement/src/setup/gen_chain_table.py \   --chain_table_type CR --node_id_begin 10001 --node_id_end 10005 \   --num_disks_per_node 8 --num_targets_per_disk 6 \   --target_id_prefix 1 --chain_id_prefix 9 \   --incidence_matrix_path output/DataPlacementModel-v_5-b_10-r_6-k_3-λ_2-lb_1-ub_1/incidence_matrix.pickle
```

检查output目录是否产生了如下文件

```
root@3fs-meta:/opt/3fs# ls -lrt output-rw-r--r-- 1 root root   808 Feb 28 23:24 generated_chain_table.csv-rw-r--r-- 1 root root  3955 Feb 28 23:24 generated_chains.csv-rw-r--r-- 1 root root 27600 Feb 28 23:24 create_target_cmd.txt
```

创建storage target

```
/opt/3fs/bin/admin_cli --cfg /opt/3fs/etc/admin_cli.toml --config.mgmtd_client.mgmtd_server_addresses '["RDMA://10.99.0.1:8000"]' --config.user_info.token $(<"/opt/3fs/etc/token.txt") < output/create_target_cmd.txt
```

上传chains 和 chain table到mgmtd service

```
/opt/3fs/bin/admin_cli --cfg /opt/3fs/etc/admin_cli.toml --config.mgmtd_client.mgmtd_server_addresses '["RDMA://10.99.0.1:8000"]' --config.user_info.token $(<"/opt/3fs/etc/token.txt") "upload-chains output/generated_chains.csv"/opt/3fs/bin/admin_cli --cfg /opt/3fs/etc/admin_cli.toml --config.mgmtd_client.mgmtd_server_addresses '["RDMA://10.99.0.1:8000"]' --config.user_info.token $(<"/opt/3fs/etc/token.txt") "upload-chain-table --desc stage 1 output/generated_chain_table.csv"
```

检查是否上传成功

```
# /opt/3fs/bin/admin_cli -cfg /opt/3fs/etc/admin_cli.toml --config.mgmtd_client.mgmtd_server_addresses '["RDMA://10.99.0.1:8000"]' "list-chains"900800001  1             1             SERVING  []              101000100801(SERVING-UPTODATE)  101000200801(SERVING-UPTODATE)  101000400801(SERVING-UPTODATE)900800002  1             1             SERVING  []              101000200802(SERVING-UPTODATE)  101000300801(SERVING-UPTODATE)  101000500801(SERVING-UPTODATE)900800003  1             1             SERVING  []              101000100802(SERVING-UPTODATE)  101000200803(SERVING-UPTODATE)  101000300802(SERVING-UPTODATE)900800004  1             1             SERVING  []              101000100803(SERVING-UPTODATE)  101000200804(SERVING-UPTODATE)  101000500802(SERVING-UPTODATE)900800005  1             1             SERVING  []              101000100804(SERVING-UPTODATE)  101000400802(SERVING-UPTODATE)  101000500803(SERVING-UPTODATE)900800006  1             1             SERVING  []              101000200805(SERVING-UPTODATE)  101000300803(SERVING-UPTODATE)  101000400803(SERVING-UPTODATE)900800007  1             1             SERVING  []              101000200806(SERVING-UPTODATE)  101000400804(SERVING-UPTODATE)  101000500804(SERVING-UPTODATE)900800008  1             1             SERVING  []              101000100805(SERVING-UPTODATE)  101000300804(SERVING-UPTODATE)  101000400805(SERVING-UPTODATE)900800009  1             1             SERVING  []              101000300805(SERVING-UPTODATE)  101000400806(SERVING-UPTODATE)  101000500805(SERVING-UPTODATE)# /opt/3fs/bin/admin_cli -cfg /opt/3fs/etc/admin_cli.toml --config.mgmtd_client.mgmtd_server_addresses '["RDMA://10.99.0.1:8000"]' "list-chain-tables"ChainTableId  ChainTableVersion  ChainCount  ReplicaCount  Desc1             1                  80          3             stage
```

### 3.8 配置FUSE Client

在这个demo中我们采用在多个独立的节点部署FUSE Client的方式, 首先拷贝文件, 并创建mount点

```
cp ~/3fs/build/bin/hf3fs_fuse_main /opt/3fs/bincp ~/3fs/configs/{hf3fs_fuse_main_launcher.toml,hf3fs_fuse_main.toml,hf3fs_fuse_main_app.toml} /opt/3fs/etcmkdir -p /3fs/stage
```

修改`/opt/3fs/etc/hf3fs_fuse_main_launcher.toml`配置如下:

```
cluster_id = "stage"mountpoint = '/3fs/stage'token_file = '/opt/3fs/etc/token.txt'[mgmtd_client]mgmtd_server_addresses = ["RDMA://10.99.0.1:8000"]
```

修改`/opt/3fs/etc/hf3fs_fuse_main.toml`配置如下

```
[mgmtd]mgmtd_server_addresses = ["RDMA://10.99.0.1:8000"][common.monitor.reporters.monitor_collector]remote_ip = "10.99.0.1:10000"
```

更新Fuse client配置到mgmtd service

```
/opt/3fs/bin/admin_cli -cfg /opt/3fs/etc/admin_cli.toml --config.mgmtd_client.mgmtd_server_addresses '["RDMA://10.99.0.1:8000"]' "set-config --type FUSE --file /opt/3fs/etc/hf3fs_fuse_main.toml"
```

开启fuse client

```
cp ~/3fs/deploy/systemd/hf3fs_fuse_main.service /usr/lib/systemd/systemsystemctl start hf3fs_fuse_mainroot@3fs-client:/opt/3fs# systemctl status hf3fs_fuse_main● hf3fs_fuse_main.service - fuse_main Server     Loaded: loaded (/lib/systemd/system/hf3fs_fuse_main.service; disabled; vendor preset: enabled)     Active: active (running) since Fri 2025-02-28 23:38:18 CST; 5s ago   Main PID: 19841 (hf3fs_fuse_main)      Tasks: 49 (limit: 629145)     Memory: 318.9M        CPU: 250ms     CGroup: /system.slice/hf3fs_fuse_main.service             ├─19841 /opt/3fs/bin/hf3fs_fuse_main --launcher_cfg /opt/3fs/etc/hf3fs_fuse_main_launcher.toml             └─19903 fusermount3 --auto-unmount -- /3fs/stage             
```

检查是否mount

```
root@3fs-client:/opt/3fs# mount | grep '/3fs/stage'hf3fs.stage on /3fs/stage type fuse.hf3fs (rw,nosuid,nodev,relatime,user_id=0,group_id=0,default_permissions,allow_other,max_read=1048576)root@3fs-meta:/opt/3fs# df -khFilesystem      Size  Used Avail Use% Mounted ontmpfs           100G  1.9M  100G   1% /run/dev/nvme0n1p3  394G   28G  350G   8% /tmpfs           496G   16K  496G   1% /dev/shmtmpfs           5.0M     0  5.0M   0% /run/lock/dev/nvme0n1p2  197M  6.1M  191M   4% /boot/efitmpfs           100G  4.0K  100G   1% /run/user/0hf3fs.stage     140T  999G  139T   1% /3fs/stage
```

## 4. 性能测试

我们在5个fuse client上同时进行并发读取测试

```
fio -numjobs=128 -fallocate=none -iodepth=2 -ioengine=libaio -direct=1 -rw=read -bs=4M --group_reporting -size=100M -time_based -runtime=3000 -name=2depth_128file_4M_direct_read_bw -directory=/3fs/stagedepth_128file_4M_direct_read_bw: (groupid=0, jobs=128): err= 0: pid=11785: Sat Mar  1 13:08:54 2025  read: IOPS=2669, BW=10.4GiB/s (11.2GB/s)(6931GiB/664647msec)  ##带宽为11.2GiB/s已经打满实例规格速度    slat (usec): min=36, max=459933, avg=47946.24, stdev=11724.76    clat (usec): min=1303, max=459937, avg=47945.69, stdev=11728.42     lat (usec): min=1891, max=518800, avg=95892.19, stdev=16777.22    clat percentiles (msec):     |  1.00th=[   24],  5.00th=[   27], 10.00th=[   36], 20.00th=[   37],     | 30.00th=[   47], 40.00th=[   48], 50.00th=[   49], 60.00th=[   50],     | 70.00th=[   51], 80.00th=[   59], 90.00th=[   62], 95.00th=[   66],     | 99.00th=[   79], 99.50th=[   86], 99.90th=[   97], 99.95th=[  102],     | 99.99th=[  184]   bw (  MiB/s): min= 6192, max=13702, per=100.00%, avg=10681.29, stdev= 7.26, samples=170112   iops        : min= 1548, max= 3422, avg=2669.52, stdev= 1.81, samples=170112  lat (msec)   : 2=0.01%, 4=0.01%, 10=0.01%, 20=0.41%, 50=69.00%  lat (msec)   : 100=30.51%, 250=0.05%, 500=0.01%  cpu          : usr=0.00%, sys=0.18%, ctx=6960833, majf=0, minf=363857  IO depths    : 1=0.1%, 2=100.0%, 4=0.0%, 8=0.0%, 16=0.0%, 32=0.0%, >=64=0.0%     submit    : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.0%     complete  : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.0%     issued rwts: total=1774252,0,0,0 short=0,0,0,0 dropped=0,0,0,0     latency   : target=0, window=0, percentile=100.00%, depth=2
```

通过ECS管理控制台也可以看到已经打满带宽.

![图片](assets/ca0c733671db.png)

3FS还使用了clickhouse对运行数据进行统计分析, 可以登陆meta节点的查询

```
clickhouse-client --password 'eRDMA123!!'3fs-meta :) use 3fs3fs-meta :) select * from distributions where metricName=='storage_client.request_bw' AND host=='3fs-fuse' limit 10SELECT *FROM distributionsWHERE (metricName = 'storage_client.request_bw') AND (host = '3fs-fuse')LIMIT 10Query id: bae763a9-0c4c-413f-9103-a1c7fadaab6c    ┌───────────TIMESTAMP─┬─metricName────────────────┬─host─────┬─tag─┬─count─┬──────────────mean─┬────────────────min─┬────────────────max─┬───────────────p50─┬───────────────p90─┬────────────────p95─┬────────────────p99─┬─mount_name─┬─instance──┬─io─┬─uid─┬─method─┬─pod──────┬─thread─┬─statusCode─┐ 1. │ 2025-03-01 11:06:46 │ storage_client.request_bw │ 3fs-fuse │     │  8591 │ 613373090.8395855 │  216067587.0595508 │  1675041533.546326 │ 594001794.1159781 │ 917471066.1935523 │ 1118230720.8216615 │ 1353937621.2941322 │            │ batchRead │    │     │        │ 3fs-fuse │        │            │ 2. │ 2025-03-01 11:06:47 │ storage_client.request_bw │ 3fs-fuse │     │ 10592 │ 631580288.6128079 │ 169261662.63115415 │ 1558062407.1322436 │ 625856561.6014094 │ 929167946.9438102 │ 1101134592.7515178 │ 1319381573.1463842 │            │ batchRead │    │     │        │ 3fs-fuse │        │            │ 3. │ 2025-03-01 11:06:48 │ storage_client.request_bw │ 3fs-fuse │     │ 10627 │ 624043291.1181132 │  171476042.5183974 │ 1625699224.8062015 │ 620531561.6303563 │ 914151142.2446904 │  1070662224.208625 │ 1305868148.0882857 │            │ batchRead │    │     │        │ 3fs-fuse │        │            │ 4. │ 2025-03-01 11:06:49 │ storage_client.request_bw │ 3fs-fuse │     │ 10660 │  623494914.128004 │  186214881.9037471 │  1628223602.484472 │ 616674787.4039807 │ 914974468.4772394 │ 1089693011.0233178 │ 1281664891.6561015 │            │ batchRead │    │     │        │ 3fs-fuse │        │            │ 5. │ 2025-03-01 11:06:50 │ storage_client.request_bw │ 3fs-fuse │     │ 10627 │ 624230580.0179524 │  221218565.4008439 │ 1620673879.4435859 │ 618548223.0963331 │ 918910082.9235835 │ 1088660510.2403255 │ 1292470925.9244142 │            │ batchRead │    │     │        │ 3fs-fuse │        │            │ 6. │ 2025-03-01 11:06:51 │ storage_client.request_bw │ 3fs-fuse │     │ 10606 │ 632341527.2096547 │  185621525.9337936 │ 1605782542.1133232 │ 626439939.7497075 │ 928116483.5742279 │ 1114311679.0423079 │ 1318773172.0729723 │            │ batchRead │    │     │        │ 3fs-fuse │        │            │ 7. │ 2025-03-01 11:06:52 │ storage_client.request_bw │ 3fs-fuse │     │ 10591 │ 622737514.2896469 │ 176706437.47893494 │ 1596006088.2800608 │ 617361406.2508819 │ 910876455.9893316 │ 1076940139.2025425 │ 1297442965.0532806 │            │ batchRead │    │     │        │ 3fs-fuse │        │            │ 8. │ 2025-03-01 11:06:53 │ storage_client.request_bw │ 3fs-fuse │     │ 10635 │ 625666059.2437743 │ 188558892.28556016 │  1600879389.312977 │ 619899922.4441694 │ 919915373.7100124 │ 1081999833.3945801 │ 1277120376.2726321 │            │ batchRead │    │     │        │ 3fs-fuse │        │            │ 9. │ 2025-03-01 11:06:54 │ storage_client.request_bw │ 3fs-fuse │     │ 10626 │ 622894588.6999174 │ 193001288.42260262 │ 1635843993.7597504 │ 618193735.7544193 │ 915741041.3424696 │ 1083785446.6478138 │ 1283924101.3262858 │            │ batchRead │    │     │        │ 3fs-fuse │        │            │10. │ 2025-03-01 11:06:55 │ storage_client.request_bw │ 3fs-fuse │     │ 10622 │ 618279646.1477239 │  200684401.9138756 │  1598439024.390244 │ 610117422.3305147 │ 911149154.5515901 │ 1089556042.1725667 │ 1282777764.6938653 │            │ batchRead │    │     │        │ 3fs-fuse │        │            │    └─────────────────────┴───────────────────────────┴──────────┴─────┴───────┴───────────────────┴────────────────────┴────────────────────┴───────────────────┴───────────────────┴────────────────────┴────────────────────┴────────────┴───────────┴────┴─────┴────────┴──────────┴────────┴────────────┘    其它Metric可以通过如下命令查询3fs-meta :) select distinct metricName from distributionsSELECT DISTINCT metricNameFROM distributionsQuery id: c035f3af-9c97-4203-9d1b-ffee6eeeea44     ┌─metricName──────────────────────────────────────┐  1. │ MgmtdClient.op.succ_latency                     │  2. │ common_net_batch_read_size                      │  3. │ common_net_batch_write_size                     │  4. │ storage.check_disk.succ_latency                 │  5. │ fdb_latency_commit                              │  6. │ fdb_latency_get                                 │  7. │ fdb_latency_get_range                           │  8. │ fdb_latency_snapshot_get_range                  │  9. │ MgmtdService.WriterLatency                      │ 10. │ MgmtdService.bg.succ_latency                    │ 11. │ MgmtdService.op.succ_latency                    │ 12. │ storage.default.queue_latency                   │ 13. │ storage_client.concurrent_user_calls            │ 14. │ storage_client.inflight_requests                │ 15. │ storage_client.inflight_time                    │ 16. │ storage_client.network_latency                  │ 17. │ storage_client.num_pending_ops                  │ 18. │ storage_client.overall_latency                  │ 19. │ storage_client.request_latency                  │ 20. │ storage_client.server_latency                   │ 21. │ storage.io_submit.size                          │ 22. │ storage.io_submit.succ_latency                  │ 23. │ storage.read.queue_latency                      │ 24. │ storage.read_prepare_buffer.succ_latency        │  
```

这应该是全网首个复现3FS集群的测试, eRDMA提供的标准RDMA RC接口和全地域全可用区的弹性能力是我们能够快速复现的根本原因, 并且在云上可以根据用户需求构建更大规模的集群, 在ECS 9代服务器支持CIPU 2.0 400Gbps的处理能力及云上更大规模的资源供给能力下, 可以媲美DeepSeek线下部署的集群, 进一步优化推理的成本.

后续我们将针对3FS进行更多的测试和分析, 敬请期待~ 也希望这篇文章和阿里云eRDMA技术能够帮助您快速构建测试环境.

参考资料

[1] 
ffrecord: *https://github.com/HFAiLab/ffrecord*
[2] 
smallpond: *https://github.com/deepseek-ai/smallpond*