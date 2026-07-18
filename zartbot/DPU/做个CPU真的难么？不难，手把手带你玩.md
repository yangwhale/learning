# 做个CPU真的难么？不难，手把手带你玩

> 作者: zartbot  
> 日期: 2021年7月4日 07:23  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486043&idx=1&sn=64bbde0b024c2a495eccddf73f6fdf07&chksm=f9961a99cee1938f48f0b9248b06b5c6dba950fa75695b868d3b3c2da532c49f71622530825b#rd

---

❝
龙芯终于也开始IPO了，二十年前做一个CPU的确很难，但是放在今天其实也没有那么的难。
❞
龙芯的好坏成败就不多评价了，作为中国人希望自主可控的技术好起来，但自主可控并不代表只有龙芯、华为、鸿蒙等唯一选择，这个时代赋予我们的应是百花争鸣的勇气，例如个人比较看好的计算所的`香山`处理器,最近看到两个事情...

一个是某公众号的文章[美政府干扰中资收购 致MIPS架构发展路断](https://mp.weixin.qq.com/s?__biz=MjM5MzA0MDUyMg==&mid=2247508969&idx=1&sn=0f99aef20d87b38e21504b96e2c6a8fb&scene=21#wechat_redirect),其中说到:
❝
如果不是美国政府对MIPS的囚禁，MIPS架构本来应该有机会在中国获得新生。
❞
若是收成了，说不定国内的整个计算生态又要落后十多年了，看看RISC-V吧。然而再搜索了一下，发现一个叫`芯联芯`的声明,龙芯中科彻底抛弃MIPS与事实不符，仲裁进入审理阶段：

![图片](assets/433e9956abe7.jpg)

笑而不语....当然关于LoongArch ISA，最大的难度可能还是在软件生态上了，知乎上有个问答

❝
LoongArch 是全新的指令集，不是在 MIPS 上做的扩展。包含基础指令 337 条、虚拟机扩展 10 条、二进制翻译扩展 176 条、128 位向量扩展 1024 条、256 位向量扩展 1018 条，共计 2565 条原生指令。相对于MIPS，摒弃了部分不适合现代CPU的指令，又做了大量改进和扩展
❞
当然关于RISC-V生态，某两个动物公司说的好难...其实也不是那么的难，去年疫情期间，躲在家里安静的做了一个MIPS的小CPU,其实并不是很难，开源生态的基础上做点事情还是很容易的。

今天要讲的是Xilinx**的Principal Engineer Eugene Tarassov开源出来的vivado-risc-v[1]， CPU核来自于UCB的BOOM。其它很多IP Core**例如Ethernet Controller、AXI都是来自开源项目。

接下来会以RISC-V为蓝本，实现一套DataCentric Computing的ISA,指令集的设计会利用很多范畴论的东西:)我才是真全栈工程师~

![图片](assets/82572b81df81.png)

在这里我们使用了一块基于Artix-7的FPGA开发板Digilent Nexys A7-100T， 整个板子大概2000元RMB左右，淘宝自己搜连接，而且Vivado的免费版也可以使用。

![图片](assets/61223b52a7ac.jpg)

### 安装开发环境 

主要就是Ubuntu 20.04的操作系统，然后从Xilinx官方下载vivado 2020.2，基本上就是下载binary文件直接执行即可，再安装前需要注意安装一下:

```
sudo apt install libncurses5
```

安装路径最好还是标准的/opt/Xilinx/Vivado/2020.2.安装完成后，插上Nexys A7的usb线，安装串口驱动

```
cd /opt/Xilinx/Vivado/2020.2/data/xicom/cable_drivers/lin64/install_script/install_driverssudo ./install_drivers
```

然后需要将A7的Boardfile拷贝到vivado中:

```
https://github.com/Digilent/vivado-boards/tree/master/new/board_files选择nexys-a7-100t文件夹拷贝到/opt/Xilinx/Vivado/2020.2/data/boards/board_files
```

### 综合及编译Linux 

Eugene的脚本写的非常好,也有网友录制了视频 

https://www.youtube.com/watch?v=ECA-D6ZFnU4

首先是下载代码和构建编译环境

```
sudo apt install git makegit clone https://github.com/eugene-tarassov/vivado-risc-v.gitcd vivado-risc-vmake apt-installmake update-submodules
```

然后直接build bitstream文件即可

![图片](assets/4abc74f92184.png)

```
source /opt/Xilinx/Vivado/2020.2/settings64.shmake CONFIG=rocket64b1 BOARD=nexys-a7-100t bitstream
```

注意整个过程有可能耗时很长, 完了以后将一个带有TF的读卡器插入安装工作站的USB接口，执行

```
./mk-sd-card
```

注意选择SD卡路径，别搞错了,然后将TF卡插入到A7-100T开发板中.

![图片](assets/09bd87b9810a.png)

### 烧写 

使用如下命令打开Vivado

```
make CONFIG=rocket64b1 BOARD=nexys-a7-100t bitstream vivado-gui
```

![图片](assets/cd799d6d8886.png)

我们可以看到资源已经占用的很满了，毕竟是一个非常廉价的入门级的FPGA， 拿来玩玩挺好的

![图片](assets/0a06de8fddfd.png)

然后点击左侧Open Hardware Manager，然后再点击auto connect，然后右键点击a7-100t

![图片](assets/5b5e4f11560f.png)

选择`Add configuration memory device`, 在search框中输入s25fl128sxxxxxx0

![图片](assets/a1296766ae5e.png)

然后Vivado会自动弹出是否需要program device， 点击yes即可

![图片](assets/2ff65365d31f.png)

在弹出的窗口选择configuration file

```
/opt/fpga/vivado-risc-v/workspace/rocket64b1/nexys-a7-100t-riscv.mcs
```

![图片](assets/c06de299d35e.png)

然后点击`OK`就开始烧写flash了

![图片](assets/6396c22cad5a.png)

然后使用uart console登录,打开console后 power cycle 开发板

```
sudo apt install python3-serialsudo miniterm /dev/ttyUSB1 115200--- Miniterm on /dev/ttyUSB1  115200,8,N,1 ------ Quit: Ctrl+] | Menu: Ctrl+T | Help: Ctrl+T followed by Ctrl+H ---
```

![图片](assets/dd951a6b06df.png)

最后使用debian、debian登录即可

![图片](assets/b8176ceb5947.png)

### 感想 

然后我们基于这个底子，就可以干很多有趣的事情了，可以针对云数据中心的特定场景，通过范畴论的方法来抽取指令集的需求，然后实现很多有趣的东西了，当然这事我先保密一段时间：）

其实很多事情并没有那么的难，可惜我们总是好高骛远，或者就地躺平，伟大的工程需要一代代的人的努力，从上世纪的两弹一星，到今天载人航天和空间站，哪个项目不是要花费二三十年时间才做成的。而针对信息技术，我看到的只是大量的散兵游勇，在工业界的，很多人只是KPI-Driven的开发，而所谓潜心在院校科学界的，通常又把自己锁在理想的象牙塔中。看看Berkeley Arch Research[2] 在指令集、前后端工具、片上网络**等多个项目上齐心协力... 而我们呢？某些互联网公司的部门墙高的不多说了...

另一方面，国内现在很多科创都来自于科研人员，很多创业公司CEO是技术出生，在执行过程中通常把技术想的太完美，同时也缺乏企业运营的经验，把技术看的非常重基本上代替了CTO的职责，对于CFO和Operation估计不足，对于成本估计不足，这些都是问题，龙芯的问题就在于此，依图的问题也在于此.... 我一直跟很多同事讲，当你到了经理的职位上以后，真的需要抽时间认真读一个MBA，不是去混朋友圈关系的，而是踏踏实实的去学公司运营的知识，可惜国内的MBA... 呵呵...

吐槽结束，继续搬砖....

#### Reference

[1]
vivado-risc-v: https://github.com/eugene-tarassov/vivado-risc-v
[2]
Berkely Archtecture Research: https://bar.eecs.berkeley.edu/