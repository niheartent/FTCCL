# RCCL

ROCm Communication Collectives Library

## 简介

RCCL是一个独立的针对 GPU 的标准集合通信库，实现了all-reduce, all-gather, reduce, broadcast, reduce-scatter, gather, scatter 以及 all-to-all等通信算子。此外，它还初步支持直接的 GPU 与 GPU 之间的发送和接收操作。该库经过优化，在使用 PCIe、xGMI 以及使用 InfiniBand Verbs 或 TCP/IP 套接字的网络连接的平台上能够实现高带宽。RCCL 支持在拥有任意数量 GPU 的单节点或多节点环境进行使用，并且可以用于单进程或多进程（例如 MPI）的应用程序中。

## 环境需求

1. 支持 ROCm 平台的 GPU。
2. GPU 适配的 ROCm 软件栈(关键是HIP runtime 和 HIP-Clang)。可直接加载适配环境的 DTK 软件包的脚本文件：source ${DTK包路径}/env.sh , 如加载DTK-25.04.4的配置：
```shell
source /public/honme/dtk/dtk-25.04.4/env.sh
```
3. 6.3.13-V1.12.0a及以上的驱动版本。可通过hy-smi --showdriverversion 或 rocm-smi --showdriverversion 指令检查驱动版本，如果版本不符合，建议联系环境管理员更新驱动版本。

## RCCL包构建

### 使用CMake编译动态库:

```shell
$ git clone http://developer.sourcefind.cn/codes/OpenDAS/custom-rccl.git
$ cd rccl
$ mkdir build
$ cd build
$ CXX=hipcc cmake ..
$ make -j 32
```
## 测试
在 RCCL 中，使用了 Googletest 框架来实现的单元测试（Unit Test , UT）。这些 RCCL 单元测试依赖Googletest 1.10 或更高版本才能正确构建和执行。
### UT测试用例构建
RCCL UT需要随RCCL动态库一起编译构建。
通过添加编译选项BUILD_TESTS=ON，在编译RCCL动态库的同时，构建UT用例：
```shell
$ git clone http://developer.sourcefind.cn/codes/OpenDAS/custom-rccl.git
$ cd rccl
$ mkdir build
$ cd build
$ CXX=hipcc cmake -DBUILD_TESTS=ON ..
$ make -j 32
```
rccl/build/test/即是编译完成的UT项目目录
### 执行UT测试用例
RCCL 单元测试的名称采用以下格式：
```shell
CollectiveCall.[Type of test]
```
直接执行rccl-UnitTests即可全量测试RCCL UT
```shell
cd rccl/build/test
./rccl-UnitTests
```
可以使用--gtest_filter选项对 rccl 单元测试进行过滤操作，比如只测试 AllReduce 测例：
```shell
./rccl-UnitTests --gtest_filter="AllReduce*"
```