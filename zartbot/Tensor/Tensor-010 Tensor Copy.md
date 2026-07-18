# Tensor-010 Tensor Copy

> 作者: zartbot  
> 日期: 2024年9月14日 14:34  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492339&idx=1&sn=89f014c36116b2a5159b5daf7c3dac9d&chksm=f995f231cee27b27f6963f143a9de82f543414651bdc68329447eeec67ee78560f1d3cb836d0#rd

---

本文介绍Cute Tiled Copy的抽象结构和相应的内存拷贝流程, 目录如下:

```
1. Cute Copy范式1.1 CopyOperation1.2 Copy_Traits1.3 Copy_Atom1.4 TiledCopy1.5 ThrCopy2. Cute Copy示例
```

## 1. Cute Copy范式

Cutlass Tiled Copy的抽象结构如下所示.

![图片](assets/884dd871696e.jpg)

### 1.1 Copy_Op

Copy_Op是原始的PTX指令, 我们在《Tensor-003 TensorCore架构》中介绍了`ldmatrix`,`cp.async`以及Hopper的TMA等多种内存拷贝指令.在`include/cute/arch`有相应的实现,例如`ldmatrix`

```
struct SM75_U16x8_LDSM_T{  using SRegisters = uint128_t[1];  using DRegisters = uint32_t[4];  CUTE_HOST_DEVICE static void  copy(uint128_t const& smem_src,       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)  {    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);    asm volatile ("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"        : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)        :  "r"(smem_int_ptr));  }};
```

### 1.2 Copy_Traits

Copy_Traits相对于Copy_Op补充了一些元信息, 例如相应的线程ID,源和目的数据的Layout等信息, 通过ThreadIdx和ValueID即可得到数据的位置, 在`/include/cute/atom`目录下有各个平台相关的实现. 如下所示:

```
template <>struct Copy_Traits<SM75_U16x8_LDSM_T>{  // Logical thread id to thread idx (warp)  using ThrID = Layout<_32>;  // Map from (src-thr,src-val) to bit  using SrcLayout = Layout<Shape < _32,_128>,                           Stride<_128,  _1>>;                             // Map from (dst-thr,dst-val) to bit  using DstLayout = Layout<Shape <Shape <  _4, _8>,Shape <_16,  _2,   _4>>,                           Stride<Stride<_256,_16>,Stride< _1,_128,_1024>>>;  // Reference map from (thr,val) to bit  using RefLayout = DstLayout;};
```

### 1.3 Copy_Atom

Copy_Atom是对Copy_Op和Copy_Traits进行的封装, 绑定了数据类型, 并同时检查相应的接口,为上层的TiledCopy提供原子能力. 相关的代码在`include/cute/atom/`中

```
struct Copy_Atom<Copy_Traits<Args...>, CopyInternalType>  : Copy_Traits<Args...>{  using Traits = Copy_Traits<Args...>;  // 基于Copy_Traits的ThreadLayout  using ThrID        = typename Traits::ThrID;  using BitLayoutSrc = typename Traits::SrcLayout;  using BitLayoutDst = typename Traits::DstLayout;  using BitLayoutRef = typename Traits::RefLayout;  // Value Layout  using ValType = CopyInternalType;  using ValLayoutSrc = decltype(recast_layout<uint1_t, ValType>(BitLayoutSrc{}));  using ValLayoutDst = decltype(recast_layout<uint1_t, ValType>(BitLayoutDst{}));  using ValLayoutRef = decltype(recast_layout<uint1_t, ValType>(BitLayoutRef{}));  static constexpr int NumValSrc = size<1>(ValLayoutSrc{});  static constexpr int NumValDst = size<1>(ValLayoutDst{});...  // Check and call instruction, or recurse  template <class SEngine, class SLayout,            class DEngine, class DLayout>  CUTE_HOST_DEVICE  void  call(Tensor<SEngine,SLayout> const& src,       Tensor<DEngine,DLayout>      & dst) const  {    //针对const的情况执行copy_unpack    //针对Shape是Tuple的情况, 递归剥离Mode来处理.     ....  }...};
```

### 1.4 TiledCopy

TiledCopy 基于TiledMMA Layout来重复调用Copy Atom实现更大的块的拷贝能力. 它在`include/cute/atom/copy_atom.hpp`中定义.  首先它基于Copy_Atom, TV_Layout和Tiler_MN定义如下

```
template <class Copy_Atom,          class LayoutCopy_TV,  // (tid,vid) -> coord   [Need not be 2D...]          class ShapeTiler_MN>  // coord spacestruct TiledCopy : Copy_Atom{  // Layout information from the CopyAtom  using AtomThrID     = typename Copy_Atom::ThrID;        // thrid -> thr_idx  using AtomLayoutSrc = typename Copy_Atom::ValLayoutSrc; // (thr,val) -> offset  using AtomLayoutDst = typename Copy_Atom::ValLayoutDst; // (thr,val) -> offset  using AtomLayoutRef = typename Copy_Atom::ValLayoutRef; // (thr,val) -> offset  using AtomNumThr = decltype(size<0>(AtomLayoutRef{}));  using AtomNumVal = decltype(size<1>(AtomLayoutRef{}));  // Layout information for the TiledCopy  using Tiler_MN       = ShapeTiler_MN;  using TiledLayout_TV = LayoutCopy_TV;  using TiledNumThr    = decltype(size<0>(TiledLayout_TV{}));  using TiledNumVal    = decltype(size<1>(TiledLayout_TV{}));
```

然后在其中定义了多个函数, `tile2thrfrg`从((TileM,TileN,...), (RestM,RestN,...)) Layout转换为((ThrV,ThrX),FrgV,(RestM,RestN,...)) `tidfrg_S`和`tidfrg_D`这两个函数分别处理源(STensor)和目标(DTensor)张量的切片, 在内部调用了`tile2thrfrg`函数.

```
 tidfrg_S(STensor&& stensor)  {    // Tile the stensor and compute the (src-thr, src-val) -> (ref-thr, ref-val) layout    return tile2thrfrg(zipped_divide(stensor,Tiler_MN{}), right_inverse(AtomLayoutRef{}).compose(AtomLayoutSrc{}));  }    tidfrg_D(DTensor&& dtensor)  {    // Tile the dtensor and compute the (dst-thr, dst-val) -> (ref-thr, ref-val) layout    return tile2thrfrg(zipped_divide(dtensor,Tiler_MN{}), right_inverse(AtomLayoutRef{}).compose(AtomLayoutDst{}));  }
```

另外还有`get_layoutS_TV` 和 `get_layoutD_TV`以及`get_layoutS_MN` 和 `get_layoutD_MN`函数用于生成`(thr_idx,val_idx) -> (M,N)`和`(M,K) -> (thr_idx,val_idx)`的映射.

`get_slice(ThrIdx const& thr_idx)` 和 `get_thread_slice(ThrIdx const& thr_idx)`：是TiledCopy的核心函数, 获取特定线程索引的切片信息。 它们返回的对象是ThrCopy

```
  get_slice(ThrIdx const& thr_idx)  {    return ThrCopy<TiledCopy, ThrIdx>(thr_idx);  }
```

### 1.5 ThrCopy

基于TileCopy.get_slice(threadIdx)返回的线程级的描述符对象ThrCopy, 通过ThrCopy的`partition_S/D`函数可以获得相应的拷贝操作数, 对于某些情况还可以使用`retile_S/D`函数变换到拷贝函数所需要的形状,并最后通过调用cute::copy函数实现拷贝.

## 2. Cute Copy示例

我们以`/blob/v3.5.1/examples/cute/tutorial/tiled_copy.cu`代码来分析TileCopy实例.
原始矩阵数据类型为float, 矩阵为MxN=4096x8192

```
  using Element = float;  // 定义矩阵Shape  auto tensor_shape = make_shape(4096, 8192);  // 分配和初始化矩阵  thrust::host_vector<Element> h_S(size(tensor_shape));  thrust::host_vector<Element> h_D(size(tensor_shape));  for (size_t i = 0; i < h_S.size(); ++i) {    h_S[i] = static_cast<Element>(i);    h_D[i] = Element{};  }  thrust::device_vector<Element> d_S = h_S;  thrust::device_vector<Element> d_D = h_D;    //创建Tensor  Tensor tensor_S = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())), make_layout(tensor_shape));  Tensor tensor_D = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())), make_layout(tensor_shape));
```

然后以Block Shape 128x64进行拷贝, 并验证TensorShape和BlockShape的维度是否可以整除, 同时是否满足weakly_compatible条件, weakly_compatible条件可以参考《Tensor-008 CuTe Layout代数》相关的内容.

```
auto block_shape = make_shape(Int<128>{}, Int<64>{});  if ((size<0>(tensor_shape) % size<0>(block_shape)) || (size<1>(tensor_shape) % size<1>(block_shape))) {    std::cerr << "The tensor shape must be divisible by the block shape." << std::endl;    return -1;  }  // Equivalent check to the above  if (not weakly_compatible(block_shape, tensor_shape)) {    std::cerr << "Expected the tensors to be weakly compatible with the block_shape." << std::endl;    return -1;  }
```

然后基于BlockShape 利用TileDivide拆分原始矩阵

```
  Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape);      // ((M, N), m', n')  Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape);      // ((M, N), m', n')
```

定义一个Block内的Thread Layout和Vector Copy Layout

```
  // Thread arrangement  Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));  // Vector dimensions  Layout vec_layout = make_layout(make_shape(Int<4>{}, Int<1>{}));
```

Launch Kernel的Grid和block按照拆分的Tile数和Thread数定义

```
  dim3 gridDim (size<1>(tiled_tensor_D), size<2>(tiled_tensor_D));   // Grid shape corresponds to modes m' and n'  dim3 blockDim(size(thr_layout));    copy_kernel_vectorized<<< gridDim, blockDim >>>(    tiled_tensor_S,    tiled_tensor_D,    thr_layout,    vec_layout);
```

copy_kernel_vectorized代码如下

```
template <class TensorS, class TensorD, class ThreadLayout, class VecLayout>__global__ void copy_kernel_vectorized(TensorS S, TensorD D, ThreadLayout, VecLayout){  using namespace cute;  using Element = typename TensorS::value_type;  // 通过BlockIdx.x/y获取相应的源和目的tile  Tensor tile_S = S(make_coord(_, _), blockIdx.x, blockIdx.y);  // (BlockShape_M, BlockShape_N)  Tensor tile_D = D(make_coord(_, _), blockIdx.x, blockIdx.y);  // (BlockShape_M, BlockShape_N)  // Define `AccessType` which controls the size of the actual memory access.  using AccessType = cutlass::AlignedArray<Element, size(VecLayout{})>;  // 定义Copy_Atom  using Atom = Copy_Atom<UniversalCopy<AccessType>, Element>;  //构造TiledCopy对象  auto tiled_copy =    make_tiled_copy(      Atom{},                       // access size      ThreadLayout{},               // thread layout      VecLayout{});                 // vector layout (e.g. 4x1)  //基于ThreadIdx.x和tiled_copy对象生成thr_copy对象  auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);  //基于thrCopy对象生成thread_tile张量  Tensor thr_tile_S = thr_copy.partition_S(tile_S);             // (CopyOp, CopyM, CopyN)  Tensor thr_tile_D = thr_copy.partition_D(tile_D);             // (CopyOp, CopyM, CopyN)  //基于Thread_tile创建寄存器文件张量  Tensor fragment = make_fragment_like(thr_tile_D);             // (CopyOp, CopyM, CopyN)  //将源数据从GMEM拷贝到RMEM, 再从RMEM拷贝到目标GMEM.  copy(tiled_copy, thr_tile_S, fragment);  copy(tiled_copy, fragment, thr_tile_D);}
```