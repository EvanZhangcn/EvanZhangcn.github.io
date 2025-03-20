---
title: "L7 - 231n ConvNets"
date: 2025-03-20T09:31:51Z
draft: false
categories: ["未分类"]  # 在此编辑分类
tags: []               # 在此添加标签
---

****## Convolutional Neural Networks (CNNs / ConvNets)

- **CNNs 与普通神经网络的相似性**：
    
    - 由具有可学习权重和偏置的神经元组成。
    - 每个神经元执行点积操作，并可选地跟随非线性激活函数。
    - 整个网络表示一个可微分评分函数。
    - 最后一层使用损失函数（如 SVM/Softmax）。
- **CNN 的不同之处**：
    
    - 明确假设输入是图像。
    - 将图像属性编码到架构中，提高前向传播效率(make the forward function more efficient)。
    - 大幅减少参数数量。




### Architecture Overview

1. **常规神经网络 (Regular Neural Networks)**:
   - 输入是一个向量，通过一系列隐藏层(hidden layer)进行转换。
   - 每个隐藏层由**一组神经元**组成，每个神经元与前一层的所有神经元**全连接**。
   - 最后一层全连接层称为“输出层”(output layer)，在分类任务中表示类别分数。
   - **对于图像数据，常规神经网络的扩展性较差(Regular Neural Nets don’t scale well to full images.)。**
	   - 例如，CIFAR-10 图像的尺寸为 $32 \times 32 \times 3$，第一个隐藏层的单个神经元将有 $32 \times 32 \times 3 = 3072$ 个权重。对于更大的图像（如 $200 \times 200 \times 3$），**权重数量将急剧增加，导致参数过多和过拟合问题。**

2. **卷积神经网络 (Convolutional Neural Networks, ConvNets)**:
   - 利用输入是图像的特性，将神经元排列成三维结构：宽度、高度、深度(width, height, depth)。
	   - **强调： 此处的depth指的是the third dimension of an activation volume， 并不指的是整个神经网络的深度depth.**（神经网络深度 = the total number of layers in a network ）
   - 输入图像在 CIFAR-10 中是一个**激活量的三维体积(volume of activations)**，并且这个volume的尺寸为 $32 \times 32 \times 3$ (width, height, depth respectively).。
   - **神经元只连接到前一层的局部区域，而不是全连接。**
   - 最终输出层的尺寸为 $1 \times 1 \times 10$，将整个图像转换为类别分数的向量（reduce the full image into a single vector of class scores）。

3. **卷积神经网络的层 (Layers in ConvNets)**:
   - 每一层将输入的三维体积转换为输出的三维体积。
   - 每一层都有一个简单的 API，通过可微函数（可能有参数）进行转换。

总结：
A ConvNet is made up of Layers（可以认为神经网络是a sequence of layers）. Every Layer has a simple API: **It transforms an input 3D volume to an output 3D volume with some differentiable function that may or may not have parameters.**
- **常规神经网络** (Regular Neural Networks) 在处理图像数据时扩展性较差，特别是在图像尺寸较大时，参数数量会急剧增加，导致过拟合问题。
- **卷积神经网络** (Convolutional Neural Networks) 通过将神经元排列成三维结构，并只连接局部区域，有效地减少了参数数量，提高了模型的扩展性和性能。






### Layers used to build ConvNets

1. **ConvNet的基本结构**：
   - ConvNet由一系列层组成，每一层通过可微函数将输入的激活体积（activation volume）转换为输出的激活体积。
   - 主要使用的层类型包括：**卷积层（Convolutional Layer, CONV）、池化层（Pooling Layer, POOL）和全连接层（Fully-Connected Layer, FC）。**

2. **示例架构**：
   - 一个简单的ConvNet架构可以表示为：[INPUT - CONV - RELU - POOL - FC]。
   - **INPUT层**：输入图像的原始像素值，例如CIFAR-10数据集的图像大小为[32x32x3]，表示宽度32、高度32、3个颜色通道（R,G,B）。
   - **CONV层**：计算与输入局部区域连接的神经元的输出，每个神经元计算其权重与输入区域的点积。例如，使用12个滤波器时，输出体积可能为[32x32x12]。
   - **RELU层**：应用逐元素的激活函数，如$f(x) = \max(0, x)$，输出体积大小不变，仍为[32x32x12]。
   - **POOL层**：沿空间维度（宽度、高度）进行下采样操作(downsampling operation)，输出体积可能为[16x16x12]。
   - **FC层**：计算类别得分，输出体积大小为[1x1x10]，其中每个数字对应一个类别的得分（例如CIFAR-10的10个类别）。

3. **层的参数与超参数**：
   - **CONV层和FC层**包含**可训练的参数（权重和偏置）**，这些参数通过梯度下降法进行训练，以使ConvNet计算的类别得分与训练集中的标签一致。
   - **RELU层和POOL层**实现固定的函数，不包含可训练的参数。
   - **CONV层、FC层和POOL层**可能包含超参数（如卷积核大小、步幅等），而**RELU层**通常没有超参数。

4. **激活体积的变化**：
   - ConvNet通过逐层转换，将原始图像像素值转换为最终的类别得分。
   - 每一层的输入和输出都是3D体积，通常表示为$[宽度 \times 高度 \times 深度]$。
   - 最后一层的输出体积包含每个类别的得分，通常只可视化得分最高的几个类别。





#### Convolutional Layer

1. **卷积层 (Convolutional Layer, CONV Layer)**:
   - 是卷积神经网络的核心构建块，负责大部分计算工作。
   - **卷积层的权重矩阵通常被称为卷积核或过滤器.**
   - 参数由一组可学习的滤波器（filters）组成，每个滤波器在空间上较小（如 $5 \times 5 \times 3$），但贯穿输入体积的整个深度。Every filter is small spatially (along width and height), but extends through the full depth of the input volume.
   - 在前向传播过程中，we slide (more precisely, convolve 滑动（更准确说，进行卷积运算）) each filter across the width and height of the input volume，并计算滤波器与输入的点积，生成**二维激活图（activation map）**。
   - 每个卷积层有多个滤波器（如下图有6 个），每个滤波器生成一个独立二维的激活图（如下图28x28的activate map），(6个filters产生6个activation map)这些激活图**沿深度**维度堆叠形成**输出体积**(6x28x28 output image)。
   - 对于这个output volume: **在卷积神经网络中，3D输出体积中的每个元素可以被视为一个神经元的输出。** 这个神经元只关注输入数据中的一个小区域（即感受野receptive field），并且与空间上左右相邻的神经元共享参数。这是因为这些神经元的输出是通过应用相同的滤波器（filter）计算得到的。
   - ![[L7 - 231n ConvNets by_2025-03-07.png]]

2. **局部连接 (Local Connectivity)**:
   - 在高维输入（如图像）的情况下，如果每个神经元都连接到前一层的所有神经元，参数数量会变得极其庞大。
	   - 例如，假设输入是一张 100x100 的 RGB 图像（3 个通道），输入体积的大小为 100x100x3 = 30,000。如果下一层的每个神经元都连接到这 30,000 个输入神经元，那么仅一层就需要 30,000 个参数。如果网络有多层，参数数量会呈指数级增长，导致计算资源无法承受。**因此，我们应该考虑将每个神经元连接到a local region of the input volume.**
   - **每个神经元只连接到输入体积的局部区域，称为感受野（receptive field），其大小由滤波器尺寸决定。**
   - #感受野英文定义 The spatial extent of this connectivity is a hyperparameter called the receptive field of the neuron.(equivalently this is the filter size)
   - ![[L7 - 231n ConvNets by_2025-03-07-1.png|600x328]]
   - 连接在spatial dimension空间维度（宽度和高度）上是局部的，但在深度维度上是全连接的。
	   - 再次强调：The connections are local in 2D space (along width and height), but always full along the entire depth of the input volume.
   - 示例：
     - 输入体积(input volume)为 $[32 \times 32 \times 3]$，滤波器尺寸(the filter size ,也可以称为receptive field)为 $5 \times 5$，则每个神经元有 $5 \times 5 \times 3 = 75$ 个权重和 1 个偏置参数。
     - 输入体积为(input volume) $[16 \times 16 \times 20]$，滤波器尺寸为 $3 \times 3$，则每个神经元有 $3 \times 3 \times 20 = 180$ 个连接。


3. **空间排列 (Spatial Arrangement)**:
   - 我们接下来要探讨在output volume中，有多少个神经元，它们是如何排列的：
   - **输出体积的大小由三个超参数控制：深度（depth）、步幅（stride）和零填充（zero-padding）。**
   - 深度对应于滤波器的数量，每个滤波器学习输入中的不同特征。
	   - 一组神经元如果都在观察输入中的同一区域，则被称为一个“深度列”（depth column），也有人称之为“纤维”（fibre）。
   - 步幅决定滤波器滑动的步长，步幅为 1 时每次移动 1 个像素，步幅为 2 时每次移动 2 个像素。
   - 零填充是指在输入体积的边界周围填充零值。
	   - **零填充用于控制输出体积的空间尺寸(control the spatial size of the output)**，
	   - 常用公式为 $P = \frac{F-1}{2}$，其中 $F$ 为滤波器尺寸
	   - 通常我们设置$S=1$ 时确保输入和输出尺寸相同。
   - 输出体积的空间尺寸计算公式为：
   - ![[L7 - 231n ConvNets by_2025-03-07-2.png]]
     $$
     \frac{W - F + 2P}{S} + 1
     $$
     其中 $W$ 为输入宽度，$F$ 为滤波器尺寸，$P$ 为零填充，$S$ 为步幅。

4. **参数共享 (Parameter Sharing)**:
   - 通过假设特征在空间位置上的平移不变性（即某些特征（如水平边缘）在图像的不同位置可能同样重要），从而显著减少参数数量。
   - **每个深度切片（depth slice）中的神经元共享相同的权重和偏置。**
   - 示例：Krizhevsky 等人在 ImageNet 2012 中使用的卷积层，输入尺寸为 $[227 \times 227 \times 3]$，滤波器尺寸为 $11 \times 11$，步幅为 4，无零填充，输出体积为 $[55 \times 55 \times 96]$，每个神经元连接到 $[11 \times 11 \times 3]$ 的输入区域，共享 96 组权重。
   - 参数共享的局限性：
	   - - 当输入图像具有**特定的中心化结构**时，不同位置可能需要学习完全不同的特征。
	   - 一个典型的例子是人脸图像：如果人脸被居中放置在图像中，那么眼睛和头发的特征可能只出现在图像的特定区域。在这种情况下，不同位置需要学习不同的特征。
	   - 对于这种情况，通常会**放松参数共享**的机制，转而使用**局部连接层**（Locally-Connected Layer）。**在局部连接层中，每个神经元使用独立的参数，而不是共享参数。**
-

5. **Numpy 示例**:
   - 输入体积为 $X$，形状为 $(11, 11, 4)$，滤波器尺寸为 $5 \times 5$，步幅为 2，无零填充，输出体积为 $4 \times 4$。
   - 激活图的计算示例：
     $$
     V[0,0,0] = \text{np.sum}(X[:5,:5,:] \times W0) + b0
     $$
     其中 $W0$ 为权重，$b0$ 为偏置。



### 卷积层（Conv Layer）总结

#### 1. **输入与输出**
- **输入体积**：大小为 $W_1 \times H_1 \times D_1$。（D1是Depth1， channels）
- **输出体积**：大小为 $W_2 \times H_2 \times D_2$。

#### 2. **超参数**
卷积层需要以下四个超参数：
1. **滤波器数量**：$K$。
2. **滤波器空间尺寸**：$F$（即滤波器的宽度和高度）。
3. **步幅**：$S$。
4. **零填充**：$P$。

#### 3. **输出体积的计算**
输出体积的尺寸通过以下公式计算：
- 宽度：$W_2 = \frac{W_1 - F + 2P}{S} + 1$
- 高度：$H_2 = \frac{H_1 - F + 2P}{S} + 1$
- 深度：$D_2 = K$

#### 4. **参数共享**
- 每个滤波器的权重数量：$F \cdot F \cdot D_1$。
- 总权重数量：$(F \cdot F \cdot D_1) \cdot K$。
- 偏置数量：$K$。

#### 5. **输出体积的生成**
- 输出体积的第 $d$ 个深度切片（大小为 $W_2 \times H_2$）是通过以下操作生成的：
  1. 使用第 $d$ 个滤波器对输入体积进行有效卷积（valid convolution），步幅为 $S$。
  2. 加上第 $d$ 个偏置。










#### **Implementation as Matrix Multiplication（卷积层的矩阵乘法实现）**
[Explain: Click Me!](https://cs231n.github.io/convolutional-networks/#:~:text=Implementation%20as%20Matrix%20Multiplication.)
#### 1. **核心思想**
卷积操作本质上是滤波器与输入图像的局部区域进行点积运算。为了高效实现卷积层的前向传播，可以将卷积操作转化为一个大规模的矩阵乘法。

#### 2. **im2col 操作**
- **输入图像**：假设输入图像大小为 $227 \times 227 \times 3$。
- **滤波器**：假设使用 $11 \times 11 \times 3$ 的滤波器，步幅为 4。
- **操作**：
  1. **将输入图像(注意不是“滤波器”)中每个 $11 \times 11 \times 3$ 的局部区域（局部区域的大小时滤波器的大小）**拉伸为一个列向量，大小为 $11 \times 11 \times 3 = 363$。**
  2. 在输入图像上以步幅 4 滑动，得到 $(227 - 11) / 4 + 1 = 55$ 个位置（宽度和高度相同）。
  3. 最终生成一个矩阵 $X_{\text{col}}$，大小为 $363 \times 3025$，其中每一列是一个拉伸后的感受野，总共有 $55 \times 55 = 3025$ 个感受野，感受野的大小是：滤波器的大小size ：363。
- **注意**：由于感受野之间存在重叠，输入体积中的某些值可能会在 $X_{\text{col}}$ 中被多次复制。

#### 3. **滤波器权重拉伸**
- 将卷积层的权重（滤波器）拉伸为行向量。例如，如果有 96 个 $11 \times 11 \times 3$ 的滤波器，则生成一个矩阵 $W_{\text{row}}$，大小为 $96 \times 363$。

#### 4. **矩阵乘法**
- 卷积操作等价于进行一次大规模的矩阵乘法：$\text{np.dot}(W_{\text{row}}, X_{\text{col}})$。
- 该操作计算每个滤波器在每个感受野位置的点积。
- 在上述例子中，矩阵乘法的输出大小为 $96 \times 3025$，表示每个滤波器在每个位置的输出，在这一步加上偏置项b， 然后再进行下一步的结果重塑。

#### 5. **结果重塑**
- 将矩阵乘法的结果重塑为正确的输出维度 $55 \times 55 \times 96$。

#### 6. **优缺点**
- **优点**：
  - 可以利用高效的矩阵乘法实现（如 BLAS API）。
  - 同样的 im2col 思想可以用于池化操作。
- **缺点**：
  - **由于输入体积中的某些值被多次复制，可能会占用大量内存。**



### **卷积前向传播过程回顾**

在深入反向传播之前，让我们先回顾一下使用 im2col 方法的前向传播过程：
- **输入图像**: $X$ 维度为 $H \times W \times C$ (例如 $227 \times 227 \times 3$)
- **卷积核**: $W$ 维度为 $H_k \times W_k \times C \times F$ (例如 $11 \times 11 \times 3 \times 96$)
- **步骤**:
  1. **im2col 变换**: 将输入 $X$ 转换为矩阵 $X_{col}$ (维度: $C \times H_k \times W_k \times H_{out} \times W_{out}$)
  2. **权重重塑**: 将 $W$ 重塑为 $W_{row}$ (维度: $F \times (C \times H_k \times W_k)$)
  3. **矩阵乘法**: $Y = W_{row} \cdot X_{col} + b$ (维度: $F \times (H_{out} \times W_{out})$)
  4. **结果重塑**: 将输出重塑为 $H_{out} \times W_{out} \times F$ (例如 $55 \times 55 \times 96$)

### **反向传播的数学推导**

下面仍然是对于一张图片的讨论，假设我们已计算出损失函数 $L$ 对输出 $Y$ 的梯度 $\frac{\partial L}{\partial Y}$ (称为 $dY$)，现在需要计算:
1. $\frac{\partial L}{\partial b}$ (偏置的梯度)
2. $\frac{\partial L}{\partial W}$ (权重的梯度)
3. $\frac{\partial L}{\partial X}$ (输入的梯度)
### 1.1 计算偏置梯度 $\frac{\partial L}{\partial b}$

由于 $Y = W_{row} \cdot X_{col} + b$，我们有:

$$
\frac{\partial L}{\partial b} = \sum_{i,j} \frac{\partial L}{\partial Y_{i,j}} = \sum_{\text{空间位置}} dY
$$

这意味着我们只需对 $dY$ 在所有空间位置上求和:

```python
# 对空间位置求和，得到偏置梯度
#dY本身的维度是(F, Hout，Wout) 与  前向传播最终调整过形式的Y维度相同
db = dY.reshape(F, -1).sum(axis=1)  # 维度: (F,)
#也可以用下面代码
db = torch.sum(dY, dim = (1,2))
```

### 1.2 计算权重梯度 $\frac{\partial L}{\partial W}$

对于权重梯度，我们使用链式法则:

$$
\frac{\partial L}{\partial W_{row}} = \frac{\partial L}{\partial Y} \cdot \frac{\partial Y}{\partial W_{row}} = dY \cdot X_{col}^T
$$

具体实现:

```python
# 首先将 dY 重塑为与 Y 相同的形状
dY_reshaped = dY.reshape(F, -1)  # 维度: (F, H_out*W_out)

# 计算权重梯度
dW_row = dY_reshaped.dot(X_col.T)  # 维度: (F, C*H_k*W_k)

# 重塑为原始权重形状
dW_row = dW_row.shape(F, C, H_k, W_k)
````


### 1.3 计算输入梯度 $\frac{\partial L}{\partial X}$

这一步需要计算 $X_{col}$ 的梯度，然后将其转回原始输入形状:

$$
\frac{\partial L}{\partial X_{col}} = W_{row}^T \cdot \frac{\partial L}{\partial Y} = W_{row}^T \cdot dY
$$

然后需要执行 `col2im` 操作，该操作是 `im2col` 的逆操作:

```python
# 计算 X_col 的梯度
dX_col = W_row.T.dot(dY_reshaped)  # 维度: (C*H_k*W_k, H_out*W_out)

# 执行 col2im 操作，将 dX_col 转回原始输入形状
dX = col2im(dX_col, X.shape, H_k, W_k, stride, padding)  # 维度: (H, W, C)
```




#### **Dilated convolutions**（空洞卷积）

#### 1. **核心概念**
- 空洞卷积是卷积层的一个新扩展，引入了一个新的超参数：**空洞率（dilation）**。
- 传统的卷积滤波器是连续的（即滤波器中的每个单元紧密相邻），而空洞卷积允许滤波器单元之间存在间隔。

#### 2. **空洞率的作用**
- **空洞率 0**：滤波器是连续的，例如一维滤波器 $w$ 大小为 3，计算方式为：$w[0] \cdot x[0] + w[1] \cdot x[1] + w[2] \cdot x[2]$。
- **空洞率 1**：滤波器单元之间存在间隔，例如一维滤波器 $w$ 大小为 3，计算方式为：$w[0] \cdot x[0] + w[1] \cdot x[2] + w[2] \cdot x[4]$，即每次应用滤波器时跳过 1 个输入单元。

#### 3. **空洞卷积的优势**
- **扩大感受野**：空洞卷积可以更快速地扩大神经元的有效感受野（effective receptive field）。
  - 例如，堆叠两个 $3 \times 3$ 的传统卷积层，第二层神经元的有效感受野为 $5 \times 5$。
  - ![[L7 - 231n ConvNets by_2025-03-07-3.png]]
  - 如果使用空洞卷积，有效感受野会增长得更快。
- **减少层数**：空洞卷积可以更高效地融合输入的空间信息，从而减少所需的网络层数。

#### 4. **应用场景**
- 空洞卷积通常与传统的 0 空洞率卷积结合使用，以在保持计算效率的同时扩大感受野。
- 适用于需要捕捉更大范围上下文信息的任务，如图像分割、语义分割等。









#### **Pooling Layer**

1. **池化层的作用**：[Explain: Click Me!](https://cs231n.github.io/convolutional-networks/#pool#:~:text=Pooling%20Layer)
   - 逐步减少输入的空间尺寸(reduce the spatial size of the representation)，从而减少网络中的参数数量和计算量。
   - 控制过拟合。

1. **最大池化（Max Pooling）**：
   - 最常见的池化形式是使用2x2的滤波器，步幅为2，将输入的每个深度切片在宽度和高度上缩小为原来的一半，丢弃75%的激活值。
   - 每个最大池化操作在2x2的小区域内取最大值, take a max over 4 numbers

3. **池化层的输入输出尺寸**：
   - 输入体积尺寸为 $W_1 \times H_1 \times D_1$。
   - 输出体积尺寸为 $W_2 \times H_2 \times D_2$，其中：
     $$ W_2 = \frac{W_1 - F}{S} + 1 $$
     $$ H_2 = \frac{H_1 - F}{S} + 1 $$
     $$ D_2 = D_1 $$
   - $F$ 是滤波器的空间范围，$S$ 是步幅。

3. **Max pooling的常见变体**：
   - $F=3, S=2$（称为重叠池化, overlapping pooling）。
   - $F=2, S=2$（最常见）。

3. **除了Max pooling外的其他池化方式**：
   - 平均池化（Average Pooling）。
   - L2范数池化（L2-norm Pooling）：历史上曾经用过，但与最大池化相比，不再受人喜欢。
   - 最大池化在实践中表现更好。

6. **反向传播**：
   - **在最大池化的反向传播中，梯度只传递给前向传播中具有最大值的输入。**
   - 通常在前向传播中记录最大激活值的索引（称为开关， switches），以便在反向传播中高效地传送梯度。

7. **去除池化层**：
   - 有些人认为可以不用池化层，例如《Striving for Simplicity: The All Convolutional Net》提出使用更大的步幅来减少表示尺寸。
   - 在生成模型（如VAE和GAN）中，去除池化层也很重要。
   - 未来的架构可能会减少或完全去除池化层。


补：池化也被称为下采样(downsampling):
### **为什么叫下采样？**

- **采样（Sampling）**：在信号处理中，采样是指从连续信号中提取离散点的过程。在深度学习中，**采样可以理解为从输入数据中提取部分信息。**
- **上采样（Upsampling）**：增加特征图的分辨率，通常用于生成高分辨率输出（如图像分割中的反卷积操作）
- **下采样（Downsampling）**：池化操作通过减少特征图的空间尺寸，实际上是在**降低数据的分辨率**，即从高分辨率的输入特征图中提取低分辨率的输出特征图。因此，池化被称为下采样。






####  Other Layers In CNN

1. **归一化层 (Normalization Layer)**:
   - 在卷积神经网络中曾提出多种归一化层，目的是模拟生物大脑中的抑制机制。
   - 在实践中，这些层的贡献被证明微乎其微，因此逐渐被弃用。

2. **全连接层 (Fully-connected Layer, FC Layer)**:
   - 神经元与前一层的所有激活值全连接，类似于常规神经网络。
   - 激活值通过矩阵乘法和偏置偏移计算。

3. **全连接层与卷积层的转换 (Converting FC Layers to CONV Layers)**:
   - **FC 层与 CONV 层的唯一区别**：
     - CONV 层的神经元只连接到输入体积的局部区域，并且许多神经元共享参数。
     - 两者的神经元都计算点积，因此函数形式相同。
   - **FC 层转换为 CONV 层**：
     - 例如，一个 FC 层输入体积为 $7 \times 7 \times 512$，输出为 $4096$，可以转换为 CONV 层，**滤波器尺寸 $F=7$（强调：set the filter size to be exactly the size of the input volume）**，步幅 $S=1$，零填充 $P=0$，输出体积为 $1 \times 1 \times 4096$。
   - **CONV 层转换为 FC 层**：
     - 任何 CONV 层都可以通过一个大型稀疏矩阵（由于局部连接和参数共享）实现相同的功能。

4. **FC 层转换为 CONV 层的实际应用**:
   - 例如，AlexNet 架构中，输入图像尺寸为 $224 \times 224 \times 3$，经过一系列 CONV 和 POOL 层后，输出体积为 $7 \times 7 \times 512$。
   - 将 FC 层转换为 CONV 层：
     - 第一个 FC 层（输入 $7 \times 7 \times 512$，输出 $4096$）转换为 CONV 层，滤波器尺寸 $F=7$，输出体积 $1 \times 1 \times 4096$。
     - 第二个 FC 层转换为 CONV 层，滤波器尺寸 $F=1$，输出体积 $1 \times 1 \times 4096$。
     - 最后一个 FC 层转换为 CONV 层，滤波器尺寸 $F=1$，输出体积 $1 \times 1 \times 1000$。
   - **优势**：
     - 转换后的卷积网络可以在更大的图像上高效滑动，例如 $384 \times 384$ 的图像通过转换后的网络，输出体积为 $12 \times 12 \times 512$，最终输出为 $6 \times 6 \times 1000$。
     - 相比在原图像上多次裁剪和评估，转换后的网络只需一次前向传播，计算效率更高。

5. **步幅小于 32 像素的情况**:
   - 如果需要步幅小于 32 像素（如 16 像素），可以通过多次前向传播（multiple forward passes）实现。
   - 例如，第一次在原图像上传播，第二次在图像沿宽度和高度平移 16 像素后传播，结合两次结果。

6. **实际代码实现**:
   - 使用 Caffe 进行 FC 层到 CONV 层的转换，具体实现可参考 IPython Notebook on Net Surgery。






### ConvNet Architectures

### **1. 卷积神经网络的常见层类型**
- **CONV（卷积层）**：提取特征，通常与RELU激活函数结合使用。
- **POOL（池化层，默认为最大池化）**：对输入进行下采样，减少空间尺寸。
- **FC（全连接层）**：用于分类或回归任务。
- **RELU（激活函数）**：逐元素应用非线性激活函数。
---

### **2. 常见的卷积神经网络架构模式**
讨论通常如何把这几层堆叠在一起形成整个神经网络：
In this section we discuss how these are commonly stacked together to form entire ConvNets.
最常见的卷积神经网络架构模式如下：
```
INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC
```
- `*` 表示重复，`POOL?` 表示池化层是可选的。
- `N >= 0`（通常 `N <= 3`），`M >= 0`，`K >= 0`（通常 `K < 3`）。
- 示例：
  - `INPUT -> FC`：线性分类器。
  - `INPUT -> CONV -> RELU -> FC`：简单卷积网络。
  - `INPUT -> [CONV -> RELU -> POOL]*2 -> FC -> RELU -> FC`：每层池化前有一个卷积层。
  - `INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]*3 -> [FC -> RELU]*2 -> FC`：每层池化前有两个卷积层。

---

### **3. 小滤波器堆叠的优势**
- _Prefer a stack of small filter CONV to one large receptive field CONV layer_
- 堆叠多个小滤波器（如3x3）比使用单个大滤波器（如7x7）更有效：
  - 更少的参数：三个3x3卷积层的参数为 $3 \times (C \times (3 \times 3 \times C)) = 27C^2$，而单个7x7卷积层的参数为 $C \times (7 \times 7 \times C) = 49C^2$。
  - 更强的特征表达能力：堆叠的卷积层包含非线性激活函数。

---



#### **Layer Sizing Patterns**
### **1. 层尺寸设计规则**
- **输入层**：The **input layer** (that contains the image) should be divisible by 2 many times. 尺寸应能被2整除多次，常见尺寸为32、64、96、224、384、512。
- **卷积层**：
  - 使用小滤波器（如3x3， 最多5x5），步幅 $S=1$。
  - 使用零填充 $P = \frac{F-1}{2}$ 以保持输入尺寸不变。
  - 大滤波器（如7x7）通常仅用于第一层卷积。
  - **经典： F = 3， S = 1,  P = 1， retain the original size of the input**
  - 为什么卷积层通常用Stride  = 1？ 将所有空间维度下取样交给POOL层
	[Explain: Click Me!](https://cs231n.github.io/convolutional-networks/#:~:text=Why%20use%20stride%20of%201%20in%20CONV%3F%20Smaller%20strides%20work%20better%20in%20practice.%20Additionally%2C%20as%20already%20mentioned%20stride%201%20allows%20us%20to%20leave%20all%20spatial%20down-sampling%20to%20the%20POOL%20layers%2C%20with%20the%20CONV%20layers%20only%20transforming%20the%20input%20volume%20depth-wise.)
- **池化层**：
  - 最常见的是2x2最大池化，步幅 $S=2$，丢弃75%的激活值。
  - 3x3池化 + S = 2，较少使用，因为会导致信息丢失过多。


---

#### **Case studies**
### **1.经典卷积神经网络架构**
- **LeNet**：最早的卷积网络，用于手写数字识别。
- **AlexNet**：2012年ImageNet冠军，深度和规模更大。
- **ZF Net**：2013年ImageNet冠军，改进了AlexNet的超参数。
- **GoogLeNet**：2014年ImageNet冠军，引入了Inception模块，减少了参数数量。
- **VGGNet**：2014年亚军，证明了网络深度的重要性，从头到尾使用3x3卷积和2x2池化。
	- VGGNet 的全称是 **Visual Geometry Group Network**，它是由牛津大学的 Visual Geometry Group 在2014年提出的一种经典的卷积神经网络（CNN）架构。
	- 最著名的两个版本：
		- **VGG-16**：包含16层（13个卷积层 + 3个全连接层）。
		- **VGG-19**：包含19层（16个卷积层 + 3个全连接层）。
- **ResNet**：2015年ImageNet冠军，引入了残差连接（It features special skip connections and a heavy use of **batch normalization**），是目前最先进的卷积神经网络。

---
### **2. VGGNet的详细结构**
VGGNet由多个3x3卷积层和2x2池化层组成，具体结构如下：
```
输入层
- **INPUT**: [224x224x3]
卷积层
- **CONV3-64**: [224x224x64]  
- **CONV3-64**: [224x224x64]  
池化层
- **POOL2**: [112x112x64]  
卷积层
- **CONV3-128**: [112x112x128]  
- **CONV3-128**: [112x112x128]  
池化层
- **POOL2**: [56x56x128]  
卷积层
- **CONV3-256**: [56x56x256]  
- **CONV3-256**: [56x56x256]  
- **CONV3-256**: [56x56x256]  
池化层
- **POOL2**: [28x28x256]  
卷积层
- **CONV3-512**: [28x28x512]  
- **CONV3-512**: [28x28x512]  
- **CONV3-512**: [28x28x512]  
池化层
- **POOL2**: [14x14x512]  
卷积层
- **CONV3-512**: [14x14x512]  
- **CONV3-512**: [14x14x512]  
- **CONV3-512**: [14x14x512]  
池化层
- **POOL2**: [7x7x512]  
全连接层
- **FC**: [1x1x4096]  
- **FC**: [1x1x4096]  
- **FC**: [1x1x1000]  
```
一般来说： 前面的卷积层占用了大部分内存，最后一层全连接占了大部分参数。



#### **Computational Considerations**

#### 1. **内存瓶颈的来源**
在构建卷积神经网络时，**内存瓶颈**是一个需要特别注意的问题。现代 GPU 的内存通常为 3/4/6GB，高端 GPU 的内存约为 12GB。内存消耗主要来自以下三个方面：

---
#### 2. **中间激活值**
- **来源**：每一层的激活值及其梯度（大小与激活值相同）。
- **特点**：
  - 大多数激活值集中在 the earlier layers of a ConvNet（如第一个卷积层）。
  - 这些激活值在反向传播时需要保留，因此会占用大量内存。
- **优化**：在测试时，可以通过仅存储当前层的激活值并丢弃前一层的激活值来大幅减少内存占用。

---
#### 3. **参数大小**
- **来源**：网络参数、反向传播时的梯度，以及优化算法（如动量、Adagrad 或 RMSProp）的步长缓存。
- **特点**：存储参数向量所需的内存通常需要乘以至少 3 倍。

---
#### 4. **其他内存消耗**
- **来源**：图像数据批次、数据增强版本等。
- **特点**：这些内存消耗虽然较小，但也需要纳入总内存预算。

---
#### 5. **内存计算**
- **步骤**：
  1. 估算激活值、梯度和其他内存的总数量。
  2. 将总数量乘以 4（每个浮点数占 4 字节）或 8（双精度浮点数）。
  3. 将结果除以 1024 多次，转换为 KB、MB 和 GB。
- **示例**：如果总内存需求超过 GPU 内存限制，可以通过**减小批量大小**来减少内存占用，因为大多数内存通常被激活值消耗。
---

