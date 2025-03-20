---
title: "L10 - 231n Training I"
date: "2025-03-20 09:49:07"
author: "EvanZhangcn"
draft: false
categories: ["EECS498"]  # 在此编辑分类
tags: []               # 在此添加标签
weight: 10
url: "/posts/EECS498/L10 - 231n Training I"  # 自动生成的URL
---
### Data Preprocessing

这一段主要讲解了数据预处理的三种常见方法：**均值减法（Mean Subtraction）**、**归一化（Normalization）** 和 **PCA 与白化（PCA and Whitening）**。以下是主要知识点的总结：

---

### 1. **均值减法（Mean Subtraction）**

- **定义**：对每个特征减去其均值，使数据在每一维上以原点为中心。
- **实现**：
  $$
  X -= \text{np.mean}(X, \text{axis} = 0)
  $$
- **图像处理**：通常对所有像素减去一个均值，或分别对每个颜色通道处理。

---

### 2. **归一化（Normalization）**

- **定义**：将数据维度缩放到相同的尺度。
- **两种方法**：
  1. **标准差归一化**：
     $$
     X /= \text{np.std}(X, \text{axis} = 0)
     $$
  2. **最小-最大归一化**：将数据缩放到 $[-1, 1]$ 范围。
- **适用场景**：当不同特征的尺度或单位不同，但对学习算法的重要性相当时。

---

### 3. **PCA与白化（PCA and Whitening）**

- **步骤**：
  1. **零中心化**：
     $$
     X -= \text{np.mean}(X, \text{axis} = 0)
     $$
  2. **计算协方差矩阵**：
     $$
     \text{cov} = \frac{X^T X}{N}
     $$
  3. **SVD 分解**：
     $$
     U, S, V = \text{np.linalg.svd}(\text{cov})
     $$
  4. **去相关**：
     $$
     X_{\text{rot}} = X \cdot U
     $$
  5. **降维（PCA）**：
     $$
     X_{\text{rot\_reduced}} = X \cdot U[:, :k]
     $$
  6. **白化**：
     $$
     X_{\text{white}} = \frac{X_{\text{rot}}}{\sqrt{S + \epsilon}}
     $$
- **几何解释**：
  - **PCA**：将数据旋转到特征向量基，去除相关性。
  - **白化**：将数据缩放到各维度方差为 1，形成各向同性的高斯分布。

---

### 4. **注意事项**

- **噪声放大**：白化可能放大噪声，可通过增加平滑常数 $\epsilon$ 缓解。
- **预处理统计量**：均值、标准差等统计量应仅在训练数据上计算，然后应用于验证/测试数据。

---

### 5. **总结**

- **均值减法**：使数据以原点为中心。
- **归一化**：将数据缩放到相同尺度。
- **PCA 与白化**：去除数据相关性，降维并标准化方差。
- **实践建议**：在卷积网络中，通常仅进行零中心化和像素归一化。

---

这些公式和知识点可以直接复制到 Obsidian 中，方便你进一步整理和笔记！

### Weight Initialization

在训练神经网络时，我们通常不会预先知道每个权重的最终值。然而，通过适当的数据归一化（normalization），我们可以合理地假设大约一半的权重会是正数，另一半会是负数。这是因为：

1. **数据归一化**：归一化将输入数据调整到一个标准范围内（例如，均值为0，标准差为1）。这样，输入数据的分布更加对称，减少了极端值的影响，使得权重在训练过程中更有可能均匀分布在正负之间。
2. **权重初始化**：在训练开始时，权重通常会被随机初始化，且这些初始值通常是对称分布的（例如，均值为0的正态分布）。**这种对称性在训练过程中会被保持，尤其是在数据已经归一化的情况下。**

### 1. 全零初始化的陷阱   All zero initialization

- **问题**：如果将所有权重初始化为零，所有神经元会计算相同的输出和梯度，导致参数更新完全一致，无法打破对称性。
- **结论**：全零初始化是错误的，因为它会导致神经元之间缺乏差异性。

---

### 2. **小随机数初始化   Small random numbers.**

- **方法**：将权重初始化为接近零(但不等于0)的小随机数，打破对称性，这种做法称为“symmetry breaking”.
- **一种可能的实现如下**：

  $$
  W = 0.01 \times \text{np.random.randn}(D, H)
  $$

  其中，`randn` 从均值为 0、标准差为 1 的高斯分布中采样。
- **优点**：神经元在初始时具有随机性，能够计算不同的更新，逐渐形成多样化的网络。
- **缺点**：_Warning_: It’s not necessarily the case that smaller numbers will work strictly better.[Explain: Click Me!](http://cs231n.github.io/neural-networks-2/#init#:~:text=For%20example%2C%20a%20Neural%20Network%20layer%20that%20has%20very%20small%20weights%20will%20during%20backpropagation%20compute%20very%20small%20gradients%20on%20its%20data%20(since%20this%20gradient%20is%20proportional%20to%20the%20value%20of%20the%20weights).%20This%20could%20greatly%20diminish%20the%20%E2%80%9Cgradient%20signal%E2%80%9D%20flowing%20backward%20through%20a%20network%2C%20and%20could%20become%20a%20concern%20for%20deep%20networks.)

---

### 3. **方差校准与 $1/\sqrt{n}$ 初始化**   Calibrating the variances with 1/sqrt(n)

##### **问题描述**

上面的建议有有这样的问题:在神经网络中，神经元的输出 $s$ 是输入 $x_i$ 和权重 $w_i$ 的加权和：

$$
s = \sum_{i=1}^{n} w_i x_i
$$

如果权重 $w_i$ 是随机初始化的（例如，从均值为0、方差为1的正态分布中采样），那么输出 $s$ 的方差会随着输入数量 $n$ 的增加而增大。这是因为：

$$
\text{Var}(s) = n \cdot \text{Var}(w) \cdot \text{Var}(x)
$$

如果 $\text{Var}(w) = 1$（即权重初始化的方差为1），那么 $\text{Var}(s)$ 会与输入数量 $n$ 成正比。这意味着，当输入数量 $n$ 很大时，输出 $s$ 的方差会变得非常大。随机初始化神经元的输出分布(the distribution of the outputs)的方差会随输入数量增加而增大。

##### **主要后果之一**

**梯度爆炸或梯度消失**：

- 如果输出 $s$ 的方差过大，激活函数的输入可能会落在饱和区（例如，Sigmoid函数的极端值区域），导致梯度变得非常小（梯度消失）。
- 如果输出 $s$ 的方差过小，梯度可能会变得非常大（梯度爆炸）。
- 这两种情况都会导致训练过程变得不稳定，甚至无法收敛。

##### **解决办法**

- **解决方法**：将每个神经元的权重向量缩放为 $1/\sqrt{n}$，其中 $n$ 是输入数量。(这个方法也称为： Xavier 初始化  , 不同于He初始化)
- **公式**：

  $$
  w = \frac{\text{np.random.randn}(n)}{\sqrt{n}}
  $$
- **推导过程如下**：[Explain: Click Me!](http://cs231n.github.io/neural-networks-2/#init#:~:text=The%20sketch%20of%20the%20derivation%20is%20as%20follows%3A%20)
  考虑神经元激活值 $s = \sum_{i=1}^{n} w_i x_i$，其方差为：

  $$
  \text{Var}(s) = n \cdot \text{Var}(w) \cdot \text{Var}(x)
  $$

  如果我们想让输出s和输入x有相同的方差， $\text{Var}(s) = \text{Var}(x)$，初始化的时候需要确保每一个权重w的方差为： $\text{Var}(w) = \frac{1}{n}$。

---

### 4. **ReLU 神经网络的初始化**

- **方法**：特别针对 ReLU 激活函数，推荐使用：[Explain: Click Me!](http://cs231n.github.io/neural-networks-2/#init#:~:text=derives%20an%20initialization%20specifically%20for%20ReLU%20neurons)
  $$
  w = \text{np.random.randn}(n) \times \sqrt{\frac{2.0}{n}}
  $$
- **依据**：He et al. 的论文表明，ReLU 神经元的方差应为 $2.0/n$。

---

### 5. **稀疏初始化  Sparse initialization**

- **方法**：将权重矩阵初始化为零，但每个神经元随机连接到固定数量的下层神经元（权重从小高斯分布中采样）。
- **典型值**：每个神经元连接到 10 个下层神经元。

---

### 6. **偏置初始化**

- **方法**：通常将偏置初始化为零，因为the small random numbers in the weights已经打破了对称性。
- **ReLU 的特殊情况**：有些人喜欢将偏置初始化为小常数（如 0.01），以确保 ReLU 单元在初始时激活，但这种方法的效果并不一致。所以还是推荐初始化为0

---

### 7. **批量归一化（Batch Normalization）**

- **为什么需要**：在深度神经网络中，每一层的输入分布会随着前一层权重的更新而发生变化，这种现象被称为**Internal Covariate Sift**。这种分布的变化会导致训练过程变得不稳定，需要更小的学习率和更谨慎的权重初始化。
- **Batch Normalization**通过强制每一层的**输入分布**保持稳定(例如，均值为0，方差为1)，解决了这个问题，从而加速训练并提高模型的健壮性。
- **作用**：通过强制网络中的激活值在训练初期服从单位高斯分布，缓解初始化问题。[Explain: Click Me!](http://cs231n.github.io/neural-networks-2/#init#:~:text=properly%20initializing%20neural%20networks%20by%20explicitly%20forcing%20the%20activations%20throughout%20a%20network%20to%20take%20on%20a%20unit%20gaussian%20distribution%20at%20the%20beginning%20of%20the%20training)
- **实现**：在全连接层或卷积层后、非线性激活函数前插入 BatchNorm 层。[Explain: Click Me!](http://cs231n.github.io/neural-networks-2/#init#:~:text=In%20the%20implementation%2C%20applying%20this%20technique%20usually%20amounts%20to%20insert%20the%20BatchNorm%20layer%20immediately%20after%20fully%20connected%20layers%20(or%20convolutional%20layers%2C%20as%20we%E2%80%99ll%20soon%20see)%2C)
- **优点**：In practice networks that use Batch Normalization are significantly more robust to bad initialization.可以理解为，对于神经网络的每一层都做了预处理，相当于在每一层进行可微分的预处理。

#### **公式**

对于一个 mini-batch 的输入 $x$，Batch Normalization 的计算步骤如下：

1. **计算 mini-batch 的均值和方差**：

   $$
   \mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i
   $$

   $$
   \sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2
   $$

   其中，$m$ 是 mini-batch 的大小。
2. **归一化**：

   $$
   \hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
   $$

   其中，$\epsilon$ 是一个很小的常数（例如 $10^{-5}$），用于防止除以零。
3. **缩放和平移**：

   $$
   y_i = \gamma \hat{x}_i + \beta
   $$

   其中，$\gamma$ 和 $\beta$ 是可学习的参数，用于恢复网络的表达能力（因为归一化可能会丢失一些信息）。

### **Batch Normalization 的实现**

**在神经网络中，Batch Normalization 通常作为一个独立的层插入到全连接层或卷积层之后，激活函数之前。** 具体实现步骤如下：

1. **前向传播**：

   - 对每个 mini-batch 计算均值和方差。
   - 对输入数据进行归一化。
   - 使用可学习的参数 $\gamma$ 和 $\beta$ 进行缩放和平移。
2. **反向传播**：

   - 通过链式法则计算 $\gamma$ 和 $\beta$ 的梯度，并更新它们的值。
3. **推理阶段**：

   - 在推理阶段，使用整个训练集的均值和方差（通常通过移动平均计算）来归一化数据，而不是 mini-batch 的统计量。

### **Batch Normalization 的优点**

1. **加速训练**：通过稳定输入分布，可以使用更大的学习率，从而加速训练。
2. **减少对初始化的依赖**：网络对权重初始化的敏感性降低，训练更加稳定。
3. **正则化效果**：由于每个 mini-batch 的统计量不同，Batch Normalization 具有一定的正则化效果，可以减少过拟合。
4. **允许更深的网络**：通过缓解梯度消失问题，Batch Normalization 使得训练更深的网络成为可能。

### **Batch Normalization 的缺点**

1. **增加计算开销**：每个 mini-batch 都需要计算均值和方差，增加了计算量。
2. **对 batch size 敏感**：当 batch size 较小时，统计量的估计可能不准确，影响性能。

### **代码示例（PyTorch）**

在 PyTorch 中，Batch Normalization 可以通过 `torch.nn.BatchNorm1d`（用于全连接层）或 `torch.nn.BatchNorm2d`（用于卷积层）实现：

```python
import torch
import torch.nn as nn

# 定义一个简单的网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)  # Batch Normalization 层
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  # 应用 Batch Normalization
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# 实例化网络
model = Net()
```

为什么A3作业前半部分一直手动初始化权重矩阵，之后随着网络复杂不让你这么做了？

- **手动调整权重初始化**：在早期，研究人员通常手动调整权重的初始值，例如将权重初始化为一个固定的范围（如$[-0.1, 0.1]$ ）。然而，这种方法在深度神经网络中效率低下，**因为随着网络层数的增加，权重的分布需要更加精细的控制。**
- **权重矩阵的大小与权重尺度的关系**：在深度神经网络中，权重矩阵的规模通常很大（例如，全连接层的权重矩阵维度为$[n, m]$ ，其中 $n$ 和 $m$ 是输入和输出的维度）。**随着权重矩阵规模的增大，权重的初始值需要更小，以避免输出方差过大或过小。**

### 8.**Kaiming初始化**

Kaiming初始化(也称为 He初始化)是由何凯明（Kaiming He）等人提出的一种权重初始化方法，专门用于解决深度神经网络中的初始化问题。

它的核心思想是根据输入神经元的数量（$n$）来调整权重的初始值，使得**每一层的输出方差**保持一致。

#### **公式**

对于使用 ReLU 激活函数的网络，Kaiming初始化的公式为：

$$
w \sim \mathcal{N}(0, \frac{2}{n})
$$

其中：

- $w$ 是权重矩阵。
- $\mathcal{N}(0, \frac{2}{n})$ 表示从均值为0、方差为 $\frac{2}{n}$ 的正态分布中采样。
- $n$ 是输入神经元的数量。
  对于其他激活函数（如 Leaky ReLU），公式可能会有所不同。

---

#### **`nn.init.kaiming_normal_` 的具体实现**

`nn.init.kaiming_normal_` 函数会根据 Kaiming初始化的公式，从均值为0、方差为 $\frac{2}{n}$ 的正态分布中采样权重值。具体步骤如下：

1. **计算方差**：

   - 如果 `mode='fan_in'`，则方差为 $\frac{2}{n}$，其中 $n$ 是输入神经元的数量。
   - 如果 `mode='fan_out'`，则方差为 $\frac{2}{m}$，其中 $m$ 是输出神经元的数量。
   - 对于 ReLU 激活函数，公式中的系数为2；对于其他激活函数（如 Leaky ReLU），系数可能会有所不同。
2. **从正态分布中采样**：

   - 根据计算出的方差，从均值为0的正态分布中采样权重值。
3. **更新权重矩阵**：

   - 将采样得到的权重值赋值给 `self.fc1.weight`。

---

### **为什么 Kaiming初始化有效？**

1. **控制输出方差**：通过将权重的方差设置为 $\frac{2}{n}$，Kaiming初始化可以确保每一层的输出方差保持一致，从而避免梯度消失或梯度爆炸。
2. **适应 ReLU 激活函数**：ReLU 激活函数会将负值置为0，这会导致输出的方差减半。Kaiming初始化通过将方差加倍（$\frac{2}{n}$）来补偿这种效应。
3. **自动化权重初始化**：Kaiming初始化提供了一种自动化的方法，无需手动调整权重尺度，特别适合深度神经网络。

---

### **实现**

在 PyTorch 中，Kaiming初始化可以通过 `torch.nn.init.kaiming_normal_` 或 `torch.nn.init.kaiming_uniform_` 实现。例如：

```python
import torch
import torch.nn as nn

# 定义一个简单的网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self._initialize_weights()

    def _initialize_weights(self):
        # 使用 Kaiming初始化
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

# 实例化网络
model = Net()
```

---

### Regularization

There are several ways of **controlling the capacity of Neural Networks** to prevent overfitting.

### 1. **L2 正则化**

- **定义**：在目标函数中增加所有权重的平方和，公式为：

  $$
  \frac{1}{2} \lambda w^2
  $$

  其中，$\lambda$ 是正则化强度。
- 注意Data100课程中前面没有1/2这个系数，EECS498大部分作业中也没有这个系数。
- 为什么喜欢前面多加一个系数？求导结果好看。It is common to see the factor of $\frac{1}{2}$ in front because then the gradient of this term with respect to the parameter $w$ is simply $\lambda w$ instead of $2\lambda w$.
- **梯度更新**：权重更新时线性衰减：using the L2 regularization ultimately means that **every weight is decayed linearly**

  $$
  W += -\lambda \cdot W
  $$
- **作用**：惩罚峰值权重向量，鼓励网络使用所有输入，而不是过度依赖某些输入。The L2 regularization has the intuitive interpretation of heavily penalizing peaky weight vectors and preferring diffuse weight vectors.

---

### 2. **L1 正则化**

- **定义**：在目标函数中增加所有权重的绝对值，公式为：
  $$
  \lambda |w|
  $$
- **特点**：使权重向量稀疏，神经元仅使用最重要的输入。In other words, neurons with L1 regularization **end up using only a sparse subset of their most important inputs** and become nearly invariant to the “noisy” inputs.
- 相比之下，采用L2正则化，得到的权重矩阵通常数值更小，且更分散diffuse.
- **将L1 结合 L2**：弹性网络正则化（Elastic Net）：combine the L1 regularization with the L2 regularization
  $$
  \lambda_1 |w| + \lambda_2 w^2
  $$

---

### 3. **最大范数约束（Max Norm Constraints）**

- **纯正英文定义介绍**：[Explain: Click Me!](https://cs231n.github.io/neural-networks-2/#:~:text=Another%20form%20of%20regularization%20is%20to%20enforce%20an%20absolute%20upper%20bound%20on%20the%20magnitude%20of%20the%20weight%20vector%20for%20every%20neuron%20and%20use%20projected%20gradient%20descent%20to%20enforce%20the%20constraint.%20)
- **定义**：限制每个神经元的权重向量的范数不超过某个预先设定的阈值 $c$（通常在3或4左右）：
  $$
  \| \vec{w} \|_2 < c
  $$
- **实现方式**：[Explain: Click Me!](https://cs231n.github.io/neural-networks-2/#:~:text=In%20practice%2C%20this,4)
  1. **参数更新**：首先，像平常一样使用梯度下降或其他优化算法更新神经网络的权重参数。
  2. **约束裁剪**：在更新完权重后，检查每个神经元的权重向量 $w⃗$ 的 $L_2$ 范数是否超过了 $c$。如果超过了，就将权重向量 $w⃗$ 进行缩放，使其 $L_2$ 范数等于 $c$。这个过程称为“裁剪（clamping）”或“投影”。
- **优点**：防止权重爆炸，即使学习率设置过高也能稳定训练。

---

### 4. **Dropout**

- **定义**：在训练时，以概率 $p$ 随机将神经元输出置零。（部分资料认为：p代表the probability of keeping a neuron output, 此时p越大，保留的神经元越多）
- **对应英文定义：**[Explain: Click Me!](https://cs231n.github.io/neural-networks-2/#:~:text=While%20training%2C%20dropout,otherwise.)

#### **Vanilla Dropout**

Vanilla Dropout 的实现方式如下：

1. **训练阶段**：

   - 在每一层中，随机生成一个二值掩码 $U$，掩码中的每个元素以概率 $p$ 为 1，以概率 $1-p$ 为 0。
   - 将掩码 $U$ 与激活值相乘，实现“丢弃”操作。
   - 例如，对于第一层隐藏层：
   - 而且强调：对于反向传播阶段的操作是The backward pass remains unchanged, but of course has to take into account the generated masks `U1,U2`.[Explain: Click Me!](https://cs231n.github.io/neural-networks-2/#:~:text=The%20backward%20pass%20remains%20unchanged%2C%20but%20of%20course%20has%20to%20take%20into%20account%20the%20generated%20masks%20U1%2CU2.)

     $$
     H_1 = \text{ReLU}(W_1 X + b_1)
     $$

     $$
     U_1 = (\text{rand}(*H_1.\text{shape}) < p)
     $$

     $$
     H_1 *= U_1
     $$
2. **预测阶段**：

   - **不再进行丢弃，但对激活值进行缩放，以保持训练和测试时输出的期望一致。**
   - 例如，对于第一层隐藏层：
     $$
     H_1 = \text{ReLU}(W_1 X + b_1) * p
     $$
3. **数学原理**：[Explain: Click Me!](https://cs231n.github.io/neural-networks-2/#:~:text=This%20is%20important%20because%20at%20test%20time%20all%20neurons%20see%20all%20their%20inputs%2C%20so%20we%20want%20the%20outputs%20of%20neurons%20at%20test%20time%20to%20be%20identical%20to%20their%20expected%20outputs%20at%20training%20time.)

   - 训练时，神经元的输出 $x$ 被丢弃的概率为 $1-p$，因此其期望输出为：
     $$
     E[x] = p \cdot x + (1-p) \cdot 0 = p x
     $$
   - 测试时，为了保持期望一致，需要对输出进行缩放：
     $$
     x \rightarrow p x
     $$

#### **Inverted Dropout（推荐实现）**

对于普通版本的Dropout， 不太好的一个特点就是：必须在测试阶段\*p， 而测试阶段的表现很重要。因此我们采用另一个版本

**Inverted Dropout 在训练阶段进行缩放，测试阶段无需额外操作，实现更简洁。**

具体代码实现（看清楚概率p的含义）：[Explain: Click Me!](https://cs231n.github.io/neural-networks-2/#:~:text=%22%22%22%20,%22%22%22)

1. **训练阶段**：

   - 生成掩码时，直接对掩码进行缩放：

     $$
     U_1 = (\text{rand}(*H_1.\text{shape}) < p) / p
     $$
   - 例如，对于第一层隐藏层：

     $$
     H_1 = \text{ReLU}(W_1 X + b_1)
     $$

     $$
     U_1 = (\text{rand}(*H_1.\text{shape}) < p) / p
     $$

     $$
     H_1 *= U_1
     $$
2. **预测阶段**：

   - 无需进行任何缩放操作：
     $$
     H_1 = \text{ReLU}(W_1 X + b_1)
     $$

#### **Dropout 的优点**

1. 减少过拟合，增强模型泛化能力。
2. 可以看作是对指数级子网络的集成学习。
3. 实现简单，计算开销小。

---

### 5. **Dropout 的理论解释**

- **训练时**：Dropout 可以看作是从完整网络中采样一个子网络进行训练。
- **测试时**：评估所有子网络的集成预测，相当于对激活值进行期望值调整：
  $$
  \text{期望输出} = p \cdot x + (1 - p) \cdot 0 = p \cdot x
  $$

---

### 6. **其他正则化方法**

- **偏置正则化(Bias regularization)**：通常不对偏置进行正则化，因为偏置不通过乘法与数据交互，and therefore do not have the interpretation of controlling the influence of a data dimension on the final objective.
- **分层正则化(Per-layer regularization)**：不同层使用不同的正则化强度并不常见。

---

### 7. **实践建议**

- **L2 正则化**：通常使用全局 L2 正则化强度，并通过交叉验证调整。
- **Dropout**：默认 $p = 0.5$，可在验证数据上调整。
- **结合使用**：L2 正则化与 Dropout 结合使用效果最佳。

---

### Loss functions

这一段主要讲解了监督学习中的**数据损失（Data Loss）**，以及在不同任务（如分类、回归、结构化预测）中常用的损失函数。

### 1. **数据损失的定义**

- **公式**：数据损失是每个样本损失的平均值：

  $$
  L = \frac{1}{N} \sum_{i} L_i
  $$

  其中，$N$ 是训练数据的数量，$L_i$ 是第 $i$ 个样本的损失。

---

### 2. **分类任务**

- **SVM 损失（Hinge Loss）**：

  $$
  L_i = \sum_{j \neq y_i} \max(0, f_j - f_{y_i} + 1)
  $$

  其中，$f_j$ 是第 $j$ 类的得分，$f_{y_i}$ 是正确类别的得分。
- **平方 Hinge Loss**：

  $$
  L_i = \sum_{j \neq y_i} \max(0, f_j - f_{y_i} + 1)^2
  $$
- **Softmax 损失（交叉熵损失）**：

  $$
  L_i = -\log \left( \frac{e^{f_{y_i}}}{\sum_j e^{f_j}} \right)
  $$
- **大规模分类问题**：

  - **分层 Softmax**：将类别组织成树结构，每个节点训练一个 Softmax 分类器。

---

### 3. **属性分类任务**

- **多标签分类**：每个样本可能有多个标签，使用独立的二分类器。
- **Hinge Loss**：

  $$
  L_i = \sum_j \max(0, 1 - y_{ij} f_j)
  $$

  其中，$y_{ij}$ 是第 $i$ 个样本在第 $j$ 个属性上的标签（$+1$ 或 $-1$）。
- **Logistic 回归损失**：

  $$
  L_i = -\sum_j \left( y_{ij} \log(\sigma(f_j)) + (1 - y_{ij}) \log(1 - \sigma(f_j)) \right)
  $$

  其中，$\sigma$ 是 Sigmoid 函数。

---

### 4. **回归任务**

- **L2 损失（均方误差）**：
  $$
  L_i = \| f - y_i \|_2^2
  $$
- **L1 损失（绝对误差）**：
  $$
  L_i = \| f - y_i \|_1 = \sum_j | f_j - (y_i)_j |
  $$
- **注意事项**：
  - L2 损失对异常值敏感，优化难度较大。
  - 优先考虑将输出离散化，使用分类任务代替回归任务。

---

### 5. **结构化预测任务**

- **定义**：标签是复杂结构（如图、树等），通常使用结构化 SVM 损失。
- **核心思想**：要求正确结构与最高分错误结构之间有一定的间隔。
- **优化方法**：通常使用特殊求解器，而非简单的梯度下降。

---

### 6. **总结**

- **分类任务**：常用 SVM 损失或 Softmax 损失。
- **属性分类任务**：使用独立的二分类器，常用 Hinge Loss 或 Logistic 回归损失。
- **回归任务**：常用 L2 或 L1 损失，但优先考虑离散化输出。
- **结构化预测任务**：使用结构化 SVM 损失，通常需要特殊求解器。
