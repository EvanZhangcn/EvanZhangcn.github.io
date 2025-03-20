---
title: "L11 - 231n Training II"
date: "2025-03-20 09:49:07"
author: "EvanZhangcn"
draft: false
categories: ["未分类"]  # 在此编辑分类
tags: []               # 在此添加标签
url: "/posts/EECS498/L11 - 231n Training II"  # 自动生成的URL
---


### Gradient Checks

### 梯度检查（Gradient Checking）的主要知识点总结

#### 1. 使用中心差分公式
- **公式**：
  - 不推荐使用：$$\frac{df(x)}{dx} = \frac{f(x+h) - f(x)}{h}$$
  - 推荐使用：$$\frac{df(x)}{dx} = \frac{f(x+h) - f(x-h)}{2h}$$
- **原因**：中心差分公式的误差为$O(h^2)$，比前向差分公式的误差$O(h)$更精确。

#### 2. 使用相对误差进行比较
- **公式**：$$\text{相对误差} = \frac{|f'_a - f'_n|}{\max(|f'_a|, |f'_n|)}$$
- **误差范围**：
  - 相对误差 > 1e-2：梯度可能错误
  - 1e-2 > 相对误差 > 1e-4：可能有问题
  - 1e-4 > 相对误差：通常可以接受
  - 1e-7 及以下：理想情况

#### 3. 使用双精度浮点数
- **原因**：单精度浮点数可能导致较高的相对误差，双精度浮点数可以显著降低误差。

#### 4. 注意浮点数的活跃范围
- **建议**：确保梯度值不在极端小的范围内（如1e-10及以下），可以通过缩放损失函数来调整。

#### 5. 目标函数中的“kinks”（非可微点）
- **问题**：如ReLU函数在$x=0$处的非可微性可能导致数值梯度与解析梯度不一致。
- **解决方法**：跟踪“winners”的变化，判断是否跨越了kink。

#### 6. 使用少量数据点
- **原因**：减少数据点可以降低kinks的数量，使梯度检查更高效。

#### 7. 步长$h$的选择
- **建议**：步长$h$不宜过小，否则可能引入数值精度问题。可以尝试调整$h$的值（如1e-4或1e-6）。

#### 8. 在“特征”模式下进行梯度检查
- **建议**：在损失开始下降后进行梯度检查，避免在初始随机参数下引入病态情况。

#### 9. 不要让正则化项主导梯度
- **建议**：先关闭正则化，单独检查数据损失的梯度，然后再检查正则化项的梯度。

#### 10. 关闭Dropout和数据增强
- **原因**：非确定性操作（如Dropout）会引入误差，建议在梯度检查时关闭这些操作。

#### 11. 只检查部分维度
- **建议**：对于大规模参数，只检查部分维度的梯度，但要确保每个参数都得到正确的梯度。


### Before learning: sanity checks Tips/Tricks

### 优化前的几个基本检查（Sanity Checks）

#### 1. 检查初始损失是否符合预期
- **目的**：确保在参数初始化较小时，损失值符合预期。
- **步骤**：
  - 首先关闭正则化（设置正则化强度为0），单独检查数据损失。
  - **例子**：
    - 对于CIFAR-10的Softmax分类器，初始损失应为2.302，因为每个类的概率为0.1（共10个类），Softmax损失是正确类别的负对数概率：$$-\ln(0.1) = 2.302$$
    - 对于Weston Watkins SVM，由于所有分数接近0，所有期望的边界都被违反，因此损失应为9（每个错误类别的边界为1）。
- **问题**：如果初始损失不符合预期，可能是初始化有问题。

#### 2. 增加正则化强度应增加损失
- **目的**：验证正则化是否正常工作。
- **步骤**：逐步增加正则化强度，观察损失是否随之增加。
- **问题**：如果损失没有增加，可能是正则化实现有误。

#### 3. 在小数据集上过拟合
- **目的**：确保模型能够在小数据集上达到零损失。
- **步骤**：
  - 使用非常小的数据集（如20个样本）进行训练，关闭正则化（设置正则化强度为0）。
  - 确保模型能够过拟合并达到零损失。
- **注意**：
  - 即使在小数据集上过拟合成功，仍可能存在实现错误。例如，如果数据特征由于某些错误是随机的，模型可能在小数据集上过拟合，但在完整数据集上无法泛化。
- **建议**：在未通过此检查之前，不要继续在完整数据集上训练。




### Babysitting the learning process
在深度学习中，一旦通过反向传播计算出解析梯度，这些梯度将被用于参数更新。以下是几种常见的参数更新方法及其主要知识点：




### Parameter updates
#### 1. **Vanilla Update（普通更新）**
这是最简单的更新形式，参数沿着负梯度方向进行更新。公式如下：
$$x += - \text{learning\_rate} * dx$$
其中，`learning_rate` 是一个超参数，通常是一个固定的常数。
当在完整数据集上评估，学习率足够低时，这种方法可以保证在损失函数上取得非负的进展。

#### 2. **Momentum Update（动量更新）**
动量更新的梯度更新方法通常能带来更好的收敛速度。
1. **物理背景**：
   - 损失函数可以被视为一个丘陵地形的高度，相当于势能（$U = mgh$，因此 $U \propto h$）。
   - 参数初始化相当于将一个初始速度为零的粒子放置在某个位置。
   - **速度的积累**：在物理中，粒子在滚动时会积累速度。如果地形在某一个方向上持续有梯度（即持续有下坡），粒子会加速。
   - **摩擦力的作用**：在现实中，粒子不会无限加速，因为存在摩擦力。摩擦力会逐渐减慢粒子的速度。
   - **动量更新中的** ：
    -  $\mu$是一个超参数，通常称为动量（momentum），但其物理意义更接近**摩擦系数**。
    -  $\mu * v$表示当前速度的一部分被保留下来，类似于摩擦力对速度的阻尼作用。
    - 如果没有$\mu * v$，粒子会完全依赖当前的梯度来更新位置，相当于没有积累速度的过程，这会降低优化的效率。
   - 优化过程可以看作是模拟参数向量（即粒子）在丘陵地形上滚动的过程。
   动量更新的公式如下：
   $$v = \mu * v - \text{learning\_rate} * dx$$
   $$x += v$$
   其中，`v` 是速度变量，初始化为零，$\mu$ 是动量系数（通常取值在 [0.5, 0.9, 0.95, 0.99] 之间）。动量更新使得参数向量在梯度一致的方向上积累速度。

2. **动量的作用**：
   - 动量更新使得参数向量在梯度一致的方向上积累速度，从而加速收敛。
   - 动量系数的典型值通常在 $[0.5, 0.9, 0.95, 0.99]$ 之间，可以通过交叉验证选择最佳值。
   - 有时，动量的调度（momentum scheduling）也会带来优化效果，例如在训练后期逐渐增加动量值（如从 0.5 增加到 0.99）。


### 3. **Nesterov Momentum（Nesterov 动量更新）**
Nesterov 动量是动量更新的一个变种，具有更强的理论收敛保证，并且在实践中通常表现更好。

**在标准动量更新中，动量项会推动参数向量向某个方向移动。因此，在计算梯度时，可以先考虑参数向量即将移动到的位置，而不是当前的位置。即在“前瞻lookahead”位置计算梯度。**

1. **公式如下**：
$$x_{\text{ahead}} = x + \mu * v$$
$$v = \mu * v - \text{learning\_rate} * dx_{\text{ahead}}$$
$$x += v$$
具体来说，如果当前参数向量为 $x$，动量项会将其推向 $x + \mu v$。因此，Nesterov动量建议在计算梯度时，使用这个“前瞻”位置 $x + \mu v$，而不是当前的位置 $x$。

为了与普通动量更新形式一致，通常会将更新公式改写为：
$$v_{\text{prev}} = v$$
$$v = \mu * v - \text{learning\_rate} * dx$$
$$x += -\mu * v_{\text{prev}} + (1 + \mu) * v$$
----
如何推导改写的？
#### 变量变换的推导

在标准动量更新中，我们是在当前位置计算梯度。而在Nesterov动量中，我们是在前瞻位置计算梯度。为了使这两种方法的代码实现更一致，我们进行变量变换，将我们存储的参数设为前瞻版本 $x_{ahead}$，并重命名为 $x'$。

##### 1. 确定变量变换

令 $x' = x_{ahead} = x + \mu \cdot v$，即我们存储的是前瞻参数。

##### 2. 求解原始参数 $x$ 与新参数 $x'$ 的关系

从定义可得：
$$x = x' - \mu \cdot v$$

##### 3. 推导新的更新规则

首先，让我们从原始公式开始：

$$v_{new} = \mu \cdot v - \text{learning\_rate} \cdot dx_{ahead}$$
$$x_{new} = x + v_{new}$$

##### 4. 将 $x_{new}$ 用 $x'$ 表示

我们需要计算下一步的 $x'_{new}$：

$$x'_{new} = x_{new} + \mu \cdot v_{new}$$

将 $x_{new} = x + v_{new}$ 代入上面的等式：

$$x'_{new} = x + v_{new} + \mu \cdot v_{new}$$
$$x'_{new} = x + (1 + \mu) \cdot v_{new}$$

现在将 $x = x' - \mu \cdot v$ 代入：

$$x'_{new} = x' - \mu \cdot v + (1 + \mu) \cdot v_{new}$$
$$x'_{new} = x' - \mu \cdot v + (1 + \mu) \cdot v_{new}$$

##### 5. 得出最终的更新规则

现在我们把 $v_{new} = \mu \cdot v - \text{learning\_rate} \cdot dx_{ahead}$ 代入：

$$x'_{new} = x' - \mu \cdot v + (1 + \mu) \cdot (\mu \cdot v - \text{learning\_rate} \cdot dx_{ahead})$$
$$x'_{new} = x' - \mu \cdot v + (1 + \mu) \cdot \mu \cdot v - (1 + \mu) \cdot \text{learning\_rate} \cdot dx_{ahead}$$
$$x'_{new} = x' - \mu \cdot v + \mu \cdot v + \mu^2 \cdot v - (1 + \mu) \cdot \text{learning\_rate} \cdot dx_{ahead}$$
$$x'_{new} = x' + \mu^2 \cdot v - (1 + \mu) \cdot \text{learning\_rate} \cdot dx_{ahead}$$

这是新参数 $x'$ 的完整更新规则。但在实际代码实现中，我们希望简化表达式。

##### 6. 转换为代码实现形式

在代码实现中，我们通常会按如下方式更新：

1. 首先保存旧的速度：$v_{prev} = v$
2. 然后更新速度：$v = \mu \cdot v - \text{learning\_rate} \cdot dx$
   - 注意这里的 $dx$ 是在 $x'$ 点计算的梯度，即原始的 $dx_{ahead}$
3. 最后更新参数：$x' += -\mu \cdot v_{prev} + (1 + \mu) \cdot v$

让我们验证这个实现是否等价于我们推导的更新规则：

$$x'_{new} = x' - \mu \cdot v_{prev} + (1 + \mu) \cdot v$$
$$x'_{new} = x' - \mu \cdot v_{prev} + (1 + \mu) \cdot (\mu \cdot v_{prev} - \text{learning\_rate} \cdot dx)$$
$$x'_{new} = x' - \mu \cdot v_{prev} + (1 + \mu) \cdot \mu \cdot v_{prev} - (1 + \mu) \cdot \text{learning\_rate} \cdot dx$$
$$x'_{new} = x' - \mu \cdot v_{prev} + \mu \cdot v_{prev} + \mu^2 \cdot v_{prev} - (1 + \mu) \cdot \text{learning\_rate} \cdot dx$$
$$x'_{new} = x' + \mu^2 \cdot v_{prev} - (1 + \mu) \cdot \text{learning\_rate} \cdot dx$$

这与我们之前推导的结果一致。因此，最终的代码实现如下：

```python
v_prev = v  # 备份当前速度
v = mu * v - learning_rate * dx  # 速度更新保持不变
x += -mu * v_prev + (1 + mu) * v  # 位置更新形式改变
```

其中，$x$ 实际上是存储的前瞻参数（即原始公式中的 $x_{ahead}$），而 $dx$ 是在当前 $x$ 点计算的梯度。
这种变换的好处是，我们不再需要显式计算前瞻点，减少了一次参数复制操作，同时使Nesterov动量的实现与标准动量的实现更为一致。



2. **代码如下**：
   - 原始的Nesterov动量更新公式如下：
     ```
     x_ahead = x + mu * v
     dx_ahead = gradient at x_ahead
     v = mu * v - learning_rate * dx_ahead
     x += v
     ```
   - 为了与标准动量更新保持一致，通常会对公式进行变量变换，将参数向量存储为“前瞻”版本 $x_{\text{ahead}}$，并将其重命名为 $x$。最终的更新公式为：
     ```
     v_prev = v  # 备份当前速度
     v = mu * v - learning_rate * dx  # 速度更新保持不变
     x += -mu * v_prev + (1 + mu) * v  # 位置更新形式改变
     ```

![[L11 - 231n Training II by_2025-03-09.png]]



#### Annealing the learning rate（退火学习率）
在训练深度网络时，通常需要随着时间的推移逐渐降低学习率（学习率衰减）。以下是学习率衰减的主要知识点：

### 1. **学习率衰减的直觉**
- **高学习率**：系统包含过多的动能，参数向量会在损失函数中混沌地跳动，无法稳定地进入更深、更窄的区域。
- **低学习率**：系统冷却过快，可能无法达到最佳位置。
- **关键点**：学习率衰减的时机很重要。衰减过慢会导致计算资源浪费，衰减过快则可能无法找到最优解。

### 2. **常见的学习率衰减方法**
#### **Step Decay（步进衰减）**
每隔几个 epoch 将学习率减少一个固定比例。例如：
- 每 5 个 epoch 将学习率减半。
- 每 20 个 epoch 将学习率减少到 0.1 倍。

这种方法依赖于问题的类型和模型，通常可以通过观察验证误差来决定何时衰减学习率。例如，当验证误差停止改善时，将学习率减少一个常数（如 0.5）。

#### **Exponential Decay（指数衰减）**
数学形式为：
$$\alpha = \alpha_0 e^{-kt}$$
其中，$\alpha_0$ 和 $k$ 是超参数，$t$ 是迭代次数（也可以用 epoch 作为单位）。

#### **1/t Decay（1/t 衰减）**
数学形式为：
$$\alpha = \frac{\alpha_0}{1 + kt}$$
其中，$\alpha_0$ 和 $k$ 是超参数，$t$ 是迭代次数。

### 3. **实践中的选择**
- **步进衰减** 通常更受欢迎，因为其超参数（衰减比例和步进时间）更易于解释。
- 如果计算资源允许，建议选择较慢的衰减速度，并训练更长时间。

这些方法在深度学习中广泛应用，选择合适的学习率衰减策略可以显著提高模型的性能。








#### Second order methods
在深度学习中，基于牛顿法的优化方法是另一类流行的优化技术。以下是其主要知识点：

### 1. **牛顿法**
牛顿法的更新公式为：
$$x \leftarrow x - [Hf(x)]^{-1} \nabla f(x)$$
其中：
- $Hf(x)$ 是 Hessian 矩阵，表示函数的二阶偏导数矩阵。
- $\nabla f(x)$ 是梯度向量，与梯度下降法中的梯度相同。

**直觉**：Hessian 矩阵描述了损失函数的局部曲率，通过乘以 Hessian 的逆矩阵，优化过程可以在曲率较浅的方向上采取更大的步长，而在曲率较陡的方向上采取更小的步长。牛顿法的一个显著优势是**不需要学习率超参数**。

### 2. **牛顿法的局限性**
- **计算成本高**：显式计算和求逆 Hessian 矩阵在时间和空间上都非常昂贵。例如，一个包含 100 万个参数的神经网络，其 Hessian 矩阵的大小为 [1,000,000 x 1,000,000]，需要约 3725 GB 的内存。
- **存储问题**：Hessian 矩阵的存储需求极高，难以应用于大规模深度学习任务。

### 3. **拟牛顿法**
为了解决牛顿法的计算和存储问题，发展了多种拟牛顿法，其中最流行的是 **L-BFGS**。L-BFGS 通过利用梯度信息隐式地近似 Hessian 的逆矩阵，而无需显式计算完整的 Hessian 矩阵。

### 4. **L-BFGS 的局限性**
- **全数据集计算**：L-BFGS 需要在**整个训练集**上计算，而训练集可能包含数百万个样本。
- **小批量处理困难**：与 mini-batch SGD 不同，L-BFGS 在小批量数据上的应用更为复杂，目前仍是一个活跃的研究领域。

### 5. **实践中的选择**
- **大规模深度学习**：目前，L-BFGS 或其他二阶方法在大规模深度学习和卷积神经网络中并不常见。
- **主流方法**：基于（Nesterov）动量的 SGD 变体更为标准，因为它们更简单且更容易扩展。

这些方法在优化领域中具有重要意义，但在深度学习的实际应用中，一阶方法（如 SGD 及其变体）仍然是主流选择。






#### Per-parameter adaptive learning rate methods（**基于每个参数的自适应学习率方法**）
在神经网络训练中，传统的优化算法（如梯度下降法）通常对所有参数使用相同的学习率。All previous approaches we’ve discussed so far **manipulated the learning rate globally and equally** for all parameters.

### 1. **Adagrad**
Adagrad 是一种自适应学习率方法，通过累积梯度的平方来调整每个参数的学习率。其更新公式为：
$$\text{cache} += dx^2$$
$$x += - \text{learning\_rate} * \frac{dx}{\sqrt{\text{cache}} + \text{eps}}$$
其中：
- `cache` 记录每个参数的梯度平方和，which is then used to **normalize the parameter update step, element-wise**。
- `eps` 是一个平滑项the smoothing term（通常取 $1e-4$ 到 $1e-8$），用于避免除零错误(avoid division by zero)。

**特点**：
- 梯度较大的参数学习率会减小，梯度较小或更新不频繁的参数学习率会增大。
- 缺点是学习率单调递减，可能导致学习过早停止。

---

### 2. **RMSprop**
RMSprop 是对 Adagrad 的改进，使用梯度平方的移动平均来调整学习率，避免学习率单调递减(to reduce its aggressive, monotonically decreasing learning rate)。其更新公式为：
$$\text{cache} = \text{decay\_rate} * \text{cache} + (1 - \text{decay\_rate}) * (dx)^2$$
$$x += - \text{learning\_rate} * \frac{dx}{\sqrt{\text{cache}} + \text{eps}}$$
其中：
- `decay_rate` 是超参数，通常取 [0.9, 0.99, 0.999]。
- $(dx)^2$这个平方操作是为了计算梯度的二阶矩（即梯度的平方的移动平均），用于调整学习率。

**特点**：
- 学习率不再单调递减，适合处理非平稳目标函数。


当x是个矩阵，即我们要求dw， w时候该怎么办？
```python
# 逐元素平方
dw_squared = dw ** 2
# 或者
dw_squared = torch.pow(dw, 2)
```
当自变量是一个矩阵（例如$dw$）时，$dw^2$ 表示的是矩阵的**逐元素平方**，而不是对整个矩阵求和。
因为RMSprop的核心思想是为每个参数（即矩阵中的每个元素）单独调整学习率。因此，需要对每个梯度元素分别计算其平方，而不是对整个矩阵求和。这样可以实现**参数级别的自适应学习率**。

- 如果 $dw$ 是一个矩阵，例如：
  $$
  dw = \begin{bmatrix}
  a & b \\
  c & d
  \end{bmatrix}
  $$
  那么 $dw^2$ 表示的是逐元素平方：
  $$
  dw^2 = \begin{bmatrix}
  a^2 & b^2 \\
  c^2 & d^2
  \end{bmatrix}
  $$

- **不是** `torch.sum(dw * dw)`，因为这会计算矩阵的**Frobenius范数**（即所有元素的平方和），而不是逐元素平方。
---



### 3. **Adam**
Adam 结合了 RMSprop 和动量的思想，是目前推荐使用的默认优化算法。 However, it is often also worth **trying SGD+Nesterov Momentum as an alternative**.

**什么是矩(moment)?**
在统计学中，**矩（moment）是用来描述随机变量分布特征的量：**
- **First moment（一阶矩）**：是随机变量的期望值，也就是平均值。在Adam中，它表示梯度的移动平均值，用于估计梯度的方向。
- **Second moment（二阶矩）**：是随机变量的方差，表示数据的离散程度。在Adam中，它表示梯度平方的移动平均值，用于估计梯度的变化幅度。
通过结合一阶矩和二阶矩，Adam能够同时考虑梯度的方向和变化幅度，从而实现更高效的优化。


**Adam的核心思想**
Adam的核心是通过维护两个**移动平均值**来调整梯度更新的方向和步长：
- **First moment（一阶矩）**：梯度的移动平均值，类似于动量法中的动量项，用于加速收敛。
- **Second moment（二阶矩）**：梯度平方的移动平均值，类似于RMSProp中的自适应学习率，用于调整每个参数的学习率。
其简化更新公式为：
$$m = \text{beta1} * m + (1 - \text{beta1}) * dx$$
$$v = \text{beta2} * v + (1 - \text{beta2}) * (dx)^2$$
$$x += - \text{learning\_rate} * \frac{m}{\sqrt{v} + \text{eps}}$$
其中：
- `beta1` 和 `beta2` 是超参数，推荐值为`beta1 = 0.9`和`beta2 = 0.999`
- `eps` 通常取 $1e-8$。

**完整更新公式（带偏差校正 include a bias correction mechanism）**：
$$m = \text{beta1} * m + (1 - \text{beta1}) * dx$$
$$m_t = \frac{m}{1 - \text{beta1}^t}$$
$$v = \text{beta2} * v + (1 - \text{beta2}) * (dx)^2$$
$$v_t = \frac{v}{1 - \text{beta2}^t}$$
$$x += - \text{learning\_rate} * \frac{m_t}{\sqrt{v_t} + \text{eps}}$$
**参数解释**
- **learning_rate**：学习率，控制每次更新的步长。
- **beta1**：一阶矩的衰减率，通常设置为0.9，用于计算梯度的移动平均值。
- **beta2**：二阶矩的衰减率，通常设置为0.999，用于计算梯度平方的移动平均值。
- **epsilon**：一个很小的常数（如1e-8），用于避免除以零的情况。
- **m**：梯度的移动平均值（一阶矩）。**强调：m的更新公式应该用$dx$**
- **v**：梯度平方的移动平均值（二阶矩）。**强调：v的更新公式应该用$(dx)^2$**
- **t**：当前的迭代次数：**强调：`t`初始值为0，第一次调用时，使用t = 0，用完以后再+1.**
- 对于t， 每次调用Adam优化器时应该增加1 。上面涉及t的都是次方取指数运算。

**特点**：
- 结合了动量和自适应学习率的优点。
- 在实践中通常表现略微优于 RMSprop。
-  In the first few time steps the vectors `m,v` are both initialized and **therefore biased at zero**, before they fully “warm up”.

**实战代码：**
```python
def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', torch.zeros_like(w))
    config.setdefault('v', torch.zeros_like(w))
    config.setdefault('t', 0)

    next_w = None

    # Update t before any calculations
    config['t'] += 1

    # Implement the Adam update formula
    config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dw
    config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * (dw ** 2)

#这两个m_hat, v_hat并不会赋值更新config["m"]和config["v"]
    m_hat = config['m'] / (1 - config['beta1'] ** config['t']) 
    v_hat = config['v'] / (1 - config['beta2'] ** config['t'])
    
    next_w = w - config['learning_rate'] * m_hat / (torch.sqrt(v_hat) + config['epsilon'])

    return next_w, config
```
### 4. **实践建议**
- **默认选择**：Adam 是当前推荐的默认优化算法。
- **替代方案**：SGD + Nesterov Momentum 也是一个值得尝试的替代方案。
---





### Hyperparameter optimization

### 1. **常见超参数**
- **初始学习率**：训练开始时使用的学习率。
- **学习率衰减计划**：如衰减常数。
- **正则化强度**：如 L2 惩罚项、Dropout 强度。
- **其他超参数**：如动量及其调度（在自适应学习率方法中）。

---

### 2. **超参数搜索的实现**
- **长时间训练**：大型神经网络的训练时间较长，超参数搜索可能需要数天甚至数周。
- **工作节点设计**：设计一个工作节点，持续采样随机超参数并进行优化。训练过程中，工作节点会记录每个 epoch 后的验证性能，并将模型检查点（包括损失等训练统计信息）写入文件。
- **主节点设计**：主节点负责启动或终止工作节点，并检查工作节点生成的检查点，绘制训练统计信息等。

---

### 3. **验证集的使用**
- **单验证集**：在大多数情况下，使用一个足够大的验证集可以简化代码，无需进行多折交叉验证。
- **“交叉验证”**：有时人们会说“交叉验证”了某个参数，但实际上可能只使用了单个验证集。

---

### 4. **超参数范围**
- **对数尺度搜索**：对于学习率和正则化强度等超参数，建议在对数尺度上进行搜索。例如：
  $$\text{learning\_rate} = 10^{\text{uniform}(-6, 1)}$$
  这是因为学习率和正则化强度对训练动态有乘性影响。
- **原始尺度搜索**：对于 Dropout 等参数，通常在原始尺度上搜索（如 $\text{dropout} = \text{uniform}(0,1)$）。

---

### 5. **随机搜索 vs 网格搜索**
- **随机搜索更高效**：Bergstra 和 Bengio 的研究表明，随机搜索比网格搜索更高效，尤其是在某些超参数比其他超参数更重要的情况下。
- **实现更简单**：随机搜索通常也更容易实现。

---

### 6. **边界值检查**
- **避免边界值**：如果搜索到的超参数值位于搜索区间的边界（如学习率），可能需要扩展搜索范围，以避免错过更优的超参数设置。

---

### 7. **从粗到细的搜索**
- **分阶段搜索**：首先在较宽的范围内进行粗搜索（如 $10^{[-6, 1]}$），然后根据结果缩小范围。
- **短时训练**：初始粗搜索时可以只训练 1 个 epoch 或更少，以快速排除无效的超参数设置。
- **详细搜索**：在最终范围内进行详细搜索，训练更多 epoch。

---

### 8. **贝叶斯超参数优化**
- **核心思想**：在探索和利用之间找到平衡，以更高效地搜索超参数空间。
- **常用工具**：如 Spearmint、SMAC 和 Hyperopt。
- **实践建议**：在卷积神经网络中，随机搜索在精心选择的区间内通常仍然是最有效的。




## Evaluation
模型集成（Model Ensembles）是一种通过组合多个独立模型的预测结果来提升神经网络性能的有效方法。以下是模型集成的主要知识点：

---

### 1. **模型集成的基本思想**
- **性能提升**：通过训练多个独立模型并在测试时平均它们的预测结果，通常可以将性能提升几个百分点。
- **性能趋势**：随着集成模型中模型数量的增加，性能通常会单调提升（但收益递减）。
- **模型多样性**：集成模型中模型的多样性越高，性能提升越显著。

---

### 2. **构建集成模型的方法**
#### **相同模型，不同初始化**
- **方法**：使用交叉验证确定最佳超参数，然后用不同的随机初始化训练多个模型。
- **缺点**：模型的多样性仅来自初始化，可能不足。

#### **交叉验证中的最佳模型**
- **方法**：使用交叉验证确定最佳超参数，然后选择表现最好的几个模型（如 10 个）进行集成。
- **优点**：无需额外训练。
- **缺点**：可能包含次优模型。

#### **单个模型的不同检查点**
- **方法**：如果训练成本很高，可以取单个模型在不同训练阶段的检查点（如每个 epoch 后）进行集成。
- **优点**：成本低。
- **缺点**：模型多样性不足。

#### **训练期间参数的指数移动平均**
- **方法**：在训练期间维护一个权重副本，记录权重的指数衰减和。
- **优点**：几乎总能提升 1-2% 的性能。
- **直觉**：目标函数呈碗状，网络在模式附近跳动，平均权重更可能接近模式。

---

### 3. **模型集成的缺点**
- **测试时间增加**：集成模型在测试时需要评估多个模型，时间成本较高。

---

### 4. **知识蒸馏（Dark Knowledge）**
- **核心思想**：将集成模型的预测结果“蒸馏”回单个模型，通过修改目标函数来融入集成模型的预测概率。
- **优点**：减少测试时间，同时保留集成模型的性能优势。

“Dark Knowledge”（暗知识）这一术语是由 Geoffrey Hinton 提出的，用来描述在模型集成中隐含的、未被直接利用的知识。以下是其命名原因和核心思想的解释：

### 1. **命名原因**
- **“Dark”**：这里的“Dark”并不是指“暗黑”或“邪恶”，而是指“隐藏的”或“未被直接观察到的”。类似于宇宙中的“暗物质”（Dark Matter），虽然无法直接观测，但通过其影响可以推断其存在。
- **“Knowledge”**：指的是集成模型中蕴含的丰富知识，这些知识通过多个模型的预测结果体现出来。

因此，“Dark Knowledge”可以理解为**隐藏在集成模型中的、未被直接利用的知识**。


### 2. **核心思想**
在模型集成中，多个模型的预测结果通常比单个模型更准确，但集成模型的缺点是计算成本高。知识蒸馏（Knowledge Distillation）的目标是将集成模型的“暗知识”转移到单个模型中，从而在保持性能的同时降低计算成本。

具体来说：
- **集成模型的预测**：集成模型的预测结果（如分类概率）包含了多个模型的“共识”和“不确定性”，这些信息比单个模型的预测更丰富。
- **知识蒸馏**：通过将集成模型的预测结果作为“软标签”（Soft Labels），训练一个单独的模型（学生模型）来模仿集成模型的行为。这样，学生模型可以学习到集成模型中的“暗知识”，从而提升性能。

### 3. **为什么重要**
- **性能提升**：知识蒸馏可以在不增加计算成本的情况下，提升单个模型的性能。
- **模型压缩**：将复杂的集成模型压缩为更小的单一模型，便于部署在资源受限的设备上。

