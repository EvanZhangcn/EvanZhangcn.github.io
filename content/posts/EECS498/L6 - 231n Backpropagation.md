---
title: "L6 - 231n Backpropagation"
date: "2025-03-20 09:49:07"
author: "EvanZhangcn"
draft: false
categories: ["EECS498"]  # 在此编辑分类
tags: []               # 在此添加标签
weight: 6
url: "/posts/EECS498/L6 - 231n Backpropagation"  # 自动生成的URL
---
本节两个核心概念： 前向传播 与 反向传播

1. **Forward Pass (前向传播)**：

   - **目标**：计算神经网络的输出（即预测结果）。
   - **过程**：给定输入数据 `X` 和网络的权重（`W1`, `b1`, `W2`, `b2`），前向传播通过网络的各层进行计算，最终输出每个样本的得分或预测结果（`scores`）。在这个函数中，前向传播通过调用 `nn_forward_pass` 函数来计算这些得分。前向传播不仅仅是计算输出，还需要记录中间层的结果（如隐藏层 `h1`），这些信息在后续的反向传播中用来计算梯度。
2. **Backward Pass (反向传播)**：

   - **目标**：计算损失函数(loss)关于每个参数的梯度。
   - **过程**：反向传播通过链式法则计算损失函数（包括数据损失和正则化损失,Ps:正则化项损失一般单独算）对模型参数的导数（即梯度）。这些梯度反映了如何调整每个参数，以最小化损失。在这个函数中，反向传播需要计算每个层的梯度，存储在字典 `grads` 中，之后这些梯度会用来更新参数。`grads['W1']` 和 `grads['b1']` 分别是 `W1` 和 `b1` 的梯度，表示如何调整这些参数以减少损失。

总结：

- **前向传播**负责计算网络输出，即预测值。
- **反向传播**负责计算损失函数对模型参数的梯度，用于后续的参数更新。

### Simple expressions and interpretation of the gradient

1. **乘法函数的偏导数**对于函数 $f(x, y) = xy$，其偏导数为：

   $$
   \frac{\partial f}{\partial x} = y, \quad \frac{\partial f}{\partial y} = x
   $$

   解释：偏导数表示函数在某个变量上的变化率。例如，$\frac{\partial f}{\partial x} = y$ 表示当 $x$ 增加一个微小量 $h$ 时，函数值 $f(x, y)$ 会增加 $y \cdot h$。
2. **导数的定义**导数的定义为：

   $$
   \frac{df(x)}{dx} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
   $$

   解释：导数表示函数在某一点附近的**线性近似**，即函数在该点的斜率。
3. **梯度的定义**梯度是偏导数的向量形式，对于函数 $f(x, y) = xy$，其梯度为：

   $$
   \nabla f = \left[ \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right] = [y, x]
   $$
4. **加法函数的偏导数**对于函数 $f(x, y) = x + y$，其偏导数为：

   $$
   \frac{\partial f}{\partial x} = 1, \quad \frac{\partial f}{\partial y} = 1
   $$

   解释：无论 $x$ 和 $y$ 的值如何，增加任何一个变量都会使函数值增加相同的量。
5. **最大值函数的偏导数**
   对于函数 $f(x, y) = \max(x, y)$，其偏导数为：

   $$
   \frac{\partial f}{\partial x} = \begin{cases} 
   1 & \text{如果 } x \geq y \\
   0 & \text{如果 } x < y 
   \end{cases}, \quad
   \frac{\partial f}{\partial y} = \begin{cases} 
   1 & \text{如果 } y \geq x \\
   0 & \text{如果 } y < x 
   \end{cases}
   $$

   解释：最大值函数的偏导数在较大的输入上为 1，在较小的输入上为 0。

### Compound expressions with chain rule

1. **复合函数的分解**考虑复合函数 $f(x, y, z) = (x + y)z$，可以将其分解为两个简单的表达式：

   - $q = x + y$
   - $f = qz$
2. **局部梯度的计算**复合函数可以分解为简单的表达式，对于分解后的表达式，可以分别计算其局部梯度：

   - 对于 $f = qz$：
     $$
     \frac{\partial f}{\partial q} = z, \quad \frac{\partial f}{\partial z} = q
     $$
   - 对于 $q = x + y$：
     $$
     \frac{\partial q}{\partial x} = 1, \quad \frac{\partial q}{\partial y} = 1
     $$
3. **链式法则的应用****通过链式法则，可以将梯度从最终输出传播到输入变量：**

   - 对于 $x$：
     $$
     \frac{\partial f}{\partial x} = \frac{\partial f}{\partial q} \cdot \frac{\partial q}{\partial x} = z \cdot 1 = z
     $$
   - 对于 $y$：
     $$
     \frac{\partial f}{\partial y} = \frac{\partial f}{\partial q} \cdot \frac{\partial q}{\partial y} = z \cdot 1 = z
     $$
   - 对于 $z$：
     $$
     \frac{\partial f}{\partial z} = q
     $$
4. **反向传播的示例**给定输入 $x = -2$, $y = 5$, $z = -4$，进行前向传播和反向传播：

   - 前向传播：
     $$
     q = x + y = 3, \quad f = q \cdot z = -12
     $$
   - 反向传播：
     - 计算 $f = qz$ 的梯度：

       $$
       \frac{\partial f}{\partial z} = q = 3, \quad \frac{\partial f}{\partial q} = z = -4
       $$
     - 计算 $q = x + y$ 的梯度：

       $$
       \frac{\partial q}{\partial x} = 1, \quad \frac{\partial q}{\partial y} = 1
       $$
     - 应用链式法则：

       $$
       \frac{\partial f}{\partial x} = \frac{\partial f}{\partial q} \cdot \frac{\partial q}{\partial x} = -4 \cdot 1 = -4
       $$

       $$
       \frac{\partial f}{\partial y} = \frac{\partial f}{\partial q} \cdot \frac{\partial q}{\partial y} = -4 \cdot 1 = -4
       $$
5. **梯度的意义**
   最终得到的梯度 $[\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z}] = [-4, -4, 3]$ 表示 $x$, $y$, $z$ 对 $f$ 的敏感性。

### Intuitive understanding of backpropagation

1. **反向传播的局部性**反向传播是一个局部过程。每个门（gate）在电路图中接收输入，并可以立即计算两件事：

   - 它的输出值(its output value)。
   - 它的输出相对于输入的**局部梯度(the local gradient)**。
2. **链式法则的应用**在反向传播过程中，门会通过链式法则递归地学习其输出对整个电路最终输出的梯度。链式法则要求门**将这个梯度乘以**其通常为所有输入计算的**局部梯度**。

   公式表示为：

   $$
   \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}
   $$

   其中，$L$ 是最终输出，$y$ 是门的输出，$x$ 是门的输入。
3. **加法门的例子**加法门接收输入 $[-2, 5]$，计算输出 $3$。由于加法操作的局部梯度对两个输入都是 $+1$，在反向传播过程中，加法门得知其输出的梯度为 $-4$。根据链式法则，加法门将这个梯度乘以其局部梯度，得到两个输入的梯度为 $-4$。

   公式表示为：

   $$
   \frac{\partial L}{\partial x} = -4 \cdot 1 = -4, \quad \frac{\partial L}{\partial y} = -4 \cdot 1 = -4
   $$
4. **反向传播的直观理解**
   [Explain: Click Me!](https://cs231n.github.io/optimization-2/#:~:text=Backpropagation%20can%20thus%20be%20thought%20of%20as%20gates%20communicating%20to%20each%20other%20(through%20the%20gradient%20signal)%20whether%20they%20want%20their%20outputs%20to%20increase%20or%20decrease%20(and%20how%20strongly)%2C%20so%20as%20to%20make%20the%20final%20output%20value%20higher.)
   反向传播可以看作是门与门之间通过梯度信号进行通信，告诉彼此它们的输出是否需要增加或减少（以及强度），以便使最终输出值更高。

### Modularity: Sigmoid example

1. **任意可微函数作为门**任何可微函数都可以作为门（gate），并且我们可以**将多个门组合成一个门（group multiple gates into a single gate）**，或者将一个函数分解为多个门(decompose a function into multiple gates)，以便于计算。
2. **Sigmoid 激活函数的表达式**一个二维神经元的 Sigmoid 激活函数表达式为：

   $$
   f(w, x) = \frac{1}{1 + e^{-(w_0x_0 + w_1x_1 + w_2)}}
   $$

   这个函数由多个门组成，包括加法、乘法、指数函数和倒数函数。
3. **Sigmoid 函数的导数**Sigmoid 函数的导数为：

   $$
   \sigma(x) = \frac{1}{1 + e^{-x}} \quad \Rightarrow \quad \frac{d\sigma(x)}{dx} = \sigma(x) \cdot (1 - \sigma(x))
   $$

   这个导数形式简单且高效，适合在实际应用中使用。
4. **反向传播的实现**反向传播通过递归地应用链式法则，计算每个变量对最终输出的梯度。以下是一个 Sigmoid 神经元的反向传播代码示例：

   ```python
   w = [2, -3, -3]  # 假设一些随机权重和数据
   x = [-1, -2]

   # 前向传播
   dot = w[0] * x[0] + w[1] * x[1] + w[2]
   f = 1.0 / (1 + math.exp(-dot))  # Sigmoid 函数

   # 反向传播
   ddot = (1 - f) * f  # dot 变量的梯度
   dx = [w[0] * ddot, w[1] * ddot]  # 传播到 x
   dw = [x[0] * ddot, x[1] * ddot, 1.0 * ddot]  # 传播到 w
   ```
5. **分阶段反向传播(Staged Backpropagation)的实现技巧**
   在实际应用中，将前向传播分解为多个阶段，每个阶段都易于反向传播。例如，在上面的代码中，我们创建了一个中间变量 `dot`，它保存了 $w$ 和 $x$ 的点积结果。在反向传播过程中，我们依次计算相应的梯度变量（如 `ddot`，最终是 `dw` 和 `dx`）。

### Backprop in practice: Staged computation

对于下面的函数，我们进行编写前向传播和反向传播：

$$
f(x, y) = \frac{x + \sigma(y)}{\sigma(x) + (x + y)^2}
$$

其中，$\sigma$ 是 Sigmoid 函数，定义为：

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

### 1. **前向传播（Forward Pass）**

- **分阶段计算**：将复杂的函数分解为多个简单的中间变量，逐步计算。例如：
  ```python
  sigy = 1.0 / (1 + math.exp(-y))  # 计算sigmoid(y)
  num = x + sigy                   # 计算分子
  sigx = 1.0 / (1 + math.exp(-x))  # 计算sigmoid(x)
  xpy = x + y                      # 计算x + y
  xpysqr = xpy**2                  # 计算(x + y)^2
  den = sigx + xpysqr              # 计算分母
  invden = 1.0 / den               # 计算1/den
  f = num * invden                 # 最终结果
  ```
- **缓存中间变量**（**Cache forward pass variables**.）：在前向传播过程中，缓存中间变量的值，以便在反向传播时使用。

### 2. **反向传播（Backward Pass）**

- **链式法则**：反向传播通过链式法则逐步计算每个中间变量的梯度。例如：
  ```python
  dnum = invden                    # 计算num的梯度
  dinvden = num                    # 计算invden的梯度
  dden = (-1.0 / (den**2)) * dinvden  # 计算den的梯度
  dsigx = (1) * dden               # 计算sigx的梯度
  dxpysqr = (1) * dden             # 计算xpysqr的梯度
  dxpy = (2 * xpy) * dxpysqr       # 计算xpy的梯度
  dx = (1) * dxpy                  # 计算x的梯度
  dy = (1) * dxpy                  # 计算y的梯度
  dx += ((1 - sigx) * sigx) * dsigx  # 累加x的梯度
  dx += (1) * dnum                 # 累加x的梯度
  dsigy = (1) * dnum               # 计算sigy的梯度
  dy += ((1 - sigy) * sigy) * dsigy  # 累加y的梯度
  ```
- **梯度累加**（**Gradients add up at forks**.）：当变量在多个分支中使用时，使用 `+=` 来累加梯度，而不是覆盖。

### 3. **注意事项**

- **缓存前向传播变量**：为了高效计算反向传播，需要缓存前向传播中的中间变量。
- **梯度累加**：在多分支的情况下，梯度会累加，遵循多变量链式法则。

### Patterns in backward flow

1. **反向传播的直观解释**反向传播过程中，梯度可以通过直观的方式解释。以下是神经网络中三个最常用的门（add, mul, max）在反向传播中的行为：
2. **加法门（Add Gate）**加法门将输出梯度**均匀分配**给所有输入，无论前向传播时输入的值是多少。这是因为加法操作的局部梯度为 $+1.0$，因此所有输入的梯度等于输出梯度。

   公式表示为：

   $$
   \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}, \quad \frac{\partial L}{\partial y} = \frac{\partial L}{\partial z}
   $$

   其中，$L$ 是最终输出，$y$ 是加法门的输出。
3. **最大值门（Max Gate）**最大值门将输出梯度**路由到前向传播时值最大的输入**，其他输入的梯度为 $0$。这是因为最大值门的局部梯度为 $1.0$ 对于最大值输入，$0.0$ 对于其他输入。

   公式表示为：

   $$
   \frac{\partial L}{\partial x} = \begin{cases} 
   \frac{\partial L}{\partial y} & \text{如果 } x \text{ 是最大值} \\
   0 & \text{否则}
   \end{cases}
   $$
4. **乘法门（Multiply Gate）**乘法门的局部梯度是输入值（交换后的值），并在链式法则中乘以输出梯度。例如，如果乘法门的输入为 $x$ 和 $y$，则：

   $$
   \frac{\partial L}{\partial x} = y \cdot \frac{\partial L}{\partial y}, \quad \frac{\partial L}{\partial y} = x \cdot \frac{\partial L}{\partial y}
   $$
5. **乘法门的反直觉效应**
   如果乘法门的一个输入非常小，另一个输入非常大，则乘法门会为小输入分配一个相对较大的梯度，为大输入分配一个较小的梯度。这在数据预处理中尤为重要，因为数据的缩放会影响权重的梯度大小。**例如，如果将所有输入数据大小乘以 $1000$，则反而会导致权重W的梯度dW会增大 $1000$ 倍，需要相应地降低学习率。**
   公式表示为：

   $$
   \text{梯度缩放效应} = \text{输入缩放因子} \times \text{原始梯度}
   $$

### Gradients for vectorized operations

### 1. **向量化操作的梯度**

- **扩展到矩阵和向量**：之前讨论的单变量梯度计算可以直接扩展到矩阵和向量操作，但需要特别注意维度和转置操作。
- **矩阵-矩阵乘法的梯度**：矩阵-矩阵乘法的梯度计算是最复杂的操作之一，但它可以推广到矩阵-向量和向量-向量乘法。

---

### 2. **矩阵-矩阵乘法的梯度推导**

- **前向传播**：

  $$
  D = W \cdot X
  $$

  其中：

  - $W$ 是 $5 \times 10$ 的矩阵
  - $X$ 是 $10 \times 3$ 的矩阵
  - $D$ 是 $5 \times 3$ 的矩阵
- **反向传播**：
  假设已知 $D$ 的梯度 $dD$，则 $W$ 和 $X$ 的梯度分别为：

  $$
  dW = dD \cdot X^T
  $$

  $$
  dX = W^T \cdot dD
  $$

  其中：

  - $dD$ 是 $5 \times 3$ 的矩阵
  - $dW$ 是 $5 \times 10$ 的矩阵
  - $dX$ 是 $10 \times 3$ 的矩阵

---

### 3. **维度分析技巧**

- **维度匹配**：通过维度分析可以快速推导梯度公式。例如：
  - $X$ 的维度是 $[10 \times 3]$，$dD$ 的维度是 $[5 \times 3]$。
  - 为了得到 $dW$（维度 $[5 \times 10]$），唯一的方法是计算 $dD \cdot X^T$。
- **无需死记公式**：通过维度匹配可以轻松重新推导梯度公式。

---

### 4. **小规模示例**

- **显式推导**：对于初学者，建议通过小规模的显式示例推导梯度，然后在纸上验证，最后推广到高效的向量化形式。
- **推荐阅读**：Erik Learned-Miller 写了一篇关于矩阵/向量导数的详细文档，可以参考：[链接](https://arxiv.org/abs/1802.01528)。
