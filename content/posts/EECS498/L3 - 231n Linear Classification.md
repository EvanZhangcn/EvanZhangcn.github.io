---
title: "L3 - 231n Linear Classification"
date: 2025-03-20T09:31:51Z
draft: false
categories: ["未分类"]  # 在此编辑分类
tags: []               # 在此添加标签
---




---
### Multiclass Support Vector Machine Loss: 多类支持向量机
Multiclass SVM loss如何要求的？ 
对于某一数据点(each image)正确类得分应高于不正确得分$\Delta$,则损失loss = 0
若未满足，则损失loss = $s_j + \Delta - s_{y_i}$,从而积累损失(accumulate loss)
**(这里的$\Delta$是一个预先确定好的超参数Hyperparameter)**
The SVM loss is set up so that the SVM “wants” the correct class for each image to  **have a score higher than the incorrect classes** by some fixed margin $\Delta$.


---
Multiclass SVM Loss的表达式是？ Hinge Loss
#SVM_Loss函数

$$
L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta)
$$
解释：
- $L_i$：第 $i$ 个样本的损失值。
- $\sum_{j \neq y_i}$：对所有不等于真实类别 $y_i$ 的类别 $j$ 求和。
- $\max(0, s_j - s_{y_i} + \Delta)$：计算类别 $j$ 的得分 $s_j$ 与 （真实类别 $y_i$ 的得分 $s_{y_i}$） 之间的差异，并加上一个间隔 $\Delta$。如果结果为负，则取 0（即不贡献损失）。
这个公式是支持向量机（SVM）中常用的**合页损失函数（Hinge Loss）**，用于衡量分类模型的性能。
---

对于多类SVM loss函数，如何计算它的梯度？（此时我们不考虑Sj = Wxj ,就是看这个函数本身）

$$L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + 1)$$

其中：这个函数中的变量 s 是一个向量，每个 sj 和 syi 都是这个向量的分量（是个常数），对函数求导时需要对向量 s 中的每个分量求偏导数，所以说这是一个向量对向量的求导：
- $s_j$ 是第 $j$ 类的得分
- $s_{y_i}$ 是正确类别的得分
- $\max(0, s_j - s_{y_i} + 1)$ 是hinge loss

对正确类别 $y_i$ 求导 $\frac{\partial L_i}{\partial s_{y_i}}$：

$$\frac{\partial L_i}{\partial s_{y_i}} = \sum_{j \neq y_i} \frac{\partial}{\partial s_{y_i}} \max(0, s_j - s_{y_i} + 1)$$

对每个margin项求导：

$$\frac{\partial}{\partial s_{y_i}} \max(0, s_j - s_{y_i} + 1) = \begin{cases} 
-1 & \text{if } s_j - s_{y_i} + 1 > 0 \\ 
0 & \text{if } s_j - s_{y_i} + 1 \leq 0
\end{cases}$$

因此，总梯度为：

$$\frac{\partial L_i}{\partial s_{y_i}} = -\sum_{j \neq y_i} \mathbb{1}[s_j - s_{y_i} + 1 > 0]$$

其中 $\mathbb{1}[\cdot]$ 是示性函数。

---
图示Multiclass SVM loss:
![[L3 - CS231n Convolutional Neural Networks for Visual Recognition by_2025-03-04.png]]
 If any class has a score inside the red region (or higher), then there will be accumulated loss. Otherwise the loss will be zero.


---
什么是hinge loss（合页损失）  和  the squared hing loss SVM（平方合页损失）?
Hinge Loss 与 Squared Hinge Loss
- **Hinge Loss**: $\max(0, -)$
- **Squared Hinge Loss (L2-SVM)**: $\max(0, -)^2$

**说明**：
- Hinge Loss 是标准的损失函数形式。
- Squared Hinge Loss 对**违反间隔的情况**惩罚更强烈（二次而非线性）。
- 在某些数据集中，Squared Hinge Loss 可能表现更好，可以通过交叉验证确定。



---
上面的表达式因为没有进行对于权重矩阵的正则化，而存在一个bug?
假设我们有一个数据集，基于此训练出一个权重矩阵W，能够正确分类每一个样本，那么$\lambda W$  where $\lambda > 1$同样可以满足要求，分类每一个样本点，同样损失为0
>原因解释：
>this transformation uniformly stretches all score magnitudes and hence also their absolute differences.

因此我们通过加入对权重矩阵的正则化项，encode some preference 从而确定a certain set of weights:

---
什么是正则化惩罚项？
#正则化惩罚项 
$R(W)$
最常见的正则化惩罚项是 **L2 范数的平方**，它通过对所有参数进行逐元素的二次惩罚来抑制较大的权重：

$$
R(W) = \sum_k \sum_l W_{k,l}^2
$$

**说明**：
- L2 正则化通过惩罚较大的权重，防止模型过拟合。
- 公式中的 $W_{k,l}$ 表示权重矩阵中的每个元素。
强调：该正则化函数仅基于权重，而不是数据样本点**Notice that the regularization function is not a function of the data, it is only based on the weights.**



---
多类支持向量机损失（Multiclass SVM Loss）完整形式是？
#多类支持向量机损失完整形式
多类支持向量机损失由两部分组成：  
1. **数据损失（Data Loss）**：所有样本的平均损失 $L_i$。  
2. **正则化损失（Regularization Loss）**：用于抑制模型复杂度的惩罚项。  

完整的多类 SVM 损失函数为：

$$
L = \frac{1}{N} \sum_i L_i \quad \text{(数据损失)} + \lambda R(W) \quad \text{(正则化损失)}
$$

展开后的完整形式为：

$$
L = \frac{1}{N} \sum_i \sum_{j \neq y_i} \left[ \max(0, f(x_i; W)_j - f(x_i; W)_{y_i} + \Delta) \right] + \lambda \sum_k \sum_l W_{k,l}^2
$$

**说明**：  
- $N$ 是训练样本的数量。  
- $\lambda$ 是一个超参数，用于权衡数据损失和正则化损失。  
- $\lambda$ 的值通常通过交叉验证确定，没有简单的设置方法。

---
Ps：该损失函数的梯度是什么？ 
当我们对损失函数求导时，自变量是参数，我们要找optimal parameter,因此上面的表达式中自变量是权重矩阵W，因此整个表达式对矩阵W进行求导：
#多分类SVM损失函数梯度推导
完整的多分类 SVM 损失函数为：

对 $W$ 计算梯度，我们首先对数据损失项求导。
$$
L_{\text{data}} = \frac{1}{N} \sum_{i} \sum_{j \neq y_i} \max(0, W_j x_i - W_{y_i} x_i + \Delta)
$$
![[L3 - 231n Linear Classification by_2025-03-05.png]]


首先对 ($W_j$ ) 求导，其中：$( j \neq y_i)$:
- **情况1**: 如果 $S_j - S_{y_i} + \Delta \leq 0$，则 max 取 0，损失对 $W$的梯度为 0。
- **情况2**: 如果  $S_j - S_{y_i} + \Delta > 0$，对 $W_j$ 求导：

$$
\frac{\partial}{\partial W_j} (S_j - S_{y_i}) = \frac{\partial}{\partial W_j} (W_j x_i - W_{y_i} x_i) = x_i
$$

于是：

$$
\frac{\partial L_{\text{data}}}{\partial W_j} = \frac{1}{N} \sum_{i} \sum_{j \neq y_i} 1(S_j - S_{y_i} + \Delta > 0) x_i
$$
 解释：
- $1(\cdot)$ 是指示函数，表示这个项是否大于 0。
- 只有满足 $S_j - S_{y_i} + \Delta > 0$ 时，这个梯度才生效。


然后对 $W_{y_i}$ 求导，其中：$( j \neq y_i)$:：
- **情况1**: 如果 $S_j - S_{y_i} + \Delta \leq 0$，则 max 取 0，损失对 $W$的梯度为 0。
- **情况2**: 如果  $S_j - S_{y_i} + \Delta > 0$，对 $W_{y_i}$ 求导：

$$
\frac{\partial}{\partial W_{y_i}} (S_j - S_{y_i}) = \frac{\partial}{\partial W_{y_i}} (W_j x_i - W_{y_i} x_i) = -x_i
$$

于是：

$$
\frac{\partial L_{\text{data}}}{\partial W_{y_i}} = -\frac{1}{N} \sum_{i} \sum_{j \neq y_i} 1(S_j - S_{y_i} + \Delta > 0) x_i
$$

最后，对正则化项求导：
正则化项：

$$
R(W) = \sum_{k} \sum_{l} W_{k,l}^2
$$

$$
\frac{\partial R(W)}{\partial W} = 2W
$$
因此，完整的梯度表达式为：

$$
\frac{\partial L}{\partial W} = \frac{\partial L_{\text{data}}}{\partial W} + 2\lambda W
$$
---
惩罚大权重(penalize large weights)的好处是什么？ 提高了模型泛化能力，减少过拟合
[Explain: Click Me!](http://cs231n.github.io/linear-classify/#:~:text=The%20most%20appealing,overfitting.)





### Softmax Classifier
两个比较常见的分类器： SVM classifier 和 Softmax classifier
Softmax classifier可以认为是Data100的**二元逻辑回归分类器**的针对多个类的泛化版本。

Softmax Loss Function：
Softmax分类器将（X @ W）得到的分数解释为每个类别的**未归一化对数概率**（于是采用softmax函数进行normalize），并用**交叉熵损失替换铰链损失（hinge loss）**，交叉熵损失的形式为：
对于第i个样本：

$$
L_i = -\log\left(\frac{e^{f_{y_i}}}{\sum_j e^{f_j}}\right) \quad \text{或等价地} \quad L_i = -f_{y_i} + \log\sum_j e^{f_j}
$$

其中，$f_j$ 表示类别分数向量 $f$ 的第 $j$ 个元素。
#softmax函数
**函数 $f_j(z) = \frac{e^{z_j}}{\sum_k e^{z_k}}$ 被称为 softmax 函数：它将任意实值分数的向量（在 $z$ 中）压缩为一个介于零和一之间且总和为一的向量。**

与之前一样，整个数据集的损失是**所有训练样本的 $L_i$ 的平均值**加上**正则化项 $R(W)$**。

---
等价形式如何推导出来的？
**等价形式**：$L_i = -f_{y_i} + \log\sum_j e^{f_j}$  
   这个形式通过将原始形式中的对数运算展开得到。具体来说，$\log\left(\frac{e^{f_{y_i}}}{\sum_j e^{f_j}}\right) = \log(e^{f_{y_i}}) - \log(\sum_j e^{f_j}) = f_{y_i} - \log\sum_j e^{f_j}$。因此，取负值后得到 $L_i = -f_{y_i} + \log\sum_j e^{f_j}$。



---
从信息论的角度理解上面的公式：
1. **交叉熵的定义 (Definition of Cross-Entropy)**
The cross-entropy between a “true” distribution $p$ and an estimated distribution $q$ is defined as:  
交叉熵用于衡量**两个概率分布 $p$ 和 $q$ 之间的差异**，定义为：  
$$
H(p, q) = -\sum_x p(x) \log q(x)
$$
其中，$p$ 是“真实”分布，$q$ 是估计分布。
**交叉熵越大，说明两个分布之间的差异越大。**
1.当q完全匹配p时，交叉熵达到最小值，等于真实分布p的熵H(p)。 
2.当q与p差异越大时，交叉熵的值也会越大。

---
2. **Softmax 分类器的目标 (Objective of the Softmax Classifier)**
Softmax 分类器通过最小化交叉熵来优化模型。其中，估计分布 $q$ 是 Softmax 函数的输出：  
$$
q = \frac{e^{f_{y_i}}}{\sum_j e^{f_j}}
$$
而“真实”分布 $p$ 是一个独热编码（one-hot）向量，即在正确类别位置为 1，其余位置为 0：  
$$
p = [0, \dots, 1, \dots, 0]
$$

由于 $p$ 是一个独热编码向量，只有在正确类别 $y_i$ 的位置为 1，其余位置为 0，因此：

- 当 $j = y_i$ 时，$p_j = 1$。
- 当 $j \neq y_i$ 时，$p_j = 0$。

将这些值代入交叉熵损失函数：

$$
L_i = -\sum_{j=1}^{C} p_j \log(q_j) = -p_{y_i} \log(q_{y_i}) - \sum_{j \neq y_i} p_j \log(q_j)
$$

由于 $p_j = 0$ 对于所有 $j \neq y_i$，所以：

$$
L_i = -1 \cdot \log(q_{y_i}) - 0 = -\log(q_{y_i})
$$

因此，交叉熵损失函数简化为：

$$
L_i = -\log(q_{y_i})
$$





---
 3. **交叉熵与 KL 散度的关系 (Relationship Between Cross-Entropy and KL Divergence)**

**中文**：  
交叉熵可以分解为真实分布的熵 $H(p)$ 和 KL 散度 $D_{KL}(p || q)$ 之和：
The cross-entropy can be decomposed into the entropy of the true distribution $H(p)$ and the **Kullback-Leibler (KL) divergence** $D_{KL}(p || q)$:    
$$
H(p, q) = H(p) + D_{KL}(p || q)
$$
其中：  
1. **$H(p)$ 是真实分布 $p$ 的熵，是一个固定值。**  
2. $D_{KL}(p || q)$ 是 KL 散度，衡量 $q$ 与 $p$ 之间的差异。

因此，交叉熵 $H(p, q)$ 的大小主要由 KL 散度 $D_{KL}(p || q)$ 决定：  
- **KL 散度越大**，说明 $q$ 与 $p$ 的差异越大，交叉熵也越大。  
- **KL 散度越小**，说明 $q$ 与 $p$ 的差异越小，交叉熵也越小。
在 Softmax 分类器中，真实分布 $p$ 是一个独热编码，其熵 $H(p)$ 为零。**因此，最小化交叉熵等价于最小化 KL 散度，即让预测分布 $q$ 尽可能接近真实分布 $p$。**

要点总结
1. **交叉熵的定义**：衡量真实分布 $p$ 和估计分布 $q$ 之间的差异。
2. **Softmax 分类器的目标**：通过最小化交叉熵，使预测分布 $q$ 接近真实分布 $p$。
3. **交叉熵与 KL 散度的关系**：最小化交叉熵等价于最小化 KL 散度。
4. **核心思想**：让预测分布 $q$ 的所有概率质量集中在正确答案上。
---
从概率论的角度理解： 最小化负对数概率 等价于 最大化似然估计
[Explain: Click Me!](https://cs231n.github.io/linear-classify/#softmax-classifier#:~:text=Probabilistic%20interpretation.%20Looking%20at%20the%20expression%2C%20we%20see%20that)
具体解释超过课程范围了。

---
用代码实现Softmax函数时，由于指数函数的存在，导致中间项 $e^{f_{y_i}}$ 和 $\sum_j e^{f_j}$ 可能会变得非常大怎么办？
采用归一化技巧：[Explain: Click Me!](https://cs231n.github.io/linear-classify/#softmax-classifier#:~:text=Practical%20issues%3A%20Numeric%20stability.)
1. **归一化技巧**：
   我们可以通过乘以一个常数 $C$ 并将其推入求和项中，来保持数学上的等价性：
   $$
   \frac{e^{f_{y_i}}}{\sum_j e^{f_j}} = \frac{C e^{f_{y_i}}}{C \sum_j e^{f_j}} = \frac{e^{f_{y_i} + \log C}}{\sum_j e^{f_j} + \log C}
   $$
   这里，我们可以自由选择 $C$ 的值，这不会改变结果，但可以提高计算的数值稳定性。

2. **选择 $C$ 的值**：
   一个常见的选择是令 $\log C = -\max_j f_j$。**这意味着我们将向量 $f$ 中的值进行平移，使得最大值变为零。**这样可以避免指数函数产生过大的值。  找出分数最大的值并减去

3. **代码实现**：
   ```python
   f = np.array([123, 456, 789])  # 示例：3个类别，每个类别有较大的分数
   p = np.exp(f) / np.sum(np.exp(f))  # 错误做法：可能导致数值问题
   
   # 正确做法：先将 f 中的值平移，使得最大值为 0
   f -= np.max(f)  # f 变为 [-666, -333, 0]
   p = np.exp(f) / np.sum(np.exp(f))  # 安全操作，得到正确结果
   ```

---
SVM分类器使用hinge loss（有时候也叫做max-margin loss）
Softmax分类器采用cross-entropy， 命名得益于采用Softmax函数将所得分数**压缩变为0-1之间的正值**，从而方便采用交叉熵衡量损失。





### SVM vs. Softmax
![[L3 - CS231n Convolutional Neural Networks for Visual Recognition by_2025-03-05.png]]
解释:[Explain: Click Me!](https://cs231n.github.io/linear-classify/#softmax-classifier#:~:text=Example%20of%20the%20difference%20between%20the%20SVM%20and%20Softmax%20classifiers%20for%20one%20datapoint.)


----
Softmax分类器对于某个样本点给出每个类的得分如何理解？[Explain: Click Me!](https://cs231n.github.io/linear-classify/#softmax-classifier#:~:text=Softmax%20classifier%20provides%20%E2%80%9Cprobabilities%E2%80%9D%20for%20each%20class.)

the ordering of the scores is interpretable :得分的相对大小排序是可以解释的，但是the absolute numbers无法从技术上解释，不能完全理解为“概率”。
因为正则化项对于权重约束越大，Wl里面大权重值被惩罚越厉害，导致最后都是较小权重，得分出来也比较近似均匀般的小，所以这个得分绝对数值没啥意义。


---
实际上，两种分类器能力差不多相同，差别很小。
从损失函数的角度比较两者：
In other words, the Softmax classifier is never fully happy with the scores it produces: the correct class could always have a higher probability and the incorrect classes always a lower probability and **the loss would always get better**.（损失永远不可能为0） 

However, the SVM is happy once the margins are satisfied（损失可以出现为0情况） and it does not micromanage the exact scores beyond this constraint.



