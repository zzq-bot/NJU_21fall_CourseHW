# HSEA-HW1

**匡亚明学院		张子谦		191240076		191240076@smail.nju.edu.cn**	

**本人承诺该实验全程由本人独立完成，无抄袭或给予他人抄袭行为，代码在截至日期后上传至本人github**

**摘要：**





### 1、引言

**TODO**

### 2、实验内容

#### **Task1**

**1.1 部件设计**

**Representation**

​	action space是离散的5个动作（向east, west, north, south四个方向前进以及stop)，但stop认为并没有什么帮助，所以只考虑四种前进，通过integer表示。（速度按照游戏设定默认为1.0不进行修改）每个解包括$N$个动作，故表示为$\{0, 1, 2, 3\}^N$的一个向量。解码时，如果在$N$个动作执行前已经到达终点，则后续不必考虑。（在hw1中测试过，pacman吃掉豆子后会直接结束游戏）

**Population size**

​	初始化为k个

**Recombination**

​	考虑one-point crossover 与 uniform crossover

**Mutation**

​	简单的random setting

**Fitness Evaluation**

​	一方面考虑positionHeuristic，值越小，fitness越好。另一方面考虑cost，同样，值越小，fitness越小（主要针对N相对最佳动作较小的情况）

**1.2 整体流程**

通过self.actions存储一系列动作，如果self.actions存有信息，直接取用，否则通过调用getActions生成当前局面的新的actions。具体到getActions函数，会先对当前的population计算fitness（由action）进行





![image-20211108183811798](C:\Users\19124\AppData\Roaming\Typora\typora-user-images\image-20211108183811798.png)
