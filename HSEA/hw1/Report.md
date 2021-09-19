# HSEA-HW1

**匡亚明学院		张子谦		191240076		191240076@smail.nju.edu.cn**	

**本人承诺该实验全程由本人独立完成，无抄袭或给予他人抄袭行为，代码在截至日期后上传至本人github**

**摘要：**



#### 1、引言

（0，0）在左下角

#### 2、实验内容

**Task0：理解代码框架**

- **util.py**中固定了随机数种子，定义了堆、栈、优先队列、Counter字典等数据结构，以及一些归一化、采样等辅助函数。

- **game.py**中 

  *Agent*相当于Abstract类，用于定义游戏实体（即吃豆人和幽灵），通过特定的policy选取action。

  *Configuration*标识agent的位置(x,y pair)与当前朝向(direction)，并通过generateSuccessor方法根据action生成后继configure。

  *AgentState*标识agent的状态，包括configure, speed等数据。

  *Grid*是方格世界，记录了每个2dim的位置信息。

  *Actions*定义了一些agent执行action前后的辅助函数。

  *GameStateData*记录了游戏过程中的状态信息。

  *Game*控制游戏进行，主要根据run方法跑一局游戏，期间依次由agent 产生action并执行，通过本身定义的transition对游戏状态进行更新。

- **pacman.py** 将game.py中的一些类进行了一定的封装，*GameState*允许获取游戏中agent的legalactions，产生state的后继（transition model），获取分数等。*classicGameRules*, *PacmanRules*, *GhostRules*描述了agent与environment交互的规则。

  

我们需要进行完成的内容包括 **search.py**与 **searchAgent.py**两部分，通过命令行参数指定**searchAgent.py**中的Agent进行加载，Agent通过fn，heuristic参数指定**search.py**中的搜索函数（A*）和所采用的启发式函数

##### **Task1：**





**Reference:**