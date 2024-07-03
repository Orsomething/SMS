# 船厂调度系统算法设计

# 特点

与常见的集成工艺规划和调度问题不同，船厂的调度规划中，由于场地规模广、工件体积大、工艺数量多的特点，调度过程中，对工件移动时间的计算不可忽略。因此，在调度过程中要更加关注用于移动工件的机器如龙门吊、卡车等的移动时间及其位置，减少其空转运行时间和总移动距离。

综上，在进行规划时，将“移动”作为调度管理的一个内容（job）时，其所需的时间是不固定的，每次发生机器移动后，需要重新计算其从出发点到目标点的运行时间。