import copy
import json
import jsonpath
import numpy as np
import random
import math
from copy import deepcopy
import matplotlib.pyplot as plt
import itertools
import datetime

params={
    "file":"KIM.json",
    #车间参数
    "num_machine": 15,
    "num_job": 24,
    #json文件读取参数
    "process": [],
    "num_process": 0,
    "job": [],
    "machine": [],
    "time": [],
    "preprocess": [],               # pre_id
    "preprocess_combine": [],
    "postprocess": [],
    "postprocess_combine":[],
    "process_before":[],
    "num_process_after": [],
    "num_postprocess": [],
    "mean_time": [],
    "m_mean_time": 0,
    "mut_machine": [],
    "or_list": [[["2-9", "2-10", "2-11"], ["2-12", "2-13"]],[["4-2", "4-3", "4-4"], ["4-5", "4-6"]],[["4-14"], ["4-15"]],
                [["5-2", "5-3", "5-4", "5-5", "5-6", "5-7", "5-8", "5-9"], ["5-10", "5-11","5-12", "5-13"]],
                [["5-4", "5-5", "5-6"], ["5-7", "5-8"]],[["6-6", "6-7", "6-8", "6-9", "6-10"], ["6-11", "6-12", "6-13"]],
                [["6-8"], ["6-10"]],[["7-2", "7-3", "7-4", "7-5", "7-6"], ["7-7", "7-8", "7-9"]],[["7-3"], ["7-4","7-5"]],
                [["7-12", "7-13"], ["7-14", "7-15", "7-16","7-17"]],[["7-15"], ["7-16"]],
                [["8-2", "8-3", "8-4", "8-5", "8-6", "8-7", "8-8", "8-9", "8-10", "8-11"], ["8-12", "8-13", "8-14","8-15","8-16"]],
                [["8-4","8-5"], ["8-6"]],[["8-9"], ["8-10"]],[["8-18"], ["8-19"]],[["9-3"], ["9-4", "9-5"]],
                [["9-8","9-9"], ["9-10"]],[["9-17"], ["9-18","9-19"]],[["10-4","10-5","10-6"],["10-7","10-8"]],
                [["12-14","12-15"],["12-16","12-17"]],
                [["13-2", "13-3", "13-4", "13-5", "13-6", "13-7", "13-8", "13-9", "13-10", "13-11"], ["13-12", "13-13", "13-14","13-15","13-16"]],
                [["13-3","13-4"], ["13-5"]],[["13-8"], ["13-9","13-10"]],[["14-2","14-3"], ["14-4"]],[["14-6"], ["14-7"]],
                [["15-2"], ["15-3","15-4","15-5"]],[["15-8","15-9"], ["15-10"]],
                [["16-2", "16-3", "16-4", "16-5"], ["16-6", "16-7", "16-8", "16-9", "16-10", "16-11", "16-12", "16-13", "16-14","16-15","16-16","16-17"]],
                [["16-7", "16-8", "16-9", "16-10", "16-11"], ["16-12", "16-13", "16-14"]],[["16-19"], ["16-20"]],
                [["17-2", "17-3", "17-4", "17-5", "17-6", "17-7"], ["17-8", "17-9", "17-10", "17-11"]],
                [["17-3","17-4","17-5"], ["17-6"]],[["17-14"], ["17-15","17-16"]],[["17-19","17-20"], ["17-21"]],
                [["18-2", "18-3"], ["18-4", "18-5"]],[["18-8"], ["18-9"]],[["18-13"], ["18-14", "18-15"]]],
    "or_list_time": [],
    "or_list_combined": [],
    "mut_orlist": [],
    #GA参数
    "num_group":500, # 种群数
    "proba_cross":0.8,# 交叉率
    "proba_mutate":0.2,  # 变异率
    "num_epoch":500
}

#从json文件读取
with open(params["file"], "r", encoding="utf-8") as f:
    data = json.load(f)
params["process"] = jsonpath.jsonpath(data,'$...id')
params["num_process"] = len(params["process"])
for i in params["process"]:
    if i[1] == '-':
        params["job"].append(int(i[0]))
    else:
        params["job"].append(int(i[1])+10)
params["machine"] = jsonpath.jsonpath(data,'$...machine')
params["time"] = jsonpath.jsonpath(data, '$...time')
params["preprocess"] = jsonpath.jsonpath(data, '$...preprocess_id')
params["preprocess_combine"] = jsonpath.jsonpath(data, '$...preprocess_combine')
params["postprocess"] = jsonpath.jsonpath(data, '$...postprocess_id')
params["postprocess_combine"] = jsonpath.jsonpath(data, '$...postprocess_combine')
print(params)

#依据前置工序等信息求出所有前位任务列表、后位任务数量和权重
process_before_old = [[] for i in range(params["num_process"])]
process_before_new = deepcopy(params["preprocess"])
for i in range(params["num_process"]):
    while process_before_old[i] != process_before_new[i]:
        process_before_old[i] = deepcopy(process_before_new[i])
        for j in process_before_old[i]:
            index_j = params["process"].index(j)
            process_before_new[i].extend(params["preprocess"][index_j])
            process_before_new[i] = list(dict.fromkeys(process_before_new[i]))
params["process_before"] = process_before_new
params["num_process_after"] = [0 for i in range(params["num_process"])]
params["num_postprocess"] = [0 for i in range(params["num_process"])]
for i in range(params["num_process"]):
    for j in params["process_before"][i]:
        params["num_process_after"][params["process"].index(j)] += 1
    if params["postprocess_combine"][i]=="or":
        params["num_postprocess"][i] = 1
    elif params["postprocess_combine"][i]=="orand":
        params["num_postprocess"][i] = 2
    else:
        params["num_postprocess"][i] = len(params["postprocess"][i])
def find(matrix, item):
    position = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            try:
                k = matrix[i][j].index(item)
                position.append([i,j,k])
            except ValueError:
                pass
    if position:
        return position
    else:
        print("Item not found in or_list")
        raise ValueError
for i in params["process"]:
    if i in list(itertools.chain.from_iterable(itertools.chain.from_iterable(params["or_list"]))):
        for j in params["process_before"][params["process"].index(i)]:
            if j not in list(itertools.chain.from_iterable(itertools.chain.from_iterable(params["or_list"]))):
                params["num_process_after"][params["process"].index(j)] -= 0.5
            else:
                position_i = find(params["or_list"], i)
                position_j = find(params["or_list"], j)
                feature = False
                for pos1 in position_i:
                    for pos2 in position_j:
                        if pos1[0]==pos2[0] and pos1[1]==pos2[1]:
                            feature = True
                if feature==False:
                    params["num_process_after"][params["process"].index(j)] -= 0.5
params["mean_time"] = [0 for i in range(params["num_process"])]
for i in range(params["num_process"]):
    params["mean_time"][i] = np.mean(params["time"][i])
params["m_mean_time"] = np.mean(params["mean_time"])
for i in range(params["num_process"]):
    if len(params["machine"][i])>1:
        params["mut_machine"].append(i)
for i in range(len(params["or_list"])):
    time0=0
    time1=0
    for j0 in params["or_list"][i][0]:
        if j0 in params["process"]:
            if list(itertools.chain.from_iterable(itertools.chain.from_iterable(params["or_list"][i:]))).count(j0)>1:
                time0 += params["mean_time"][params["process"].index(j0)] / 2
            else:
                time0 += params["mean_time"][params["process"].index(j0)]
    for j1 in params["or_list"][i][1]:
        if j1 in params["process"]:
            if list(itertools.chain.from_iterable(itertools.chain.from_iterable(params["or_list"][i:]))).count(j1)>1:
                time1 += params["mean_time"][params["process"].index(j1)] / 2
            else:
                time1 += params["mean_time"][params["process"].index(j1)]
    params["or_list_time"].append([time0, time1])
for i in range(len(params["or_list"])-1):
    for o in itertools.chain.from_iterable(params["or_list"][i]):
        for j in range(i+1, len(params["or_list"])):
            if o in itertools.chain.from_iterable(params["or_list"][j]):
                if [i, j] not in params["or_list_combined"]:
                    params["or_list_combined"].append([i,j])
for j in range(len(params["or_list"])):  # 工序更换or型选择
    if params["or_list"][j][0][0] in params["process"]:
        params["mut_orlist"].append(j)
print(params)


class GA_solve:
    def __init__(self, params):
        # 车间参数
        self.num_machine = params["num_machine"]
        self.num_job = params["num_job"]
        # json文件读取参数
        self.process = params["process"]
        self.num_process = params["num_process"]
        self.job = params["job"]
        self.machine = params["machine"]
        self.time = params["time"]
        self.preprocess = params["preprocess"]
        self.preprocess_combine = params["preprocess_combine"]
        self.process_before = params["process_before"]
        self.num_process_after = params["num_process_after"]
        self.num_postprocess = params["num_postprocess"]
        self.m_mean_time = params["m_mean_time"]
        self.mut_machine = params["mut_machine"]
        self.or_list = params["or_list"]
        self.or_list_time = params["or_list_time"]
        self.or_list_combined = params["or_list_combined"]
        self.mut_orlist = params["mut_orlist"]
        # GA参数
        self.num_group = params["num_group"]
        self.proba_cross = params["proba_cross"]
        self.proba_mutate = params["proba_mutate"]
        self.num_epoch = params["num_epoch"]

    def find_or(self, item):
        or_list = self.or_list
        position = []
        for i in range(len(or_list)):
            for j in range(len(or_list[i])):
                try:
                    k = or_list[i][j].index(item)
                    position.append([i, j, k])
                except ValueError:
                    pass
        if position:
            return position
        else:
            print("Item not found in or_list")
            raise ValueError

    def __encoding(self):
        """
        初始化数量为num_group的种群，染色体为两个个长度是num_process的全排列+一个or_list长度的排列
        （第一个代表1，2，3...工序的运作机器，第二个代表工序顺序，第三个0/1的集合代表选择or的哪个分支进行排产）
        初始化过程中如无OR型选择将使第三个染色体全部为1
        第一个染色体按照在对应机器上的工作时间反比概率选择
        """
        process = self.process
        num_process = self.num_process
        machine = self.machine
        time = self.time
        preprocess = self.preprocess
        num_process_after = self.num_process_after
        num_postprocess = self.num_postprocess
        num_group = self.num_group
        m_mean_time = self.m_mean_time
        or_list = self.or_list
        or_list_time = self.or_list_time

        group = []
        if num_group > math.factorial(num_process):
            print("参数num_group过大")
            raise ValueError

        while len(group)<0.2 * num_group:
            ## 等概率随机输出是否选择
            or_list_chosen = []
            for i in range(len(or_list_time)):
                if or_list_time[i]==[0,0]:          # 当该分支不存在于当前问题中时
                    or_list_chosen.append(-1)
                else:
                    or_list_chosen.append(random.randint(0,1))
            for i in range(len(or_list_time)):      # 确保嵌套型or统一
                if or_list_chosen[i]==0:
                    for j in or_list[i][1]:
                        if list(itertools.chain.from_iterable(itertools.chain.from_iterable(or_list[i:]))).count(j)>1:
                            pos_j = self.find_or(j)
                            for p in pos_j:
                                if p[0]!=i:
                                    or_list_chosen[p[0]] = -1
                if or_list_chosen[i]==1:
                    for j in or_list[i][0]:
                        if list(itertools.chain.from_iterable(itertools.chain.from_iterable(or_list[i:]))).count(j)>1:
                            pos_j = self.find_or(j)
                            for p in pos_j:
                                if p[0]!=i:
                                    or_list_chosen[p[0]] = -1
            ## 等概率随机输出gene_machine(按工序原始顺序排序)
            gene_machine = []
            for i in range(num_process):
                gene_machine.append(machine[i][random.randrange(len(machine[i]))])
            ## 通过排序将随机序列整理成为符合前后置关系的序列输出gene_order
            gene_order = random.sample(range(num_process), num_process)
            pointer = 0  # 指针指向第一个
            while pointer < len(gene_order) - 1:  # 类似quicksort的排序方法
                pivot = gene_order[pointer]  # 指针指向的元素
                preprocess_id = preprocess[pivot]
                if len(preprocess_id) == 0:  # 如果指针指向的元素没有前置元素，指针向后挪动一位
                    pointer = pointer + 1
                else:  # 如果指针指向的有前置元素，找出所有前置的元素序号
                    indexesingene_pre = []
                    for id in preprocess_id:
                        for index, value in enumerate(process):
                            if value == id:
                                indexesingene_pre.append(gene_order.index(index))  # 通过序号定位其在当前gene排序中的index
                    if len(indexesingene_pre) == 0 or pointer > max(
                            indexesingene_pre):  # 如果所有前置元素位置都在pivot之前或前置元素不在当前列表中，指针向后挪动一位
                        pointer = pointer + 1
                    else:  # 否则，找到位置最后的前置元素，并将pivot移动到该元素后一位，其余顺序不变
                        gene_order[pointer:max(indexesingene_pre)], gene_order[max(indexesingene_pre)] = \
                            gene_order[pointer + 1:max(indexesingene_pre) + 1], gene_order[pointer]
            if gene_machine+gene_order+or_list_chosen not in group:
                group.append(gene_machine+gene_order+or_list_chosen)
        print(len(group))

        while len(group)<num_group:
            ## 根据选择型分支中的工序时间和反比，概率随机输出选择的分支
            or_list_chosen = []
            for i in range(len(or_list_time)):
                if or_list_time[i] == [0, 0]:  # 当该分支不存在于当前问题中时
                    or_list_chosen.append(-1)
                else:
                    if random.random() < or_list_time[i][0]/sum(or_list_time[i]):
                        or_list_chosen.append(1)
                    else:
                        or_list_chosen.append(0)
            for i in range(len(or_list_time)):      # 确保嵌套型or统一
                if or_list_chosen[i]==0:
                    for j in or_list[i][1]:
                        if list(itertools.chain.from_iterable(itertools.chain.from_iterable(or_list[i:]))).count(j)>1:
                            pos_j = self.find_or(j)
                            for p in pos_j:
                                if p[0]!=i:
                                    or_list_chosen[p[0]] = -1
                if or_list_chosen[i]==1:
                    for j in or_list[i][0]:
                        if list(itertools.chain.from_iterable(itertools.chain.from_iterable(or_list[i:]))).count(j)>1:
                            pos_j = self.find_or(j)
                            for p in pos_j:
                                if p[0]!=i:
                                    or_list_chosen[p[0]] = -1
            ## 根据工序在机器上的时间反比，概率随机输出gene_machine(按工序原始顺序排序)
            gene_machine = []
            for i in range(num_process):
                temp_t_inverse = [1/t for t in time[i]]
                for j in range(len(machine[i])):
                    if random.random() < temp_t_inverse[0]/sum(temp_t_inverse):
                        gene_machine.append(machine[i][j])
                        break
                    else:
                        temp_t_inverse.pop(0)
            ## 在满足先序关系的前提下，根据工序权重，概率随机输出gene_order
            gene_order = []
            weight = []
            for i in range(num_process):                    # 根据已经定下的gene_machine对应的时间计算权重
                w_p_after = num_process_after[i]
                w_time = time[i][machine[i].index(gene_machine[i])]
                weight.append(w_p_after + w_time/m_mean_time)
            list_WD = []
            for i in range(num_process):                    # 将无前置任务的工序序号放入待选list WD
                if len(preprocess[i])==0:
                    list_WD.append(i)
            while list_WD!=[]:
                list_weight_WD = []
                for i in list_WD:                           # 计算list_WD工序权重
                    list_weight_WD.append(weight[i])
                while list_WD!=[]:                 # 根据工序权重，概率随机输出gene_order
                    if random.random() < list_weight_WD[0] / sum(list_weight_WD):
                        gene_order.append(list_WD[0])
                        break
                    else:
                        list_weight_WD.pop(0)
                        list_WD.pop(0)
                list_WD = []                        # list WD更新： 将无前置任务或前置任务全部安排过的工序序号放入待选list WD
                for i in range(num_process):
                    inWD = True
                    for p in preprocess[i]:
                        if process.index(p) not in gene_order:
                            inWD = False
                    if inWD == True and i not in gene_order:
                        list_WD.append(i)
            if gene_machine+gene_order+or_list_chosen not in group:
                group.append(gene_machine+gene_order+or_list_chosen)
        print(len(group))
        return group

    def decoding(self, gene):
        """
        对于每个个体(排列)，解码出他的完成时间
        gene:是当前排列顺序
        """
        num_machine = self.num_machine
        num_job = self.num_job
        process = self.process
        num_process = self.num_process
        job = self.job
        machine = self.machine
        time = self.time
        preprocess = self.preprocess
        or_list = self.or_list
        gene_machine = gene[:num_process]
        gene_order = gene[num_process:2*num_process]
        or_list_chosen = gene[2*num_process:]
        process_chosen = []
        process_not_chosen = []
        for i in range(len(or_list)):
            if or_list_chosen[i]!=0:
                process_not_chosen.extend(or_list[i][0])
            if or_list_chosen[i]!=1:
                process_not_chosen.extend(or_list[i][1])
        for i in process:
            if i not in process_not_chosen:
                process_chosen.append(i)

        gene_starttime = [0 for _ in range(num_process)]        # 记录每个步骤的开始时间   按照读取的工序原顺序排序
        gene_finishtime = [0 for _ in range(num_process)]       # 记录每个步骤的结束时间
        job_finishtime = [0 for _ in range(num_job)]            # 记录每个job的结束时间
        machine_finishtime = [0 for _ in range(num_machine)]    # 记录每个machine工作结束的时间
        machine_use = [[] for _ in range(num_machine)]          # 记录用过machine的process（以序号记录）

        for i in gene_order:
            if process[i] in process_chosen:
                m = gene_machine[i]                                 # 找到该process需要的machine
                gene_starttime[i] = machine_finishtime[m-1]
                # 与job导致的最早开始时间进行比较，gene_starttime[i]替换成两者中较后的一个
                if job_finishtime[job[i] - 1] > gene_starttime[i]:
                    gene_starttime[i] = job_finishtime[job[i] - 1]

                # 确定最终gene_starttime[i]后，更新所有列表[i]
                gene_finishtime[i] = gene_starttime[i] + time[i][machine[i].index(gene_machine[i])]
                job_finishtime[job[i] - 1] = gene_finishtime[i]
                machine_finishtime[m-1] = gene_finishtime[i]
                machine_use[m-1].append(i)

        total_time = max(machine_finishtime)

        # 定义约束函数：使不符合前后置关系的排序得出时间为无穷大
        for i in range(num_process):
            preprocess_id = preprocess[i]
            if len(preprocess_id) != 0:  # 如果指针指向的元素有前置元素
                indexesingene_pre = []
                for j in range(len(preprocess_id)):
                    for index, value in enumerate(process):
                        if value == preprocess_id[j]:
                            indexesingene_pre.append(gene_order.index(index))  # 通过序号定位其在当前gene排序中的位置
                    if gene_order.index(i) < max(indexesingene_pre):
                        total_time = np.inf

        # print(f"{total_time}:{gene},{gene_starttime},{gene_finishtime}")
        return [total_time, gene_starttime, gene_finishtime, machine_use, job_finishtime]

    def __fitness(self, time_list):
        """
        适应度取1/x
        """
        a = np.array(list(map(lambda x: 1/x, time_list)))
        return a / sum(a)

    def __choose(self, fitness_list, group_list):
        """
        选择两个父代（轮盘赌）
        两个父代不一样
        """
        a, b = np.random.choice(range(len(group_list)), 2, replace=False, p=fitness_list)
        return (group_list[a], group_list[b])

    def __cross(self, sample_tuple):
        """
        子代1交叉点及以内继承父代2内排列，交叉点外剩余顺序继承父代1。子代2相反
        """
        gene1, gene2 = sample_tuple
        num_process = self.num_process
        or_list_combined = self.or_list_combined

        # 对or_list_chosen进行交叉：list中的互换，list外的不变
        l = len(self.or_list)
        n = random.randint(math.ceil(l/3), math.floor(l*2/3))
        list_cross_orchosen = sorted(random.sample(range(l), n))
        for i in list_cross_orchosen:
            if i in list(itertools.chain.from_iterable(or_list_combined)):
                index = list(itertools.chain.from_iterable(or_list_combined)).index(i)
                if or_list_combined[math.floor(index / 2)][0] not in list_cross_orchosen or \
                        or_list_combined[math.floor(index / 2)][1] not in list_cross_orchosen:
                    list_cross_orchosen.extend(or_list_combined[math.floor(index / 2)])
        list_cross_orchosen = list(set(list_cross_orchosen))
        new_gene1_or_chosen, new_gene2_or_chosen = [0 for _ in range(l)], [0 for _ in range(l)]
        for i in range(l):
            if i in list_cross_orchosen:
                new_gene1_or_chosen[i], new_gene2_or_chosen[i] = gene2[i+2*num_process], gene1[i+2*num_process]
            else:
                new_gene1_or_chosen[i], new_gene2_or_chosen[i] = gene1[i+2*num_process], gene2[i+2*num_process]

        # 对machine进行交叉：list中的互换，list外的不变
        n = random.randint(math.ceil(num_process/3), math.floor(num_process*2/3))
        list_cross_machine = sorted(random.sample(range(num_process), n))
        new_gene1_machine, new_gene2_machine = [0 for _ in range(num_process)], [0 for _ in range(num_process)]
        for i in range(num_process):
            if i in list_cross_machine:
                new_gene1_machine[i], new_gene2_machine[i] = gene2[i], gene1[i]
            else:
                new_gene1_machine[i], new_gene2_machine[i] = gene1[i], gene2[i]

        # 对order进行交叉：保证前后置关系满足的交叉方法
        n = random.randint(0, num_process-1)
        gene1_order, gene2_order = gene1[num_process:2*num_process], gene2[num_process:2*num_process]
        new_gene1_order, new_gene2_order = gene2_order[:n], gene1_order[:n]
        for i in gene1_order:
            if i not in new_gene1_order:
                new_gene1_order.append(i)
        for i in gene2_order:
            if i not in new_gene2_order:
                new_gene2_order.append(i)
        new_gene1 = new_gene1_machine + new_gene1_order + new_gene1_or_chosen
        new_gene2 = new_gene2_machine + new_gene2_order + new_gene2_or_chosen
        return new_gene1, new_gene2

    def __mutate(self, gene):
        """
        变异-任意交换两点的顺序，或任意修改一个工序的machine
        """
        num_process = self.num_process
        job = self.job
        mut_machine = self.mut_machine
        or_list = self.or_list
        or_list_combined = self.or_list_combined
        mut_orlist = self.mut_orlist
        gene_new = deepcopy(gene)
        r = random.random()
        if r<0.2:                  # 任意交换两点的顺序
            index1, index2 = sorted(np.random.choice(range(num_process), 2, replace=False))
            while job[gene_new[num_process+index1]] == job[gene_new[num_process+index2]]:
                index1, index2 = sorted(np.random.choice(range(num_process), 2, replace=False))
            gene_new[num_process+index1], gene_new[num_process+index2] = gene_new[num_process+index2], gene_new[num_process+index1]
        elif r<0.6:                                       # 任意修改一个工序的machine
            index = np.random.choice(mut_machine, 1)[0]
            machine_possible = deepcopy(self.machine[index])
            machine_possible.remove(gene_new[index])        # 找到除了当前machine之外可选择的machine并随机选择
            gene_new[index] = machine_possible[random.randrange(len(machine_possible))]
        else:                       # 任意修改一个or_list的选项
            or_list_chosen = gene[2*num_process:]
            index = np.random.choice(mut_orlist, 1)[0]  # 工序更换or型选择
            if index not in itertools.chain.from_iterable(or_list_combined):
                or_list_chosen[index] = 1-or_list_chosen[index]
            else:
                for i in range(len(or_list_combined)):
                    try:
                        j = or_list_combined[i].index(index)
                    except ValueError:
                        pass
                if j==1:                              # 确保嵌套型or统一
                    if or_list_chosen[index]!=-1:
                        or_list_chosen[index] = 1 - or_list_chosen[index]
                    else:
                        or_list_chosen[index-1] = 1 - or_list_chosen[index-1]
                        or_list_chosen[index] = random.randint(0,1)
                else:
                    or_list_chosen[index] = 1 - or_list_chosen[index]
                    if or_list_chosen[index+1]==-1:
                        or_list_chosen[index+1] = random.randint(0,1)
                    else:
                        or_list_chosen[index+1] = -1
            gene_new[2*num_process:] = or_list_chosen
        return gene_new

    def fit(self):
        process = self.process
        num_process = self.num_process
        job = self.job
        or_list = self.or_list
        num_epoch = self.num_epoch
        self.group = self.__encoding()
        self.deco = list(map(lambda item: self.decoding(item)[0], self.group))
        best_time = min(self.deco)
        best_gene = self.group[self.deco.index(min(self.deco))]
        print(f"epoch0: best time in this epoch: {best_time}")
        temp = best_time
        count_cst = 0

        for epoch in range(num_epoch):
            self.fitness = self.__fitness(self.deco)
            self.new_group = []
            while len(self.new_group) < self.num_group:
                self.sample = self.__choose(self.fitness, self.group)
                # 交叉
                if random.random() < self.proba_cross:
                    self.son = self.__cross(self.sample)
                else:
                    self.son = self.sample
                # 变异
                if random.random() < self.proba_mutate:
                    self.mute = list(map(self.__mutate, self.son))
                else:
                    self.mute = self.son
                self.new_group.extend(deepcopy(self.mute))

            if count_cst>=50:
                for _ in range(self.num_group):
                    ## 等概率随机输出是否选择
                    or_list_chosen = []
                    for i in range(len(self.or_list_time)):
                        if self.or_list_time[i] == [0, 0]:  # 当该分支不存在于当前问题中时
                            or_list_chosen.append(-1)
                        else:
                            or_list_chosen.append(random.randint(0, 1))
                    for i in range(len(self.or_list_time)):  # 确保嵌套型or统一
                        if or_list_chosen[i] == 0:
                            for j in or_list[i][1]:
                                if list(itertools.chain.from_iterable(itertools.chain.from_iterable(or_list[i:]))).count(
                                        j) > 1:
                                    pos_j = self.find_or(j)
                                    for p in pos_j:
                                        if p[0] != i:
                                            or_list_chosen[p[0]] = -1
                        if or_list_chosen[i] == 1:
                            for j in or_list[i][0]:
                                if list(itertools.chain.from_iterable(itertools.chain.from_iterable(or_list[i:]))).count(
                                        j) > 1:
                                    pos_j = self.find_or(j)
                                    for p in pos_j:
                                        if p[0] != i:
                                            or_list_chosen[p[0]] = -1
                    ## 等概率随机输出gene_machine(按工序原始顺序排序)
                    gene_machine = []
                    for i in range(num_process):
                        gene_machine.append(self.machine[i][random.randrange(len(self.machine[i]))])
                    ## 通过排序将随机序列整理成为符合前后置关系的序列输出gene_order
                    gene_order = random.sample(range(num_process), num_process)
                    pointer = 0  # 指针指向第一个
                    while pointer < len(gene_order) - 1:  # 类似quicksort的排序方法
                        pivot = gene_order[pointer]  # 指针指向的元素
                        preprocess_id = self.preprocess[pivot]
                        if len(preprocess_id) == 0:  # 如果指针指向的元素没有前置元素，指针向后挪动一位
                            pointer = pointer + 1
                        else:  # 如果指针指向的有前置元素，找出所有前置的元素序号
                            indexesingene_pre = []
                            for id in preprocess_id:
                                for index, value in enumerate(process):
                                    if value == id:
                                        indexesingene_pre.append(gene_order.index(index))  # 通过序号定位其在当前gene排序中的index
                            if len(indexesingene_pre) == 0 or pointer > max(
                                    indexesingene_pre):  # 如果所有前置元素位置都在pivot之前或前置元素不在当前列表中，指针向后挪动一位
                                pointer = pointer + 1
                            else:  # 否则，找到位置最后的前置元素，并将pivot移动到该元素后一位，其余顺序不变
                                gene_order[pointer:max(indexesingene_pre)], gene_order[max(indexesingene_pre)] = \
                                    gene_order[pointer + 1:max(indexesingene_pre) + 1], gene_order[pointer]
                    if gene_machine + gene_order + or_list_chosen not in self.new_group:
                        self.new_group.append(gene_machine + gene_order + or_list_chosen)

            # 将子种群与父种群合并并去重
            for gene in self.group:
                if gene not in self.new_group:
                    self.new_group.append(gene)
            # 留下0.1num_group的最优解，随后使用三元锦标赛选择法从合并去重后的种群里选择组成新种群替代原种群
            deco_new_group = list(map(lambda item: self.decoding(item)[0], self.new_group))
            self.group = []
            while len(self.group) < 0.1*self.num_group:
                index = deco_new_group.index(min(deco_new_group))
                self.group.append(self.new_group[index])
                self.new_group.pop(index)
                deco_new_group.pop(index)
            while len(self.group)<self.num_group:
                index1, index2, index3 = random.sample(range(len(self.new_group)), 3)
                if deco_new_group[index1] <= deco_new_group[index2]:
                    index = index1
                else:
                    index = index2
                if deco_new_group[index] > deco_new_group[index3]:
                    index = index3
                self.group.append(self.new_group[index])
                self.new_group.pop(index)
                deco_new_group.pop(index)

            self.deco = list(map(lambda item: self.decoding(item)[0], self.group))

            # 对适应度前10的基因串进行order的邻域搜索
            list_index_sort = list(np.array(self.deco).argsort())
            for i in list_index_sort[:10]:
                count = 0
                replace = True
                while count < 3 and replace == True:
                    count += 1
                    replace = False
                    gene = deepcopy(self.group[i])
                    gene_order = gene[num_process:2*num_process]
                    total_time = self.decoding(gene)[0]
                    gene_finishtime = self.decoding(gene)[2]
                    last_processes = []
                    for index, value in enumerate(gene_finishtime):  # 找到所有最后结束的工序
                        if value == total_time:
                            last_processes.append(index)
                    for p in last_processes:  # 工序移动到可移动的前面随机位置（即满足前后置关系的最早点与当前位置中间）
                        index_now = gene_order.index(p)
                        indexes_pre = []
                        for p_pre in self.preprocess[p]:
                            indexes_pre.append(gene_order.index(self.process.index(p_pre)))
                        if indexes_pre == []:
                            index_min = 0
                        else:
                            index_min = max(indexes_pre) + 1
                        if index_now > index_min + 1:
                            index_new = random.randint(index_min, index_now - 1)
                            gene_order[index_new], gene_order[index_new+1:index_now+1] = gene_order[index_now], gene_order[
                                                                                                  index_new:index_now]
                        elif index_now == index_min + 1:
                            index_new = index_now - 1
                            gene_order[index_new], gene_order[index_now] = gene_order[index_now], gene_order[index_new]
                        gene[num_process:2*num_process] = gene_order
                    if self.decoding(gene) <= self.decoding(self.group[i]) and gene not in self.group:  # 如果最后的工序前移之后更好，则更换为此基因序列
                        self.group[i] = gene
                        replace = True
                        # print("order邻域搜索到更优解")

            # 对适应度前10的基因串进行machine的邻域搜索
            list_index_sort = list(np.array(self.deco).argsort())
            for i in list_index_sort[:20]:
                count = 0
                while count < 5:
                    count += 1
                    gene = deepcopy(self.group[i])
                    gene_machine = gene[:num_process]
                    total_time = self.decoding(gene)[0]
                    gene_finishtime = self.decoding(gene)[2]
                    machine_use = self.decoding(gene)[3]
                    last_machines = []
                    for index, value in enumerate(gene_finishtime):  # 找到所有最后结束的机器的序号（非123，是012）
                        if value == total_time:
                            last_machines.append(gene_machine[index]-1)
                    p_used = []
                    for m in last_machines:  # 机器上的工序更换随机机器
                        p_used.extend(machine_use[m])
                    for p in p_used:
                        gene_machine[p] = self.machine[p][random.randrange(len(self.machine[p]))]
                    if gene_machine!=gene[:num_process] and gene_machine+gene[num_process:2*num_process] not in self.group:
                        gene[:num_process] = gene_machine
                        if self.decoding(gene) <= self.decoding(self.group[i]):  # 如果更好，则更换为此基因序列
                            self.group[i] = gene
                            # print("machine邻域搜索到更优解")

            # 对适应度前10的基因串进行最后结束的job的邻域搜索（尝试更换machine）
            list_index_sort = list(np.array(self.deco).argsort())
            for i in list_index_sort[:30]:
                count = 0
                while count < 5:
                    count += 1
                    gene = deepcopy(self.group[i])
                    gene_machine = gene[:num_process]
                    total_time = self.decoding(gene)[0]
                    job_finishtime = self.decoding(gene)[4]
                    last_jobs = []
                    for index, value in enumerate(job_finishtime):  # 找到所有最后结束的工序job
                        if value == total_time:
                            last_jobs.append(index+1)
                    for p in range(num_process):  # 工序更换随机机器
                        if job[p] in last_jobs:
                            gene_machine[p] = self.machine[p][random.randrange(len(self.machine[p]))]
                    if gene_machine + gene[num_process:2 * num_process] not in self.group:
                        gene[:num_process] = gene_machine
                        if self.decoding(gene) < self.decoding(self.group[i]):  # 如果更好，则更换为此基因序列
                            self.group[i] = gene
                            # print("job - machine邻域搜索到更优解")

            # 对适应度前10的基因串进行job的邻域搜索（尝试更换or型选择及顺序）
            list_index_sort = list(np.array(self.deco).argsort())
            for i in list_index_sort[:30]:
                count = 0
                while count < 5:
                    count += 1
                    gene = deepcopy(self.group[i])
                    gene_order = gene[num_process:2 * num_process]
                    or_list_chosen = gene[2 * num_process:]
                    total_time = self.decoding(gene)[0]
                    job_finishtime = self.decoding(gene)[4]
                    last_jobs = []
                    for index, value in enumerate(job_finishtime):  # 找到所有最后结束的工序job
                        if value == total_time:
                            last_jobs.append(index + 1)
                    processes_job = []
                    for p in range(num_process):
                        if job[p] in last_jobs:
                            processes_job.append(p)
                    processes_job = random.sample(processes_job, len(processes_job))
                    for p in gene_order:            # 工序更换or型选择及顺序
                        if job[p] in last_jobs:                 # 工序更换顺序
                            gene_order[gene_order.index(p)] = processes_job[0]
                            processes_job.pop(0)
                    for j in range(len(or_list_chosen)):        # 工序更换or型选择
                        job_process1 = or_list[j][0][0]
                        if job_process1[1]=='-':
                            if int(job_process1[0]) in last_jobs:
                                or_list_chosen[j] = random.randint(0,1)
                        else:
                            if int(job_process1[1])+10 in last_jobs:
                                or_list_chosen[j] = random.randint(0,1)
                    for k in range(len(or_list_chosen)):    # 确保嵌套型or统一
                        if or_list_chosen[k] == 0:
                            for j in or_list[k][1]:
                                if list(itertools.chain.from_iterable(itertools.chain.from_iterable(or_list[k:]))).count(j) > 1:
                                    pos_j = self.find_or(j)
                                    for p in pos_j:
                                        if p[0] != k:
                                            or_list_chosen[p[0]] = -1
                        if or_list_chosen[k] == 1:
                            for j in or_list[k][0]:
                                if list(itertools.chain.from_iterable(itertools.chain.from_iterable(or_list[k:]))).count(j) > 1:
                                    pos_j = self.find_or(j)
                                    for p in pos_j:
                                        if p[0] != k:
                                            or_list_chosen[p[0]] = -1
                    if gene[:num_process]+gene_order+or_list_chosen not in self.group:
                        gene[num_process:] = gene_order+or_list_chosen
                        if self.decoding(gene) < self.decoding(self.group[i]):  # 如果更好，则更换为此基因序列
                            self.group[i] = gene
                            # print("job - order邻域搜索到更优解")

            temp = min(self.deco)
            temp_seq = self.group[self.deco.index(temp)]
            print(f"epoch{epoch+1}: best time in this epoch: {temp}, best time before: {best_time}")
            if temp < best_time:
                best_time = temp
                best_gene = deepcopy(temp_seq)
                count_cst = 0
            elif temp==best_time:
                count_cst += 1

        print("最优时间:", best_time)
        print("最优基因序列:", best_gene)
        gene_starttime = self.decoding(best_gene)[1]
        gene_finishtime = self.decoding(best_gene)[2]
        gene_machine = best_gene[:num_process]
        for i in range(num_process):
            print(f"工序J{self.process[i]}，开始时间：{gene_starttime[i]}，结束时间：{gene_finishtime[i]}，机器编号：{gene_machine[i]}")
        self.gante(best_gene)

        return best_time, best_gene

    def gante(self, gene):
        ## 解码后画出甘特图
        num_job = self.num_job
        process = self.process
        job = self.job
        num_process = self.num_process
        num_machine = self.num_machine
        or_list = self.or_list
        gene_starttime, gene_finishtime, machine_use = self.decoding(gene)[1], self.decoding(gene)[2], self.decoding(gene)[3]
        or_list_chosen = gene[2*num_process:]

        process_chosen = []
        process_not_chosen = []
        for i in range(len(or_list)):
            if or_list_chosen[i] != 0:
                process_not_chosen.extend(or_list[i][0])
            if or_list_chosen[i] != 1:
                process_not_chosen.extend(or_list[i][1])
        for i in process:
            if i not in process_not_chosen:
                process_chosen.append(i)

        process_list, y, width, left, color, y1, y_labels = [], [], [], [], [], [], []
        color_list = []                             #记录每个工序的颜色
        color_list_job = ['bisque', 'lightgreen', 'lightsteelblue', 'burlywood', 'lime', 'silver',
                          'slateblue', 'aquamarine', 'gold', 'lightcoral', 'teal', 'plum',
                          'yellow', 'fuchsia', 'lightblue', 'hotpink', 'sandybrown', 'pink']
        for i in range(num_process):
            color_list.append(color_list_job[job[i]-1])

        for i in range(num_machine):
            for j in range(len(machine_use[i])):
                index_process = machine_use[i][j]
                process_list.append(process[index_process])
                ## 构造y轴
                y.append(num_machine - i)
                ## 构造width
                width.append(gene_finishtime[index_process] - gene_starttime[index_process])
                ## 构造left
                left.append(gene_starttime[index_process])
                ## 构造color
                color.append(color_list[index_process])

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.figure(dpi=80)
        plt.barh(y, width, left=left, color=color, edgecolor='black')  # 绘制水平直方图
        for i in range(len(process_chosen)):
            plt.text(left[i]+width[i]/2, y[i], "J%s" % process_list[i], fontsize=12, fontweight='heavy', va='center', ha='center')
        # XY轴标签
        plt.xlabel("时间")
        for i in range(num_machine):
            y1.append(num_machine-i)
            ## 构造y轴标签
            y_labels.append("机器{}".format(i+1))
        plt.yticks(y1, y_labels)
        plt.show()  # 显示图像


starttime = datetime.datetime.now()
test=GA_solve(params)
best_time, best_gene=test.fit()
endtime = datetime.datetime.now()
print(endtime-starttime)