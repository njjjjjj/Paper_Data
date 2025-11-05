#种群初始化部分
from init import *
import numpy as np

#定义个体
class indiv:
    def __init__(self):
        #任务工作站分配矩阵
        self.TA=[-1 for _ in range(Ntask)]
        #任务模式选择
        self.AM=[-1 for _ in range(Ntask)]
        #工人分配矩阵
        self.WTS=[-1 for _ in range(NW)]
        #机器人分配矩阵
        self.CTS=[-1 for _ in range(NC)]

        #任务完成时间
        self.FT=[-1 for _ in range(Ntask)]
        #两个目标
        self.fitness=[-1 for _ in range(objnum)]

# 基于前序约束的直接生成策略
def initindiv1(mindiv, mode):
    # 初始化
    for i in range(Ntask):
        mindiv.TA[i] = -1
        mindiv.AM[i] = -1
    for i in range(NW):
        mindiv.WTS[i] = -1
    for i in range(NC):
        mindiv.CTS[i] = -1
        # 忽略函数中的输出，只运行
    # with open(os.devnull, 'w') as f:
    #    with contextlib.redirect_stdout(f):
    mindiv.TA = pregene(mode)
    # 生成模式,conumbers表示采用合作模式的工作站集合
    conumbers = ModeSelect(mindiv.AM, mindiv.TA)
    # print(f'资源分配前AM向量为：{mindiv.AM}')
    # 生成工人和机器人
    PSset = [[-1 for _ in range(Ntask)] for _ in range(NS * 2)]
    PSnum = [0 for _ in range(NS * 2)]
    mindiv.TA, mindiv.AM, mindiv.WTS, mindiv.CTS = WCA(mindiv.TA, mindiv.AM, mindiv.WTS, mindiv.CTS, PSset, PSnum,
                                                       conumbers)
    # print(f'资源分配后AM向量为：{mindiv.AM}')

#基于前序约束的生成调整策略
def initindiv2(mindiv,mode):
    #初始化
    for i in range(Ntask):
        mindiv.TA[i]=-1
        mindiv.AM[i]=-1
    for i in range(NW):
        mindiv.WTS[i]=-1
    for i in range(NC):
        mindiv.CTS[i]=-1
    #随机生成序列
    TSA=[-1 for _ in range(Ntask)]
    for i in range(Ntask):
        TSA[i]=random.randint(0, NS-1)
    TSA,mindiv.TA=adjust(TSA,mode)
    #print(f"任务站安排:{mindiv.TA}")
    #生成模式,conumbers表示采用合作模式的工作站集合
    conumbers=ModeSelect(mindiv.AM,mindiv.TA)
    # print(f'资源分配前AM向量为：{mindiv.AM}')
    #生成工人和机器人
    PSset=[[-1 for _ in range(Ntask)] for _ in range(NS*2)]
    PSnum=[0 for _ in range(NS*2)]
    mindiv.TA,mindiv.AM,mindiv.WTS,mindiv.CTS=WCA(mindiv.TA,mindiv.AM,mindiv.WTS,mindiv.CTS,PSset,PSnum,conumbers)
    # print(f'资源分配后AM向量为：{mindiv.AM}')

# 基于负载均衡的生成调整策略
def initindiv3(mindiv, mode):
    # 初始化
    for i in range(Ntask):
        mindiv.TA[i] = -1
        mindiv.AM[i] = -1
    for i in range(NW):
        mindiv.WTS[i] = -1
    for i in range(NC):
        mindiv.CTS[i] = -1

    # 生成工作站
    storeassign(mindiv.TA, Nlevel, mode)
    # 生成模式,conumbers表示采用合作模式的工作站集合
    conumbers = ModeSelect(mindiv.AM, mindiv.TA)
    # print(f'资源分配前AM向量为：{mindiv.AM}')
    # 生成工人和机器人
    PSset = [[-1 for _ in range(Ntask)] for _ in range(NS * 2)]
    PSnum = [0 for _ in range(NS * 2)]
    mindiv.TA, mindiv.AM, mindiv.WTS, mindiv.CTS = WCA(mindiv.TA, mindiv.AM, mindiv.WTS, mindiv.CTS, PSset, PSnum, conumbers)
    # print(f'资源分配后AM向量为：{mindiv.AM}')
def popinitial(mpop,pn):
    for i in range(pn):
        print(" ")
        print(f"===========第{i}个个体初始化方式为：")
        # mode=random.randint(0,1)
        mode = 1
        #生成随机数
        rand=random.randint(1,200)
        # if rand<100:
        #     print('init1')
        #     with open(os.devnull, 'w') as f:
        #         with contextlib.redirect_stdout(f):
        #             initindiv1(mpop[i], mode)
        #     print(f'第{i}个个体的编码为：')
        #     print(f'TA向量为:{mpop[i].TA}')
        #     print(f'AM向量为：{mpop[i].AM}')
        #     print(f'WTS向量为：{mpop[i].WTS}')
        #     print(f'CTS向量为：{mpop[i].CTS}')
        # else:
        #     if rand<200:
                # print('init2')
                # with open(os.devnull, 'w') as f:
                #     with contextlib.redirect_stdout(f):
                #         initindiv2(mpop[i], mode)
            # else:
        with open(os.devnull, 'w') as f:
            with contextlib.redirect_stdout(f):
                initindiv3(mpop[i], mode)
        # print(f'第{i}个个体的编码为：')
        # print(f'TA向量为:{mpop[i].TA}')
        # print(f'AM向量为：{mpop[i].AM}')
        # print(f'WTS向量为：{mpop[i].WTS}')
        # print(f'CTS向量为：{mpop[i].CTS}')


class Normalizer:
    def __init__(self, obj_num=3):
        # 初始化理想点和最差点
        self.ideal_point = np.array([float('inf')] * obj_num)
        self.nadir_point = np.array([float('-inf')] * obj_num)

    def update_bounds(self, objectives):
        """更新理想点和最差点"""
        for i in range(objectives.shape[1]):
            self.ideal_point[i] = min(self.ideal_point[i], np.min(objectives[:, i]))
            self.nadir_point[i] = max(self.nadir_point[i], np.max(objectives[:, i]))

    def normalize(self, objectives):
        """基于存储的边界进行归一化"""
        normalized = np.zeros_like(objectives, dtype=float)

        for i in range(objectives.shape[1]):
            # 避免除零错误
            if self.nadir_point[i] == self.ideal_point[i]:
                normalized[:, i] = 0.5
            else:
                normalized[:, i] = (objectives[:, i] - self.ideal_point[i]) / \
                                   (self.nadir_point[i] - self.ideal_point[i])

        return normalized


# minitpop = [indiv() for _ in range(10)]
# popinitial(minitpop, 10)


'''
print(" ")
for i in range(PN):
    # 机器和工人分配矩阵
    SW=[-1 for _ in range(NS*2)]
    SC=[-1 for _ in range(NS*2)]
    for j in range(NW):
        if testpop[i].WTS[j]!=-1:
            SW[testpop[i].WTS[j]]=j
    for j in range(NC):
        if testpop[i].CTS[j]!=-1:
            SC[testpop[i].CTS[j]]=j

    print(f"个体{i} TA为:{testpop[i].TA}")
    print(f"个体{i} AM为:{testpop[i].AM}")
    print(f"个体{i} 工作站-工人为:{SW}")
    print(f"个体{i} 工作站-机器人为:{SC}")
    print(f"SW={SW}")
    print(f"SC={SC}")
    #with open(os.devnull, 'w') as f:
    #   with contextlib.redirect_stdout(f):
    F1,F2,F3=decode(testpop[i].TA,testpop[i].AM,testpop[i].WTS,testpop[i].CTS)
    print(f"适应度为:",F1,F2,F3)'''
