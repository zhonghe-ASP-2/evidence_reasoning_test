import numpy as np
from numpy.lib.function_base import append
import pandas as pd
from collections import Counter

#置信度1*a1+置信度2*a2    设置在def testresult(M,M1,M2,M3,M4,M5,M6, r_ind):函数中
'''---叶子节点部分---'''
'''调用py文件'''
import rule_calculation
'''读取数据'''
def readdata():
    c0, c1, c2, c3 = rule_calculation.parmetersv()    #读入规则库c, 原因向量c0和预测概率c1
    c0 = np.array(c0)

    return c0, c1, c2
def readq():
    c0, c1, c2, c3 = rule_calculation.parmetersv()    #读入规则库c, 原因向量c0和预测概率c1
    c3 = np.array(c3)
    return c3

def find_reason(R):
    r_ind = []
    for j in range(len(R)):
        if R[j] == 1:
            r_ind.append(j)
    r_ind = np.array(r_ind)
    return r_ind

def delete_reason(B, r):
    for i in range(len(r)):
        for j in range(len(r[i])):
            B[i].remove(r[i][j])
    return B

def reason_flag(B, r, E):
    r_f = np.zeros(len(r))
    c_f = np.zeros(len(B))
    flag = np.zeros(len(B))
    for i in range(len(B)):
        for c in B[i]:
            if c not in E:
                c_f[i] = c_f[i] + 1
    for i in range(len(c_f)):
        if c_f[i] != 0:
            c_f[i] = 0
        else:
            c_f[i] = 1                               #趋势阈值征兆满足情况 规则内趋势阈值征兆全部满足为1
    for i in range(len(r)):
        for c in r[i]:
            if c not in E:
                r_f[i] = r_f[i] + 1
    for i in range(len(r_f)):
        if r_f[i] != 0:
            r_f[i] = 0
        else:
            r_f[i] = 1                               #规则内原因征兆满足为1
    for i in range(len(flag)):
        flag[i] = (int(c_f[i])<<1)|int(r_f[i])
    return flag

def separate(B1a, Ea, wa, B1ia, Ba, rula, Eia, rul_r):
    B1 = np.zeros((B1a.shape[0], len(rul_r)))
    E = []
    w = np.zeros((wa.shape[0], len(rul_r)))
    B = np.zeros((Ba.shape[0], len(rul_r)))
    rul = np.zeros((rula.shape[0], len(rul_r)))
    for i in range(len(rul_r)):
        B1[:, i] = B1a[:, rul_r[i]]
        w[:, i] = wa[:, rul_r[i]]
        E.append(Ea[rul_r[i]])
        B[:, i] = Ba[:, rul_r[i]]
        rul[:, i] = rula[:, rul_r[i]]


    n = B1.shape[1]  # 列数
    r = B1.shape[0]  # 行数
    Ei = [i for i, x in enumerate(E) if x == 1]  # Ei为E中“满足”的索引
    B1i = []
    for i in range(r):
        B1i.append([i for i, x in enumerate(B[i, :]) if x > 0])  # B1i第i行元素为B第i行元素>0的索引

    return n, r, E, w, B1i, B, rul, Ei


'''规则-模式'''
def dictrule(D):
    r = D.shape[0]                                                          #第一维长度
    dic1 = {}
    for i in range(1,r):
        if D[i,0] != 0:
            dic1[D[i,0]] = D[i,-1]          #规则-模式
    return dic1
'''值转换'''
def transformvalue(x):
    doucment = {'确定1':1,'高(0.5-1)':0.75,'低(0-0.5]':0.25,'不可能0':0}
    #将高中低转换为对应数值
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i,j] in doucment:
                x[i,j] = doucment[x[i,j]]
    return x
'''处理数据'''
def dealdata(D, D1):
    list2 = []
    list3 = []
    for i in range(1, D.shape[0]):
        list2.append(D[i,0])
        if D[i, -1] not in list3:
            list3.append(D[i,-1]) 
    list4=[]
    for j in range(len(list3)):
        list4.append([])
        for i in range(1,D.shape[0]):
            if D[i, -1]==list3[j]:
                list4[j].append(i-1)
    D = transformvalue(D)[:, 1:-1]
    D_ = np.zeros([D.shape[0]-1, D.shape[1]])
    for i in range(len(D1)):
        if D[0, i] in D1:                                               # ???
            inde = D1.index(D[0, i])
            for j in range(D.shape[0]-1):
                    D_[j, inde] = D[j+1, i]
    return D_,list2,list4
'''归一化置信度'''
def ruleconfidence(C2,C3):
    n = C2.shape[1]                         #n：C2列数
    C1 = np.zeros(C2.shape)                 #归一化置信矩阵度容器 C1与C2大小一致
    C4 = np.zeros([len(C3), n])             #C4{0} 行：C3长度（行？），列：C2列数
    # 对每一列和>1x进行归一化计算
    for i in range(n):
        su=0
        for j in range(len(C3)):
            sum=0
            sun=0
            for  j1 in range(len(C3[j])):
                sum=sum+C2[C3[j][j1],i]
                if C2[C3[j][j1],i]!=0:
                    sun=sun+1
            if sun!=0:
                su=su+sum/sun                                           #???
                C4[j,i]=sum/sun
        if su > 1:
            C1[:,i] = C2[:,i]/su
        else:
            C1[:,i] = C2[:,i]
    return C1,C4
'''数值类型转换'''
def transformtype(x):
    x2 = np.zeros(x.shape)
    for i in range(x2.shape[0]):
        for j in range(x2.shape[1]):
            x2[i, j] = float(x[i,j])
    return x2


'''预测置信度'''
def preconfidence(X,E,C):
    #类型转换
    x1 = transformtype(X)                   #转换为float型
    c1 = transformtype(C)
    n = X.shape[1]                          #列数
    r = X.shape[0]                          #行数
    pre1 = np.ones([r,n])                   #预测概率矩阵容器
    X1 = x1                                 #预测置信矩阵度容器
    for i in range(n):
        if E[i] == "预测":
            pre1[0,i] = c1[0,i]             # [0, i]???
        X1[:,i] = pre1[0,i]*x1[:,i]
    return X1
'''计算概率指派'''
def calculateprobability(A,w):
    n = A.shape[1]   
    r = A.shape[0]                          #列数
    M1 = np.matrix(np.zeros(A.shape))       #指派概率矩阵容器   （和直接用np.zeros()有什么区别？matrix可以使用矩阵运算？）
    for i in range(n):
        for j in range(r):
            M1[j,i] = w[0,i]*A[j,i]         #M1=A列每个元素乘以w对应列元素
    return M1
'''检索位置'''
def myfind(x,y): return [ a for a in range(len(y)) if y[a] == x]
'''检索非零位置'''
def nonzerosfind(X,Y):return [X[a] for a in range(min(len(X),len(Y))) if Y[a]!=0  ]
'''合成方法'''
def fusemethod(A,A0,A4,w):
    A0 = np.matrix(A0)
    r = A0.shape[0]                          #ru_行数
    n = A0.shape[1]
    MH = 1-w                                # -
    MH1 = np.zeros([1,n])                   # ~
    for i in range(n):
        tot = 0
        for j in range(r):
            tot = tot+A0[j,i]               #列和
        MH1[0,i] = w[0,i]*(1-tot)           # （1-列和）？ 放在MH1中（1行）
    MH2 = MH + MH1       
    pos = myfind(1,A4)                  #满足条件的失效征兆索引（E_中）
    n1 = len(pos)                           #满足条件的失效征兆个数
    b = np.zeros([r,1])                     #融合结果向量容器
    M = calculateprobability(A0,w)          #M = A0列每个元素*w对应列元素 Mn,i
    #融合算法
    A1=MH                                   # m-H,i
    A2=MH1                                  # m~H,i
    A3=MH2                                  # mH,i
    for i in range(r):
        list1 = []
        for j in range(n1):
            list1.append(M[i, pos[j]])       #每一行置信度 M （失效征兆索引对应）   pos是否对应i？
        num = sum(i!=0 for i in list1)      #置信度不为0的个数
        '''
        if num == 0:
            b[i,0] = 0
        if num == 1:
            pos1 = nonzerosfind(pos,list1)  #list1中非零元素位置
            list1.remove(0)                 #去除0元素
            b[i,0] = A0[i,pos1[0]]
        '''
        if A0.shape[0]==1:                   # ??? K为何为1
            pos1 = nonzerosfind(pos,list1)  #list1中非零元素位置
            m = np.zeros([r,1])             #融合的基本概率指派矩阵容器
            #ER中的融合算法
            for k in range(r):
                m[k,0] = M[k,pos1[0]]
            mh = A1[0,pos1[0]]
            mh1 = A2[0,pos1[0]]
            mh2 = A3[0,pos1[0]]
            for k in range(1,len(pos1)):
                K = 1
                for j1 in range(r):
                    m[j1] = K*(m[j1]*M[j1,pos1[k]]+m[j1]*A3[0,pos1[k]]+mh2*M[j1,pos1[k]])
                mh1 = K * (mh1 * A2[0,pos1[k]] + mh * A2[0,pos1[k]] + mh1 * A1[0,pos1[k]])
                mh = K * mh * A1[0,pos1[k]]
                mh2 = mh + mh1
            for k in range(r):
                if i==k:                    # ???
                    b[i,0] = m[k]/(1-mh)    # 个体被评价为等级Hn的置信度
        if A0.shape[0] > 1:                 # ???
            pos1 = nonzerosfind(pos,list1)  #list1中非零元素位置
            m = np.zeros([r,1])             #融合的基本概率指派矩阵容器
            #ER中的融合算法
            for k in range(r):
                m[k,0] = M[k,pos1[0]]
            mh = A1[0,pos1[0]]
            mh1 = A2[0,pos1[0]]
            mh2 = A3[0,pos1[0]]
            for k in range(1,len(pos1)):
                tot = 0
                for j1 in range(r):
                    for j2 in range(r):
                        if j1 != j2:
                            tot = tot + m[j1]*M[j2,pos1[k]]
                K = 1/(1-tot)
                for j1 in range(r):
                    m[j1] = K*(m[j1]*M[j1,pos1[k]]+m[j1]*A3[0,pos1[k]]+mh2*M[j1,pos1[k]])
                mh1 = K * (mh1 * A2[0,pos1[k]] + mh * A2[0,pos1[k]] + mh1 * A1[0,pos1[k]])
                mh = K * mh * A1[0,pos1[k]]
                mh2 = mh + mh1
            for k in range(r):
                if i==k:
                    b[i,0] = m[k]/(1-mh)
    b1=b[A,0]
    return b1

#判断满足失效情况
def subER(n, r, E, w, B1i, B, rul, Ei, C1):
    for i in range(n):                      # E中=‘无’的元素对应w位置也为0
        if E[i]== 0:
            w[0,i] = 0
    num = Counter(E)[1]                 #失效征兆个数
    b = np.zeros([r, 1])
    if num == 1:
        for i1 in range(r):                     # i1的作用？
            if B1i[i1]==Ei:
                for i in  range(n):
                    if E[i] == 1:
                        for j in range(r):
                            b[j, 0] = B[j, i]     # 满足的征兆对应不同规则置信度（归一前）
    if num > 1:
        for i in range(r):
            w_ = np.zeros(w.shape)                            #w_：与w相同
            for j in range(w.shape[0]):
                w_[j] = w[j]
            E_ = []
            for j in range(len(E)):
                E_.append(E[j])                             #E_：与E相同（可不为数值）
            rul_=np.zeros(rul.shape)
            for j1 in range(rul.shape[0]):
                for j2 in range(rul.shape[1]):
                    rul_[j1,j2]=rul[j1,j2]                  # rul_: 与rul相同（数值）？
            Pan=[False for c in B1i[i] if c not in Ei]      #E满足，但B值不大于0，Pan = False
            nm=0
            if Pan==[]:                                     #E包含B的所有检索
                for j in range(len(C1)):
                    if i in C1[j]:
                        nm = j                              #失效模式
                low_n = len(B1i[i])                         #规则中等级为低的个数
                for j in range(len(B1i[i])):                # 判断规则中等级是否都为低
                    if B[i, B1i[i][j]] < 0.5:
                        low_n = low_n - 1
                for j in range(n):                              # B中为0的项（规则不满足
                    if B[i, j] == 0:
                        w_[0, j] = 0
                        E_[j] = 0


                if low_n != 0:                              # 若都为低，则不做改变，继续计算
                    for j in range(n):                          # 若不都为低，则留高去低
                        if B[i, j] < 0.5:
                            w_[0, j] = 0
                            E_[j] = 0

                num1 = Counter(E_)[1]                  # 重新计算满足个数（去掉低）
#                if num1 == 0:
#                    return flag, F, b
                if num1 == 1:
                    for i1 in  range(n):
                        if E_[i1] == 1:
                            b[i,0] = B[i,i1]                # 和上一个num==1的区别？
                if num1>1:
                    b[i,0] = fusemethod(nm, rul_, E_, w_)      #融合

    return b

'''ER算法'''
def Evidentialreasoning(A,w2,E,C,C1, r_ind):
    w = np.zeros(w2.shape)                  #权重向量容器
    for i in range(w2.shape[1]):            # w = w2
        for j in range(w2.shape[0]):
            w[j,i] = w2[j,i]
    B = preconfidence(A,E,C)                #预测置信度计算 若E(i)为预测，B=A*C， 其余为B
    B1, rul = ruleconfidence(B,C1)           #置信度归一化
    for i in range(rul.shape[1]):           #rul归一化
        sum = np.sum(rul[:,i])
        if sum>1:
            rul[:,i] = rul[:,i]/sum
    n = B1.shape[1]                         #列数
    r = B1.shape[0]                         #行数
    for i in range(n):
        if E[i] == "预测":
            E[i] = 1
    Ei=[i for i,x in enumerate(E) if x== 1]               #Ei为E中“满足”的索引
    B1i=[]
    for i in range(r):
        B1i.append([i for i,x in enumerate(B[i,:]) if x>0])   #B1i第i行元素为B第i行元素>0的索引
    rul_r = []
    for i in range(r):
        rul_r.append([j for j, x in enumerate(B[i,:]) if x>0 and j in r_ind]) #各条规则中 原因征兆的索引
    B1i_ = delete_reason(B1i, rul_r)                                      #各条规则 趋势阈值征兆索引
    c_ind = []
    for i in range(B.shape[1]):
        if i not in r_ind:
            c_ind.append(i)

    flag = reason_flag(B1i_, rul_r, Ei)
    flag_dic = {3: 'D', 2: 'C', 1: 'B', 0: 'A'}
    F = []
    for i in range(len(flag)):
        F.append(flag_dic[int(flag[i])])


    n_1, r_1, E_1, w_1, B1i_1, B_1, rul_1, Ei_1 = separate(B1, E, w, B1i, B, rul, Ei, c_ind)
    n_2, r_2, E_2, w_2, B1i_2, B_2, rul_2, Ei_2 = separate(B1, E, w, B1i, B, rul, Ei, r_ind)


    b1 = subER(n_1, r_1, E_1, w_1, B1i_1, B_1, rul_1, Ei_1, C1)                                 #趋势阈值征兆ER融合后
    b2 = subER(n_2, r_2, E_2, w_2, B1i_2, B_2, rul_2, Ei_2, C1)                                 #原因征兆ER融合后

    return flag, F, b1, b2
'''规则结果展示'''
def resultshow(R,R1):
    dic = {}
    for i in range(len(R)):
 #       R[i, 0] = float(R[i, 0])
        dic[R1[i]] = int(R[i, 0]*10000)/10000
 #       dic[R1[i]] ='%.4f' % float(R[i, 0])                        #dic中R1元素分别对应R1元素（取4位小数）
    return dic
'''样例结果计算'''
def calculate_result(M,M1,M2,M3,M4,M5,M6, r_ind):
    a1 = 80                                                 #趋势阈值征兆置信度权重
    a2 = 100 - a1                                             #原因征兆置信度权重
    flag, f_dic, R_M1,R_M2 = Evidentialreasoning(M, M1, M2, M3, M6, r_ind)             #RM=ER算法
    rul_alarm = {}
    for i in range(len(M4)):                                    #各规则的报警等级
        rul_alarm[M4[i]] = f_dic[i]
    R_M = np.zeros((R_M1.shape[0], 1))
    for i in range(len(f_dic)):
        if f_dic[i] == 'D':
            R_M[i][0] = R_M1[i][0] * a1 + R_M2[i][0] * a2
        if f_dic[i] == 'C':
            R_M[i] = R_M1[i]
        if f_dic[i] == 'A' or f_dic[i] == 'B':
            R_M[i] = 0


    dd = resultshow(R_M, M4)                                # dd：dict('M4":R_M, ...)

    # print(dd)                                        # {rule1: 置信度1, rule2:置信度2, ...}
    # print(rul_alarm)                          # {rule1: alarm1, rule2:alarm2, ...}
    # print(M5)                                        # {rule1: fault mode1, rule2 fault mode2, ... }

    dic_list1 = []
    di_di = {}
    for i in range(len(M4)):
        if M5[M4[i]] in dic_list1:
            continue
        else:
            dic_list1.append(M5[M4[i]])
    alarm = {}

    an = np.zeros(len(M6))
    an_h = np.zeros(len(M6))
    for i in range(len(dic_list1)):
        alarm[dic_list1[i]] = flag[M6[i][0]]
        for j in range(len(M6[i])):
            if int(alarm[dic_list1[i]]) <= int(flag[M6[i][j]]):
                alarm[dic_list1[i]] = flag[M6[i][j]]
                an[i] = M6[i][j]                                     # 各个失效模式报警级别最高的规则索引
                an_h[i] = int(flag[M6[i][j]])                       # 各个失效模式报警类型（0,1,2,3）

    for i in range(len(alarm)):
        alarm[dic_list1[i]] = f_dic[int(an[i])]                              #各个失效模式报警文字
    # print(alarm)

    for i in range(len(dic_list1)):
        if an_h[i] == 0 or an_h[i] == 1:
            di_di[dic_list1[i]] = int(0*1000)/1000
        else:
            di_di[dic_list1[i]] = dd[M4[int(an[i])]]
            for j in range(len(M6[i])):
                if int(flag[M6[i][j]]) == an_h[i]:
                    if di_di[dic_list1[i]] <= dd[M4[M6[i][j]]]:
                        di_di[dic_list1[i]] = dd[M4[M6[i][j]]]



    '''for i in range(len(dic_list1)):                         #di_di dic's name
        dic_list2 = []
        for j in range(len(M4)):
            if M5[M4[j]] == dic_list1[i]:
                dic_list2.append(dd[M4[j]])
        di_di[dic_list1[i]] = max(dic_list2)                #di_di dic's value'''
    # print(di_di)                                                    #各个失效模式置信度
    return dd, rul_alarm, M5, di_di, alarm
'''约简'''
def removeun(Q, Q1, Q0, rea):
    Q2 = []
    for i in range(1, Q.shape[1]-1):
        Q2.append(Q[0, i])                                   #Q2 = Q[0,1:-1]
    Q3 = []
    Q4 = []
    Q5 = []
    for i in range(len(Q2)):
        if Q2[i] in Q0:
            ind = Q0.index(Q2[i])
            Q3.append(Q1[ind])                             #value?
            Q5.append(rea[ind])                            #原因标识
            Q4.append(Q0[ind])                             #attribute?
        else:
            Q3.append(0)  # value?
            Q5.append(0)
            Q4.append(Q2[i])  # attribute?

    # print(Q3)
    # print(Q4)
    return Q3, Q4, Q5

'''---中间节点部分---'''
'''读取数据'''
def readdata1(path):
    data = pd.read_excel(path).values               #二维数组形式
    return data
'''置信度计算'''
def calconfidence(A,A1):
    ro = len(A)
    A_r = np.zeros([ro,1])
    for i in range(ro):
        A_r[i,0] =  float(A[i][-1])*float(A1[A[i][0]])  #A最后一列*A1第一列
    return A_r
'''计算高一层的传递置信度'''
def maxconfidence(AR,AR_):
    r1 = AR.shape[0]
    print('传递概率：',AR[:,-1])                         #传递概率为AR最后一行
    mode_key = []
    mode_va = []
    while(AR_!={}):
        lis1 = []
        lis2 = []
        for i in range(r1):
            if AR[i,0] in AR_:
                lis2.append(i)
                lis1.append(AR[i,:])
        AR1 = calconfidence(lis1, AR_)
        list1 = []
        AR_ = {}
        for j in range(r1):
            if j in lis2:
                if AR[j, 1] in list1:
                    continue
                else:
                    list1.append(AR[j, 1])
                    mode_key.append(AR[j, 1])
        for j in range(len(list1)):
            list2 = []
            for j1 in range(r1):
                if j1 in lis2:
                    if AR[j1, 1] == list1[j]:
                        ind1 = lis2.index(j1)
                        list2.append('%.4f' % AR1[ind1, -1])

            AR_[list1[j]] = max(list2)
            mode_va.append(max(list2))

        if AR_ == {}:
            break
        print(AR_)
    mode_li = []
    mode_va1 = []
    for i in range(len(mode_key)):
        if mode_key[i] in mode_li:
            mode_va1[mode_li.index(mode_key[i])] = max(mode_va[i],mode_va1[mode_li.index(mode_key[i])])
        else:
            mode_li.append(mode_key[i])
            mode_va1.append(mode_va[i])
    mode_dic = {}
    for i in range(len(mode_li)):
        mode_dic[mode_li[i]] = mode_va1[i]
    print(mode_dic)
'''失效方法计算'''
def methondcal(data, sign, reason, omen):
    print('>>>>>叶子节点计算：')
    # data, sign, reason = readdata()
    pre = np.matrix([1,0.1,0.2,0.5])
    E, E0, rea = removeun(data, omen, sign, reason)                                      # E:满足情况, E0:征兆
    reason_ind = find_reason(rea)
    data1, RU,dic_m = dealdata(data, E0)
    di = dictrule(data)
    w1 = np.ones([1, len(E)])  # 不同失效模式下的失效征兆权重
    '''失效模式置信度结果'''
    calculated_rules, rule2failuremode, failuremodes, failuremodes_alarm_map = calculate_result(data1, w1, E, pre, RU, di, dic_m, reason_ind)
    # print('>>>>>中间节点计算：')
    # inp = calculated_rules
    # dat = readdata1("./2.xlsx")
    # maxconfidence(dat, inp)
    return calculated_rules, rule2failuremode, failuremodes, failuremodes_alarm_map


def model_ER_A1():
    q = readq()

    #实时
#    q = [['征兆1', '征兆2', '征兆3', '征兆4', '征兆5'],
#         [1,1,1,1,      0]]
    #预测
#    q_p = [['征兆1', '征兆2', '征兆3', '征兆4', '征兆5'],
#            ["预测",       0,      0,       0,      0]]

    '''-----实时情况计算-----'''
    print('-----实时情况计算-----')
    methondcal(q)
    '''-----预测情况计算-----'''
  #  print('-----预测情况计算-----')
  #  methondcal(q_p)

if __name__ == '__main__':
    model_ER_A1()


