# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 11:17:10 2023

@author: hp
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from C0_plotthing import get_fig,spatial_temporal_fig_set

def cal_para(arraylike,args):
    # Edie方法计算宏观参数
    delta_t = args[0]
    delta_x = args[1]
    da = arraylike['kilodelta'].sum()
    ta = arraylike['timedelta'].sum()
    area = delta_t*delta_x
    q = da/area
    k = ta/area
    v = da/ta
    speed_mu = arraylike['velocity'].mean()
    speed_var = arraylike['velocity'].var()
    if v>120/3.6:
        v = arraylike['velocity'].mean()/3.6
        q = len(arraylike)/delta_t
        k = q/v
    return [q,k,v,speed_mu,speed_var]

def cal_acc_field(speed_field):
    dx = np.concatenate([np.full((1,speed_field.shape[1]), np.nan),-np.diff(speed_field,axis=0)])
    dt = np.concatenate([np.diff(speed_field,axis=1),np.full((speed_field.shape[0],1),np.nan)],axis=1)
    acc = dx*speed_field + dt
    return acc

def cal_acc_field_rev(speed_field,t_step,x_step):
    print('Calculating acc...')
    acc = np.full(speed_field.shape,np.nan)
    for i in np.arange(acc.shape[0]):
        for j in np.arange(acc.shape[1]):
            # 转换单位为m/s2
            try:
                acc[i,j] = (speed_field[int(i-(t_step*speed_field[i,j])//x_step),j+1]-speed_field[i,j])/(3.6*t_step)
            except:
                continue
    return acc

def cal_DSD(macro_para,t_step,x_increase=True):
    # 计算dsd
    '''更改偏移量计算方法只用改这一小部分,后面的不用变'''
    macro_para['dsd'] = macro_para['v(km/h)']*12.1*0.278
    # 计算dsd对应的空间格子数
    t_len = len(set(macro_para['t']))
    x_step = macro_para.iloc[t_len]['x'] - macro_para.iloc[0]['x']
    if x_increase:          # 如果x的桩号是递增的就用这个,否则用下一个
        macro_para['dsd_x'] = macro_para['x'] + x_step*np.floor(macro_para['dsd']/x_step)
    else:
        macro_para['dsd_x'] = macro_para['x'] - x_step*np.floor(macro_para['dsd']/x_step)
    macro_para['dsd_t'] = macro_para['t'] + t_step*np.floor(12.1/t_step)
    
    up = max(macro_para['x'])
    low = min(macro_para['x'])
    t_up = max(macro_para['t'])
    m = macro_para[['dsd_t','dsd_x']]
    m = m[m['dsd_t']<t_up].values
    macro_para = macro_para[macro_para['dsd_t']<t_up]
    
    ka = []
    i = 0
    for data in m:
        i += 1
        if i%1000==0:
            print(i)
        if (data[1]<=up)&(data[1]>=low):
            kai = macro_para[(macro_para['t']==data[0])&(macro_para['x']>data[1]-0.1)&(macro_para['x']<data[1]+0.1)]['k']
            ka.append(kai.values)
        else:
            ka.append([np.nan])
            # ka.append(np.nan)
    macro_para['ka'] = np.array(ka)
    return macro_para,ka

def cal_speed(trajectory_data,delta_t=50,delta_x=300,t_step=20,x_step=20,plot_figure=False,rolling=True):
    # 计算路段空间平均速度,以30s和50m作为时空格子
    print(delta_t,delta_x)
    trajectory_data['x_label'] = trajectory_data['kilopost']//delta_x
    trajectory_data['t_label'] = trajectory_data['rtime']//delta_t
    if rolling:
        t_d_group1 = []
        list_t = np.arange(0,max(trajectory_data['rtime'])-delta_t,t_step)
        list_t = np.array([list_t,list_t+delta_t]).T
        list_x = np.arange(min(trajectory_data['kilopost']),max(trajectory_data['kilopost'])-delta_x,x_step)
        list_x = np.array([list_x,list_x+delta_x]).T
        for ix in list_x:
            print(ix)
            for it in list_t:
                arraylike = trajectory_data[(trajectory_data['rtime']>it[0])&(trajectory_data['rtime']<=it[1])&(trajectory_data['kilopost']>ix[0])&(trajectory_data['kilopost']<ix[1])]
                para = cal_para(arraylike,args=(delta_t,delta_x))
                para.append(ix[1])
                para.append(it[0])
                t_d_group1.append(para)
        macro_para = pd.DataFrame(t_d_group1)
        macro_para.columns = ['q','k','v','v_mu','v_var','x','t']
        macro_para['v(km/h)'] = macro_para['v']*3.6
    else:
        t_d_group = trajectory_data.groupby(['x_label','t_label']).apply(cal_para,args=(delta_t,delta_x))
        macro_para = pd.DataFrame(list(t_d_group))
        macro_para.columns = ['q','k','v','v_mu','v_var']
        macro_para['v(km/h)'] = macro_para['v']*3.6
        st_loc = np.array(tuple(t_d_group.index))
        macro_para['x'] = st_loc[:,0]*delta_x
        macro_para['t'] = st_loc[:,1]*delta_t
    while plot_figure:
        fig = get_fig((14,8))
        plt.scatter(macro_para['t'],macro_para['x'],c=macro_para['v(km/h)'],cmap='jet_r')       # 速度场
        plt.xlim(min(macro_para['t']),max(macro_para['t']))
        plt.ylim(min(macro_para['x']),max(macro_para['x']))
        spatial_temporal_fig_set(xlabel='Time (s)',ylabel='Kilopost (m)',cblabel='Velocity (km/h)',labelsize=30,ticksize=25,inverse=True,cbar=True)
        f2 = get_fig((14,8))
        plt.scatter(macro_para['k']*1000,macro_para['v(km/h)'],alpha=0.1)
        spatial_temporal_fig_set(xlabel='Density (veh/km)',ylabel='Speed (km/h)',cblabel='Velocity (km/h)',labelsize=30,ticksize=25,inverse=False,cbar=False)
        break
    
    # 找到速度的索引顺序
    speed_idx = macro_para.columns.to_list().index('v(km/h)')
    try:
        tensor = np.array(macro_para).reshape(len(list_x),len(list_t),-1)
        speed_field = tensor[:,:,speed_idx]
        acc_field = cal_acc_field_rev(speed_field,t_step,x_step)
        tensor = np.concatenate([tensor,np.expand_dims(acc_field,axis=2)],axis=2)
        m_p = pd.DataFrame(tensor.reshape(-1,tensor.shape[-1]))
        col = list(macro_para.columns)
        col.append('acc')
        m_p.columns = col
        f3 = get_fig((14,8))
        plt.scatter(m_p['t'],m_p['x'],c=m_p['acc'],cmap='jet_r')        # 加速度场
        plt.xlim(min(macro_para['t']),max(macro_para['t']))
        plt.ylim(min(macro_para['x']),max(macro_para['x']))
        spatial_temporal_fig_set(xlabel='Time (s)',ylabel='Kilopost (m)',cblabel='Acceleration (?)',labelsize=30,ticksize=25,inverse=True,cbar=True)
        f4 = get_fig((14,8))
        plt.scatter(m_p['t'],m_p['x'],c=m_p['q'],cmap='jet_r')        # 流量场
        plt.xlim(min(macro_para['t']),max(macro_para['t']))
        plt.ylim(min(macro_para['x']),max(macro_para['x']))
        spatial_temporal_fig_set(xlabel='Time (s)',ylabel='Kilopost (m)',cblabel='volume (veh/s)',labelsize=30,ticksize=25,inverse=True,cbar=True)
        f5 = get_fig((14,8))
        plt.scatter(m_p['t'],m_p['x'],c=m_p['k']*1000,cmap='jet_r')        # 密度场
        plt.xlim(min(macro_para['t']),max(macro_para['t']))
        plt.ylim(min(macro_para['x']),max(macro_para['x']))
        spatial_temporal_fig_set(xlabel='Time (s)',ylabel='Kilopost (m)',cblabel='density (veh/km)',labelsize=30,ticksize=25,inverse=True,cbar=True)
        f6 = get_fig((14,8))
        plt.scatter(m_p[(m_p['acc']>0)&(m_p['acc']<0.1)]['k']*1000,m_p[(m_p['acc']>0)&(m_p['acc']<0.1)]['v'],alpha=0.1)    # 基本图
    except:
        speed_field = np.full((len(set(macro_para['x'])),len(set(macro_para['t']))), np.nan)
        row = list(set(macro_para['x']))
        row.sort()
        col = list(set(macro_para['t']))
        col.sort()

        for i in np.arange(len(row)):
            print(i)
            for j in np.arange(len(col)):
                try:
                    speed_field[i][j] = macro_para[(macro_para['x']==row[i])&(macro_para['t']==col[j])]['v(km/h)']
                except:
                    continue
        acc_field = cal_acc_field(speed_field)               
        m_p = macro_para
    return m_p

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# os.chdir(r'H:\文档\科研\代码1\动态基本图拟合\IEEE_Transactions\trajectory_dada_preprocess')
# # from C1_DataPreprocess import load_data
# # from C0_plotthing import get_fig,spatial_temporal_fig_set,d3_fig_set
# # 读取轨迹数据
# # file_list = ['L001_F001','L001_F002','L001_F003','L001_F004','L001_F005']
# file_list = ['L001_F001']
# # trajectory_data = load_data(file_list,plot_figure=False,sp_lane=False,inverse=True,cbar=True)
# trajectory_data = pd.read_csv('tra_L001_F001.csv',index_col=0)
# macro_para = cal_speed(trajectory_data,delta_t=1,delta_x=1,plot_figure=True)
# macro_para = cal_speed(trajectory_data,delta_t=50,delta_x=300,t_step=2,x_step=10,plot_figure=True,rolling=True)


# macro_para4 = cal_speed(trajectory_data,delta_t=50,delta_x=300,t_step=2,x_step=3,plot_figure=True,rolling=True)
# mp.to_csv('m_p_a_50-300-2-3.csv')

def extract_mac_traj(macro_para,x,t):
    '''
    给定初始格子(x,t),估计该格子的宏观轨迹(这一组车在时空图里怎么走的)
    '''
    # 计算x_step和t_step
    t_step = macro_para.iloc[1]['t'] - macro_para.iloc[0]['t']
    t_len = len(set(macro_para['t']))
    x_step = macro_para.iloc[t_len]['x'] - macro_para.iloc[0]['x']
    # 迭代估计宏观格子的轨迹
    x_min = macro_para['x'].min()
    t_max = macro_para['t'].max()
    x_list = []
    t_list = []
    data_list = []
    while True:
        x_list.append(x)
        t_list.append(t)
        data_list.append(list(macro_para[(macro_para['x']==x)&(macro_para['t']==t)].values[0]))
        speed = macro_para[(macro_para['x']==x)&(macro_para['t']==t)]['v']
        b = int((speed*t_step)/x_step)
        x -= b * x_step
        t += t_step
        if x < x_min:
            break
        elif t > t_max:
            break
    data_list = pd.DataFrame(data_list)
    data_list.columns = macro_para.columns
    return x_list,t_list,data_list

