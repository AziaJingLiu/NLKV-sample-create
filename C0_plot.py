# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 21:20:35 2023

@author: hp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
plt.rc('font',family='Times New Roman')
plt.rcParams['font.size'] = 30


data = np.random.rand(10, 10)
# 绘制现有色条
plt.figure()
cax = plt.imshow(data, cmap='jet')
colorbar = plt.colorbar(cax)

# 获取现有色条的颜色映射对象
cmap = colorbar.mappable.get_cmap()

# 选择色条的一部分
vmin = -0.4  # 色条起始位置的值
vmax = 1.4  # 色条结束位置的值
norm = plt.Normalize(vmin=vmin, vmax=vmax)
new_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', cmap(norm(np.linspace(0, 1, 256))))

# 创建自定义的色条对象
custom_cmap = mcolors.ListedColormap(new_cmap(np.arange(new_cmap.N)))

# # 使用自定义的色条对象绘制图像
# plt.figure()
# plt.imshow(data, cmap=custom_cmap)
# plt.colorbar()

# plt.show()


def extract_bi_data(data,r=0,shuffle=True):
    '''
    将EKaV的加速度离散化,得到二分类数据
    r属于[0,1],r=0 -->全部稳态(都用),r=1 -->全部非稳态(都不用)
    '''
    # 确定r对应的稳态范围
    acc_max = data['acc'].max()
    acc_min = data['acc'].min()
    s_up = acc_max * r
    s_dw = acc_min * r
    # 由稳态范围划分加减速数据
    acc_data = data[data['acc']>s_up][['k','ka','v(km/h)','acc']]
    dacc_data = data[data['acc']<s_dw][['k','ka','v(km/h)','acc']]
    # 加标签
    acc_data['label']=np.zeros(len(acc_data))
    dacc_data['label']=np.ones(len(dacc_data))
    # 合并
    bi_data = pd.concat([acc_data,dacc_data])
    # 打乱数据
    if shuffle:
        bi_data = bi_data.sample(frac=1.0,random_state=42)
    return bi_data


def get_fig(fs=(12,8)):
    '''生成指定大小的画布'''
    plt.rcParams['font.size'] = 30
    f1=plt.figure(figsize=fs)
    return f1
# get_fig()

def spatial_temporal_fig_set(xlabel,ylabel,cblabel,labelsize=40,ticksize=35,inverse=True,cbar=True):
    '''时空图,坐标及colorbar设置'''
    ax = plt.gca()
    while inverse:
        ax.invert_yaxis()
        break
    while cbar:
        cb = plt.colorbar(pad=0.02)
        ax1 = cb.ax
        plt.rcParams['font.size'] = ticksize-5
        ax1.set_ylabel(cblabel,size=labelsize-5)
        break
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel(xlabel,fontsize=labelsize)
    plt.ylabel(ylabel,fontsize=labelsize)
    # plt.tight_layout(False)
    
def d3_fig_set(xlabel,ylabel,zlabel,cblabel,f,ax,labelsize=30,ticksize=25,inverse=True,cbar=True):
    '''
    设置3d图的坐标及colorbar

    Parameters
    ----------
    xlabel : str
        x轴名.
    ylabel : str
        y轴名.
    zlabel : str
        z轴名.
    cblabel : str
        colorbar名.
    f : mplot3d balabala
        要用于建立colorbar的数据
        e.g. f = ax.scatter3D(z_dim,t_d['timedelta'],t_d['kilopost'],c=t_d['velocity'],cmap='jet_r').
    ax : mplot3d balabala
        上一行中的ax.
    labelsize : TYPE, optional
        DESCRIPTION. The default is 30.
    ticksize : TYPE, optional
        DESCRIPTION. The default is 25.
    inverse : TYPE, optional
        DESCRIPTION. The default is True.
    cbar : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    '''
    ax1 = plt.gca()
    while inverse:
        ax1.invert_zaxis()
        break
    ax1.set_xlabel(xlabel,fontsize=labelsize)
    ax1.set_ylabel(ylabel,fontsize=labelsize)
    ax1.set_zlabel(zlabel,fontsize=labelsize)
    ax1.set_xticklabels(ax.get_xticklabels(),fontsize=ticksize)
    ax1.set_yticklabels(ax.get_yticklabels(),fontsize=ticksize)
    ax1.set_zticklabels(ax.get_zticklabels(),fontsize=ticksize)
    while cbar:
        cb = plt.colorbar(f,ax=ax,pad=0.02)
        ax2 = cb.ax
        plt.rcParams['font.size'] = 40
        ax2.set_ylabel(cblabel,size=45)
        break

def draw_fields(macro_para,xup=3500,tdw=0,tup=3500,xdw=0,subplot=False,kv_FD=False,kq_FD=False,speed_f=False,volume_f=False,density_f=False,
                acc_f=False,density_a_kv=False,density_a_kq=False,invs=False):
    m_p = macro_para[(macro_para['x']>xdw)&(macro_para['x']<xup)&(macro_para['t']<tup)&(macro_para['t']>=tdw)]
    m_p['x'] = m_p['x']/100
    
    if subplot:
        fig = get_fig((16,12))
        plt.subplot(2,2,1)
        plt.scatter(m_p['k']*1000,m_p['v(km/h)'],c=m_p['acc'],alpha=0.1,cmap='jet_r')
        spatial_temporal_fig_set(xlabel='Density (veh/km)',ylabel='Speed (km/h)',cblabel='Acceleration (m/s^2)',labelsize=30,ticksize=25,inverse=False,cbar=True)
    else:
        # 流量密度基本图
        # if kq_FD:
        #     print('Drawing kq_FD...')
        #     f2 = get_fig((14,8))
        #     plt.scatter(m_p['k']*1000,m_p['q']*3600,c=m_p['acc'],alpha=0.1,cmap='jet_r')
        #     spatial_temporal_fig_set(xlabel='Density (veh/km)',ylabel='volume (veh/h)',cblabel='Acceleration (m/s^2)',labelsize=35,ticksize=30,inverse=False,cbar=True)
        #     print('Done!')
        # 速度场
        if speed_f:
            # print('Drawing speed_field...')
            f3 = get_fig((8,5))
            plt.scatter(m_p['t'],m_p['x'],c=m_p['v(km/h)'],cmap='jet_r')
            plt.xlim(min(m_p['t']),max(m_p['t']))
            plt.ylim(min(m_p['x']),max(m_p['x']))
            spatial_temporal_fig_set(xlabel='Time (s)',ylabel='Kilopost (100 m)',cblabel='Speed (km/h)',labelsize=30,ticksize=25,inverse=invs,cbar=True)
            # print('Done!')
        # 流量场
        # if volume_f:
        #     print('Drawing volume_field...')
        #     f4 = get_fig((14,8))
        #     plt.scatter(m_p['t'],m_p['x'],c=m_p['q']*3600,cmap='jet')
        #     plt.xlim(min(m_p['t']),max(m_p['t']))
        #     plt.ylim(min(m_p['x']),max(m_p['x']))
        #     spatial_temporal_fig_set(xlabel='Time (s)',ylabel='Kilopost (100 m)',cblabel='Volume (veh/h)',labelsize=50,ticksize=40,inverse=invs,cbar=True)
        #     print('Done!')
        # 密度场
        if density_f:
            # print('Drawing density_field...')
            f5 = get_fig((8,5))
            plt.scatter(m_p['t'],m_p['x'],c=m_p['k']*1000,cmap='jet')
            plt.xlim(min(m_p['t']),max(m_p['t']))
            plt.ylim(min(m_p['x']),max(m_p['x']))
            spatial_temporal_fig_set(xlabel='Time (s)',ylabel='Kilopost (100 m)',cblabel='Density (veh/km)',labelsize=30,ticksize=25,inverse=invs,cbar=True)
            # print('Done!')
        # 加速度场
        if acc_f:
            # print('Drawing acc_field...')
            f6 = get_fig((8,5))
            plt.scatter(m_p['t'],m_p['x'],c=m_p['acc'],cmap='jet_r')       # 速度场
            plt.xlim(min(m_p['t']),max(m_p['t']))
            plt.ylim(min(m_p['x']),max(m_p['x']))
            spatial_temporal_fig_set(xlabel='Time (s)',ylabel='Kilopost (100 m)',cblabel='Acceleration (m/s^2)',labelsize=30,ticksize=25,inverse=invs,cbar=True)
            # print('Done!')
        # 速度密度基本图
        if kv_FD:
            # print('Drawing kv_FD...')
            f1 = get_fig((8,5))
            plt.scatter(m_p['k']*1000,m_p['v(km/h)'],c=m_p['acc'],alpha=0.1,cmap='jet_r')
            spatial_temporal_fig_set(xlabel='Density (veh/km)',ylabel='Speed (km/h)',cblabel='Acceleration (m/s^2)',labelsize=30,ticksize=25,inverse=False,cbar=True)
            # print('Done!')
        # 密度偏移后的kv FD
        if density_a_kv:
            # print('Drawing KaV-FD...')
            f7 = get_fig((8,5))
            plt.scatter(m_p['ka']*1000,m_p['v(km/h)'],c=m_p['acc'],alpha=0.1,cmap='jet_r')
            spatial_temporal_fig_set(xlabel='Anticipated density (veh/km)',ylabel='Speed (km/h)',cblabel='Acceleration (m/s^2)',labelsize=30,ticksize=25,inverse=False,cbar=True)
            print('Done!')
        # NLKV
        f10 = get_fig((8,5))
        bi_data = extract_bi_data(m_p,r=0,shuffle=True)
        plt.scatter(100,50,c=custom_cmap(0))
        plt.scatter(100,50,c=custom_cmap(0.9999))
        plt.scatter(bi_data['ka']*1000,bi_data['v(km/h)'],c=bi_data['label'],cmap=custom_cmap)
        plt.legend(['Accelerate sample','Decelerate sample'],fontsize=25)
        spatial_temporal_fig_set(xlabel='Anticipated density (veh/km)',ylabel='Speed (km/h)',cblabel='y',labelsize=30,ticksize=25,inverse=False,cbar=False)
            
            # print('Drawing KaV-FD classify...')
            # f8 = get_fig((14,8))
            # eps_a = 0.6
            # eps_d = -1
            # m_p_ac = m_p[m_p['acc']>eps_a]
            # m_p_dc = m_p[m_p['acc']<eps_d]
            # # m_p_e = m_p[(m_p['acc']<eps_a)&(m_p['acc']>eps_d)]
            # m_p_ee = m_p[(m_p['acc']<0.05)&(m_p['acc']>-0.05)]
            # # plt.scatter(m_p_e['ka']*1000,m_p_e['v(km/h)'])
            # plt.scatter(m_p_ee['ka']*1000,m_p_ee['v(km/h)'])
            # plt.scatter(m_p_ac['ka']*1000,m_p_ac['v(km/h)'])
            # plt.scatter(m_p_dc['ka']*1000,m_p_dc['v(km/h)'])
            
            # plt.legend(['Near-equilibrium data: -0.05 < acc < 0.05','Accelerate data: acc > '+str(eps_a),'Decelerate data: acc < '+str(eps_d)],fontsize=25)
            # spatial_temporal_fig_set(xlabel='Anticipated density (veh/km)',ylabel='Speed (km/h)',cblabel='Acceleration (m/s^2)',labelsize=35,ticksize=30,inverse=False,cbar=False)
            # print('Done!')
        # 密度偏移后的kq FD
        # if density_a_kq:
        #     print('Drawing KaQ-FD...')
        #     f9 = get_fig((14,8))
        #     plt.scatter(m_p['ka']*1000,m_p['q']*3600,c=m_p['acc'],alpha=0.1,cmap='jet_r')
        #     spatial_temporal_fig_set(xlabel='Anticipated density',ylabel='volume (veh/h)',cblabel='Acceleration (m/s^2)',labelsize=35,ticksize=30,inverse=False,cbar=True)
        #     print('Done!')