# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 11:19:08 2023

@author: hp
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from C0_plot import get_fig,spatial_temporal_fig_set,d3_fig_set
plt.rc('font',family='Times New Roman')
plt.rcParams['font.size'] = 20

def load_data(file_name,plot_figure=False,sp_lane=False,inverse=True,cbar=True):
    '''加载数据集,支持速度时空图可视化'''
    trajectory_list = []
    col_name = ['veh_id','datetime','veh_type','velocity','traffic_lane','lng','lat','kilopost','veh_len','detected_flag']
    tra_dt = pd.read_csv(file_name,header=None)
    tra_dt.columns = col_name
    tra_dt['datetime'] = pd.to_datetime(tra_dt['datetime'],format='%H%M%S%f')
    tra_dt['rtime'] = pd.to_timedelta(tra_dt['datetime']-tra_dt['datetime'].min()).dt.total_seconds()
    # 计算每辆车每两条记录间的距离差和时间差
    tra_data = pd.DataFrame()
    for veh in set(tra_dt['veh_id']):
        # print(veh)
        tra_veh = tra_dt[tra_dt['veh_id']==veh]
        tra_veh['kilodelta'] = -tra_veh['kilopost'].diff()
        tra_veh['timedelta'] = tra_veh['rtime'].diff()
        tra_data = tra_data.append(tra_veh)
    while plot_figure:
        if sp_lane:
            # 取巧设置,为了使不同的车道共用相同colorbar
            min_speed_lane = tra_data[tra_data['velocity']==min(tra_data['velocity'])].copy()
            max_speed_lane = tra_data[tra_data['velocity']==max(tra_data['velocity'])].copy()
            min_speed_lane['rtime'] = -1
            max_speed_lane['rtime'] = -1
            min_speed_lane['kilopost'] = -1
            max_speed_lane['kilopost'] = -1
            # 画图
            fig = get_fig((14,8))
            lane_num = set(tra_data['traffic_lane'])
            ax = Axes3D(fig)
            for lane in lane_num:
                t_d = tra_data[tra_data['traffic_lane']==lane].copy()
                t_d = t_d.append(min_speed_lane, ignore_index = True)
                t_d = t_d.append(max_speed_lane, ignore_index = True)
                z_dim = np.ones(len(t_d))*lane
                f = ax.scatter3D(z_dim,t_d['rtime'],t_d['kilopost'],c=t_d['velocity'],cmap='jet_r')
            # 坐标轴设置
            ax.set_xlim(min(lane_num),max(lane_num))
            ax.set_ylim(min(tra_data['rtime']),max(tra_data['rtime']))
            ax.set_zlim(min(tra_data['kilopost']),max(tra_data['kilopost']))
            ax.set_xticks(list(lane_num))
            d3_fig_set(xlabel='lane',ylabel='Time (s)',zlabel='Kilopost (m)',cblabel='Velocity (km/h)',f=f,ax=ax,labelsize=30,ticksize=25,inverse=inverse,cbar=cbar)
        else:
            fig = get_fig((24,8))
            plt.scatter(tra_data['rtime'],tra_data['kilopost'],c=tra_data['velocity'],cmap='jet_r')
            plt.xlim(min(tra_data['rtime']),max(tra_data['rtime']))
            plt.ylim(min(tra_data['kilopost']),max(tra_data['kilopost']))
            spatial_temporal_fig_set(xlabel='Time (s)',ylabel='Kilopost (m)',cblabel='Velocity (km/h)',labelsize=30,ticksize=25,inverse=inverse,cbar=cbar)
        break
    trajectory_list.append(tra_data)
    return trajectory_list