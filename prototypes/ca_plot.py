# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt

def ca_plot(arr,title):
    step_value=0.0005
    start_value_perc=0.98 #98% of values will be ignored for differentiation
    #arr_flat=np.ndarray.flatten(arr[arr>0]);  #flat MOPM array for >0
    arr_flat = np.array(arr[arr>0])
    #3 Compute log-transformed values
    log_arr=np.log10(arr_flat);
    log_arr.sort();
    
    #iterate log_MOPM values, compute areas for them
    quantity_log_arr=np.arange(np.min(log_arr),np.max(log_arr)+step_value,step_value);
    start_value=int(start_value_perc*len(quantity_log_arr)); #start value for differentiation
    
    area_log_array=np.array([]);
    for i in quantity_log_arr:
        ind=np.where(log_arr>=i);
        area_log_array=np.append(area_log_array,np.log10(len(ind[0])));
            
    
    
    #compute derivatives d_logArea/d_logMOPM
    #first derivative
    d_logArea=(area_log_array[1:]-area_log_array[0:-1])/step_value;
    
    #second derivative
    d2_logArea=(d_logArea[1:]-d_logArea[0:-1])/step_value;
    
    #find peaks for 2nd derivatives
    minima = np.array(argrelextrema(d2_logArea[start_value:], np.less));
    maxima = np.array(argrelextrema(d2_logArea[start_value:], np.greater));
    
    #extrema filtration by threshold values
    min_ind=np.abs(d_logArea[start_value:][minima])>1
    minima_filtered=minima[min_ind];
    max_ind=np.abs(d_logArea[start_value:][maxima])>1
    maxima_filtered=maxima[max_ind];
    
    #MOPM class boundaries
    class_boundary_ind=np.append(minima_filtered,maxima_filtered);
    class_boundary_ind.sort();
    
    #4 draw logarithmic C-A plot
    plt.plot(quantity_log_arr[start_value:],area_log_array[start_value:]);
    plt.title(f'{title} C-A plot for values indexes');
    plt.xlabel('Log transform predictor values');
    plt.ylabel('Log(Area)');
    plt.plot(quantity_log_arr[start_value:][class_boundary_ind],area_log_array[start_value:][class_boundary_ind], "xr");
    
    for x,y in zip(quantity_log_arr[start_value:][class_boundary_ind],area_log_array[start_value:][class_boundary_ind]):
        plt.text(x,y, np.round(10**x,4));
    
    
    #plt.plot(quantity_log_arr[start_value:][class_boundary_ind],area_log_array[start_value:][class_boundary_ind], "xr");
    for ind in class_boundary_ind:
        x=[quantity_log_arr[start_value:][ind],quantity_log_arr[start_value:][ind]];
        y=[np.nanmin(area_log_array[start_value:][area_log_array[start_value:] != -np.inf]),area_log_array[start_value:][ind]];
        plt.plot(x,y,'r--');
    
    #print output
    print(len(quantity_log_arr[start_value:][class_boundary_ind]))
    print(quantity_log_arr[start_value:][class_boundary_ind])
    print(10**quantity_log_arr[start_value:][class_boundary_ind])
    
    plt.savefig(f'{title} C-A_plot.png',dpi=300);
    plt.savefig(f'{title} C-A_plot.svg',dpi=300);
    plt.show(); 
    
    
    plt.plot(quantity_log_arr[start_value:-1],d_logArea[start_value:]);
    plt.title(f'{title} first derivative');
    plt.plot(quantity_log_arr[start_value:-2][minima_filtered],d_logArea[start_value:][minima_filtered], "xr");
    plt.plot(quantity_log_arr[start_value:-2][maxima_filtered],d_logArea[start_value:][maxima_filtered], "o");
    plt.show();
    
