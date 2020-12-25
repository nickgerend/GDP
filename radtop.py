# Written by: Nick Gerend, @dataoutsider
# Viz: "GDP", enjoy!

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from math import pi, cos, sin, exp, sqrt, atan2
pd.set_option('display.max_rows', None)

class point:
    def __init__(self, index, item, x, y, path = -1, value = -1, length = -1, width = -1, ratio = -1, fraction = -1, cap = 0): 
        self.index = index
        self.item = item
        self.x = x
        self.y = y
        self.path = path
        self.value = value  
        self.length = length
        self.width = width
        self.ratio = ratio   
        self.fraction = fraction
        self.cap = cap
    def to_dict(self):
        return {
            'index' : self.index,
            'item' : self.item,
            'x' : self.x,
            'y' : self.y,
            'path' : self.path,
            'value' : self.value,
            'length' : self.length,
            'width' : self.width,
            'ratio' : self.ratio,
            'fraction' : self.fraction,
            'cap' : self.cap }

def vertical_sigmoid(i, count, x1, y1, x2, y2):
    dx = abs(x2-x1)
    dy = abs(y2-y1)
    xamin = (1-1.0)*(12.0/(count-1.0))-6.0
    amin = 1.0/(1.0+exp(-xamin))
    xamax = (count-1.0)*(12.0/(count-1.0))-6.0
    amax = 1.0/(1.0+exp(-xamax))
    da = amax-amin

    xi = (i-1.0)*(12.0/(count-1.0))-6.0
    a = ((1.0/(1.0+exp(-xi)))-amin)/da
    x = a * dx + x1
    y = ((i-1.0)*(dy/(count-1.0))-(dy/2.0))+dy/2+y1
    if x2 < x1:
        x -= dx
        y = ((dy/2.0)-y)-(dy/2.0)+y1+y2
    return x, y

def arc(i, count, x1, x2, y, depth):
    # i starts at 0, count = odd integer
    x0 = (x1+x2)/2
    y0 = y-depth
    r = sqrt((x1-x0)**2 + (y-y0)**2)
    angle = 180./pi*atan2(x1-x0,y-y0)
    angle_i = i/count*(2*abs(angle))+angle
    xout = r*sin(angle_i*pi/180)+x0
    yout = r*cos(angle_i*pi/180)+y0
    return xout, yout

def Ellipse_y(x, width, height):
    a = width/2
    b = height
    return (b/a)*sqrt(a**2-x**2)

#region prepare data
df = pd.read_csv(os.path.dirname(__file__) + '/GDP_by_Country.csv')
df = df.sort_values(['2019'], ascending=[False]).head(10)
# print(df.isna().sum())
# print(df.isna().sum().sum())

years = range(1960,2020,1)
valcols = [str(x) for x in years]
dft = pd.melt(df, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], value_vars=valcols)
dft.rename({'variable': 'Year', 'value': 'GDP'}, axis=1, inplace=True)
dft = dft[['Country Name', 'Country Code', 'Year', 'GDP']].dropna()
dft = dft.astype({'Year': int, 'GDP': float})
dft = dft.loc[dft['Year']>=2000]
dft.to_csv(os.path.dirname(__file__) + '/GDP_by_Country_T.csv', encoding='utf-8', index=False)


radbycol = 'Year'
radlengthcol = 'GDP'
radcolorcol = 'Country Name'
dft_group = dft.groupby(radbycol)
width_scale = 6
length_scale = 4 #3
arc_scale = 2.3
value_scale = 1000000000

total_width = 0
for rad_i, group in dft_group:
    length_y = group[radlengthcol].sum()
    group_sort_d = group.sort_values(by=[radlengthcol], ascending=False)  
    total_width += (group_sort_d[radlengthcol].values[0]/length_y*100)**width_scale
#endregion

#region algorithm

list_xy = []
count = 51
radscale = 1
xrad = 0
yrad = 0
j = 0
switch = 0
last_ellipse = []
last_yrad = 0
value = 0
e_height = 0

for rad_i, group in dft_group:
    group_sort_a = group.sort_values(by=[radlengthcol], ascending=True)
    group_sort_d = group.sort_values(by=[radlengthcol], ascending=False)  
    length_y = group[radlengthcol].sum()
    ratio = group_sort_d[radlengthcol].values[0]/length_y*100
    ratio2 = group_sort_d[radlengthcol].values[1]/length_y*100
    width_x = ratio**width_scale
    fraction = width_x/total_width
    half = width_x/2
    xe = np.linspace(-half, half, num=count).tolist()
    e_list = []
    v_list = []
    h_list = []
    i_list = []
    
    for i, row in group_sort_a.iterrows():
        item = row[radcolorcol]
        i_list.append(item)
        value = row[radlengthcol]/value_scale
        e_height = (value/length_scale)*(1+(fraction**arc_scale*100**(arc_scale-1)))
        h_list.append(e_height)
        ye = [Ellipse_y(x,width_x,e_height)+value for x in xe]
        ellipse =  list(zip(xe+half,ye))
        
        if switch == 0:
            list_xy.append(point(rad_i, item, width_x + xrad, 0, 0, value, width_x, length_y, ratio))
            list_xy.append(point(rad_i, item, xrad, 0, 1, value, width_x, length_y, ratio, fraction))
            switch = 1
            j = 2
        else:
            for e in reversed(last_ellipse):
                xout = e[0]+xrad
                yout = e[1]+last_yrad
                list_xy.append(point(rad_i, item, xout, yout, j, value, width_x, length_y, ratio, fraction))
                j += 1
        for e in ellipse:
            xout = e[0]+xrad
            yout = e[1]+yrad
            list_xy.append(point(rad_i, item, xout, yout, j, value, width_x, length_y, ratio, fraction))
            j += 1
        
        j = 0     
        last_yrad = yrad
        v_list.append(last_yrad)
        last_ellipse = ellipse
        e_list.append(ellipse)
        yrad += value

    #region cap
    for e in reversed(ellipse):
        xout = e[0]+xrad
        yout = e[1]+last_yrad
        list_xy.append(point(rad_i, 'cap', xout, yout, j, value, width_x, length_y, ratio, fraction))
        j += 1
    
    k = 0
    height_s = e_height*3 + e_height
    x1 = ellipse[0][0]
    x2 = ellipse[len(ellipse)-1][0]
    y1 = 0
    y2 = height_s
    for e in ellipse:     
        if e[0] <= half:
            x, y = vertical_sigmoid(k+1, len(ellipse), x1, y1, x2, y2)
        else:
            x, y = vertical_sigmoid(k+1, len(ellipse), x2, y1, x1, y2)
        xout = x+xrad
        yout = y+yrad
        list_xy.append(point(rad_i, 'cap', xout, yout, j, value, width_x, length_y, ratio, fraction))
        list_xy.append(point(rad_i, 'back', xout, yout, len(ellipse)-k, value, width_x, length_y, ratio, fraction))
        list_xy.append(point(rad_i, 'back', xout, yout*10, j, value, width_x, length_y, ratio, fraction))
        j += 1
        k += 1
    #endregion

    #region cap2
    e_2 = e_list[group_sort_a.shape[0]-1]
    e_3 = e_list[group_sort_a.shape[0]-2]
    v_2 = v_list[group_sort_a.shape[0]-1]
    v_3 = v_list[group_sort_a.shape[0]-2]
    h_2 = h_list[group_sort_a.shape[0]-2]
    i_2 = i_list[group_sort_a.shape[0]-2]
    for e in reversed(e_3):
        xout = e[0]+xrad
        yout = e[1]+v_3
        list_xy.append(point(rad_i, i_2, xout, yout, j, value, width_x, length_y, ratio2, fraction, 1))
        j += 1
    
    k = 0
    height_s = h_2*3 + h_2
    x1 = e_2[0][0]
    x2 = e_2[len(e_2)-1][0]
    y1 = 0
    y2 = height_s
    for e in e_2:     
        if e[0] <= half:
            x, y = vertical_sigmoid(k+1, len(e_2), x1, y1, x2, y2)
        else:
            x, y = vertical_sigmoid(k+1, len(e_2), x2, y1, x1, y2)
        xout = x+xrad
        yout = y+v_2
        list_xy.append(point(rad_i, i_2, xout, yout, j, value, width_x, length_y, ratio2, fraction, 1))
        j += 1
        k += 1
    #endregion

    j = 0
    yrad = 0
    xrad += width_x
    switch = 0

# x = [o.x for o in list_xy]
# y = [o.y for o in list_xy]
# plt.scatter(x, y)
# plt.show()

#endregion

#region transform and output
df_out = pd.DataFrame.from_records([s.to_dict() for s in list_xy])
y_max = df_out['y'].max()
# check = df_out['fraction'].unique()
# print(check.sum())

offset = 0.0
import csv
import os
N = len(list_xy)
with open(os.path.dirname(__file__) + '/radgdp.csv', 'w',) as csvfile:
    writer = csv.writer(csvfile, lineterminator = '\n')
    writer.writerow(['index', 'item', 'x', 'y', 'path', 'value', 'length', 'width', 'ratio', 'fraction', 'cap'])
    for i, row in df_out.iterrows():
        t = row['x']
        ch1 = row['y']
        angle = (2.*pi)*(row['x']/total_width)
        angle_deg = angle * 180./pi
        angle_rotated = (abs(angle_deg-360.)+90.) % 360.
        angle_new = angle_rotated * pi/180.
        x = (offset+ch1)*cos(angle_new)
        y = (offset+ch1)*sin(angle_new)
        writer.writerow([row['index'], row['item'], x, y, row['path'], row['value'], row['length'], row['width'], row['ratio'], row['fraction'], row['cap']])
#endregion

print('finished')