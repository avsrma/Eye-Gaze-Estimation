#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:50:22 2019

@author: avneesh
"""

import matplotlib.pyplot as plt
import numpy as np

        
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
colors=['darkorange', 'crimson', 'darkseagreen', 'navy', 'wheat', 'gray', 'palevioletred', 'gold', 'lightcoral', 'forestgreen', 'cornflowerblue']

participants = ['p{:02}'.format(index) for index in range(15)]
print(participants)

#            0     1     2*     3*    4**   5     6     7*      8     9*   10*   11**    12   13     14
lenet_vals=[4.2, 5.0,  6.25, 6.53,  7.93,  6.2,  5.85,  7.1,  6.85,  8.5,  7.05, 6.25, 6.0,  6.8,  6.05]

values = [  4.35, 4.66, 5.55, 5.95, 7.42, 6.023, 5.56, 6.74,  6.17, 6.85, 5.44, 6.39, 5.43, 6.12, 6.21 ]

v_cyclic = [4.25, 4.94, 5.54, 5.47, 6.93, 6.25, 5.88,  6.57,  6.44, 7.13, 5.33, 6.33, 5.7, 6.18, 6.17 ]

v_rlrop = [ 4.70, 4.25, 5.69, 5.30, 7.12, 6.28, 5.46,  6.56,  5.98, 6.55, 5.83, 6.50, 5.30, 6.0, 6.0 ]

v_rlrop1 = [4.53, 4.39, 5.50, 5.33, 7.62, 6.02, 5.36,  6.69,  6.08, 6.84, 5.44, 6.46, 5.51, 6.03, 6.01 ]

v_resnet = [3.98, 5.21, 5.89, 6.00, 7.39, 6.19, 5.56,  6.63,  6.12,  6.95, 6.45, 5.48,  5.62, 6.24, 5.89 ]

v_inc_rlrop=[4.63, 5.42, 6.32, 6.55, 9.22, 10.05, 6.1, 7.46,  7.24,  7.17,  7.28, 5.47, 6.64, 7.49, 5.7]
v_inc_mlr = [5.6, 7.005, 6.35, 6.79, 8.84, 10.05, 6.28, 7.6,  9.83,   7.63, 6.82, 5.48, 7.12, 6.61, 5.79]
v_inc_cyclr=[5.12, 6.96, 6.27, 6.62, 6.54, 7.18,  6.74, 7.37, 7.26,  8.21,  5.97, 5.67, 6.59, 6.68, 5.96]

## issues with 11 major, 2 
### ae 7.4 for p04 is worse than ae for b-32 which was 6.9
print(values,'\n', v_cyclic, '\n', v_rlrop, '\n', v_rlrop1)
x = np.arange(len(participants))  # the label locations

width = 0.35  # the width of the bars
figg, ax = plt.subplots(figsize=(9,5))
fig = ax.bar(participants, v_inc_cyclr, color=colors)
ax.set_ylabel("Angle Error (deg)")
ax.set_ylim([0,9])
ax.set_xlabel("Subject Ids")
ax.set_title(" Inception v3 \n Using CyclicLR \n Mean Angle Error = {}".format(np.mean(v_rlrop1)))
ax.set_xticks(x)
ax.set_xticklabels(participants)

figg.tight_layout()
#figg.subplots_adjust(wspace=20)
autolabel(fig)
plt.show()

#resnet= (v_resnet[0], v_resnet[4], v_resnet[5], v_resnet[14])
#
#figg, ax = plt.subplots(figsize=(9,5))
#fig = ax.bar([0,4,5,14],resnet , color=colors)
#ax.set_ylabel("Angle Error (deg)")
#ax.set_xlabel("Subject Ids")
#ax.set_title("VGG19 \n Using ReduceLROPlateau \n Mean Angle Error = {}".format(np.mean(v_resnet)))
#ax.set_xticks(x)
#ax.set_xticklabels(participants)
#
#plt.show()

import pandas as pd
df = pd.DataFrame()
df['lenet'] = lenet_vals
df['vgg19_rlrop'] = v_rlrop
df['vgg19_cyclic'] = v_cyclic
df['vgg19_mlr'] = values
df['resnet50'] = v_resnet
df['inception_rlrop'] = v_inc_rlrop
df['inception_cycliclr'] = v_inc_cyclr
df['inception_mlr'] = v_inc_mlr
print(df)


