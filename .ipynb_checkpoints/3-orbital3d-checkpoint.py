
##### With two U_b,two J,two V when omega is fixed and U0 is not fixed


import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib import colors
from mpl_toolkits import mplot3d
from multiprocessing import Pool
from tqdm import tqdm

plt.style.use('ggplot')
cols = [p['color'] for p in plt.rcParams['axes.prop_cycle']]
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{braket}\usepackage{nicefrac}')
plt.rcParams.update({'font.size': 30,
                     'figure.figsize': (11,7),
                     'axes.facecolor': 'white',
                     'axes.edgecolor': 'lightgray',
                     "figure.autolayout": 'True',
                     'axes.xmargin': 0.03,
                     'axes.ymargin': 0.05,
                     'axes.grid': False,
                     'axes.linewidth': 5,
                     'lines.markersize': 10,
                     'text.usetex': True,
                     'lines.linewidth': 8,
                     "legend.frameon": True,
                     "legend.framealpha": 0.7,
                     "legend.handletextpad": 1,
                     "legend.edgecolor": "gray",
                     "legend.handlelength": 1,
                     "legend.labelspacing": 0,
                     "legend.columnspacing": 1,
                     "legend.fontsize": 35,
                    })
linestyles = ["-", "--", ":"]





def RG_flow(J0,U0,D0,t,V10,V20,U_b,d):
    J = [J0]
    V1 = [V10]
    V2 = [V20]
    U = [U0]
    D = [D0]
    d_0_int = D[0]-(J[0]/4) - (U_b/4)
    d_0_int1 = d_0_int - (U[0]/2)
    d_0_int2  = d_0_int + (J[0]/4) + (U[0]/2)
    d_0_int3  = d_0_int + (J[0]/4)
    flag_J = True
    flag_V1 = True
    flag_V2 = True
    flag_U = True
    dens = []
    A1 = 0
    A2 = 0
    B1 = 0
    B2 = 0
    C1 = 0
    C2 = 0
    D1 = 0
    D2 = 0
    # print (J0,U_b,"---")
    while D[-1] > 0:
        d_0=D[0]/2 + D[-1]/2-(J[-1]/4)  - (U_b/4)
        
        A1 = 1/(d_0 - t) if (d_0 - t) * (d_0_int - t) > 0 else 0
        A2 = 1/(d_0 + t) if (d_0 + t) * (d_0_int + t) > 0 else 0
         
        B1 = 1/(d_0 - (U[-1]/2) - t) if (d_0 - (U[-1]/2) - t) * (d_0_int1 - t)  > 0 else 0
        B2 = 1/(d_0 - (U[-1]/2) + t) if (d_0 - (U[-1]/2) + t) * (d_0_int1 + t)  > 0 else 0
        
        C1 = 1/(d_0 + (J[-1]/4) + (U[-1]/2) - t ) if (d_0 + (J[-1]/4) + (U[-1]/2) - t) * (d_0_int2 - t) > 0 else 0
        C2 = 1/(d_0 + (J[-1]/4) + (U[-1]/2) + t ) if (d_0 + (J[-1]/4) + (U[-1]/2) + t) * (d_0_int2 + t) > 0 else 0
        
        D1 = 1/(d_0 + (J[-1]/4) - t ) if (d_0 + (J[-1]/4) - t) * (d_0_int3 - t) > 0 else 0
        D2 = 1/(d_0 + (J[-1]/4) + t ) if (d_0 + (J[-1]/4) + t) * (d_0_int3 + t) > 0 else 0
            
        delta_J = J[-1]*(J[-1] + 4*U_b)*(A1 + A2)*0.5*d * (2/(np.pi * D[0]))*np.sqrt(1 - (D[-1]**2/D[0]**2))
        if (J[-1] + delta_J) * J[-1] > 0 and flag_J:
            J.append(J[-1] + delta_J)
        else :
            flag_J =False
            J.append(0)
            
        delta_V_1_1 = ((3*J[-1]*V1[-1])/8)*(A2 + B2)*d * (2/(np.pi * D[0]))*np.sqrt(1 - (D[-1]**2/D[0]**2))
        delta_V_1_2 = ((V1[-1] * U_b)/2)*(C1 + D1 + B2 + A2)*d * (2/(np.pi * D[0]))*np.sqrt(1 - (D[-1]**2/D[0]**2))
        delta_V_1 =  delta_V_1_1 + delta_V_1_2 
        
            
        if (V1[-1] + delta_V_1) * V1[-1] > 0 and flag_V1:
            V1.append(V1[-1] + delta_V_1)
        else :
            V1.append(0)
            flag_V1 = False
        #print(A2,B2,V1[-1])
        delta_V_2_1 = ((3*J[-1]*V2[-1])/8)*(A1 + B1)*d * (2/(np.pi * D[0]))*np.sqrt(1 - (D[-1]**2/D[0]**2))
        delta_V_2_2 = ((V2[-1] * U_b)/2)*(C2 + D2 + B1 + A1)*d * (2/(np.pi * D[0]))*np.sqrt(1 - (D[-1]**2/D[0]**2))
        delta_V_2 =  delta_V_2_1 + delta_V_2_2 
        
        if (V2[-1] + delta_V_2) * V2[-1] > 0 and flag_V2:
            V2.append(V2[-1] + delta_V_2)
        else :
            V2.append(0)
            flag_V2 = False
            
            
        delta_U = (4*(V1[-1]**2 + V2[-1]**2)*(C1 + C2 - B1 - B2)+ J[-1]**2*(A1 + A2))* d * (2/(np.pi * D[0]))*np.sqrt(1 - (D[-1]**2/D[0]**2))
    
        if (U[-1] + delta_U) * U[-1] > 0 and flag_U:
            U.append(U[-1] + delta_U)
        else :
            U.append(0)
            flag_U = False
        D.append(D[-1]-d)
        
    return V1, V2, J, U, D
    
    
    
def RG(J0_s,U_b_s,U0,D0,t_perp_s,V_by_J,d=0.01):
    W = []
    for J0 in tqdm(J0_s) :
        for t in t_perp_s:
            for r_i in U_b_s:
                V10 = J0*V_by_J
                V20 = J0*V_by_J
                U_b = r_i * J0
                #U0 = - 2* U_b
                V1, V2, J, U, D = RG_flow(J0,U0,D0,t,V10,V20,U_b,d)
                if J[-1]/J0 < 1 and V1[-1]/V10 < 1 and V2[-1]/V20 < 1 and U[-1]/U0 > 0.3:
                    flag = 0
                elif J[-1]/J0 < 1 and V1[-1]/V10 < 1 and V2[-1]/V20 < 1 and U[-1]/U0 < 0.3:
                    #print (J0, V10, V20, U0, U_b, J[-1]/J0, V1[-1]/V10, V2[-1]/V20, U[-1]/U0)
                    flag = 1
                elif J[-1]/J0 < 1  and V1[-1]/V10 > 1  and V2[-1]/V20 > 1:
                    flag = 2
                elif J[-1]/J0 < 1  and V1[-1]/V10 < 1 and V2[-1]/V20 > 1:
                    flag = 3
                elif J[-1]/J0 > 1  and V1[-1]/V10 > 1 and V2[-1]/V20 > 1:
                    flag = 4
                elif J[-1]/J0 > 1 and V1[-1]/V10 < 1 and V2[-1]/V20 < 1:
                    flag = 5
                    # print (J[-1]/J0, V[-1]/V0)
                    # print (dens)
                elif J[-1]/J0 > 1  and V1[-1]/V10 < 1 and V2[-1]/V20 > 1:
                    flag = 6
                W.append(flag)
            
    
    return W
     

D0 = 100
U0 = 0.1 * D0
m =   np.linspace(0.0001, 1, 20)
n = np.linspace(0.01, 0.5, 20)
p = np.linspace(0.0001, 1, 8)
J0_s = m * D0
#print(y)
t_perp_s = n * D0
V_by_J = 0.1
U_b_s = -p
RG_Fixed_point = RG(J0_s,U_b_s,U0,D0,t_perp_s,V_by_J,d=0.05)

C = RG_Fixed_point


from itertools import product
from matplotlib import colors
X, Y, Z = [], [], []



for x1,y1,z1 in product(m,n,-U_b_s):
    X.append(x1)
    Y.append(y1)
    Z.append(z1)

# Creating figure
fig = plt.figure(figsize = (16, 9))
ax = plt.axes(projection ="3d")
   
# Add x, y gridlines 
ax.grid(b = True, color ='grey', 
        linestyle ='-.', linewidth = 0.3, 
        alpha = 0.2) 
 
 
# Creating color map
cmap = plt.cm.Set1
norm = colors.BoundaryNorm(np.arange(-0.5, 7, 1), cmap.N)

 
# Creating plot
sctt = ax.scatter3D(X, Y, Z,
                    alpha = 0.8,
                    c = C, 
                    cmap = cmap, 
                    marker ='o',s=200, norm=norm)
#print(x * y * z) 
plt.title("3-orbital 3D plot")
ax.set_xlabel('J', fontweight ='bold') 
ax.set_ylabel('t_p', fontweight ='bold') 
ax.set_zlabel('U_b', fontweight ='bold')
p = fig.colorbar(sctt, ax = ax, ticks=np.linspace(0, 6, 7))
p.set_ticklabels(['Ins_U_R', 'Dead Zone','$V1_R,V2_R$','V$2_R$','$J_{R}$,V$1_{R}$,V$2_{R}$','Coexis_J_R', '$J_{R}$,V$2_{R}$'])
# show plot
plt.show()