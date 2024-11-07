"""
This code processes the data from data_plots_afespposter.txt to produce plots for the poster for the AFESP conference on Nov 4, 2024.
Date: 30-10-2024 (using combi and sine data from 29-10, 30-10, 31-10)
New date: 06-11-2024 (retrying the plots, with data from 05-11 and 06-11)
Author: Amber te Winkel

Data used: 
dar--l        29/10/2024     23:28                sine_t0.02_u1.00_LW3aiU-LW3aiU-LW3aiU
dar--l        29/10/2024     23:28                sine_t0.02_u3.00_LW3aiU-LW3aiU-LW3aiU
dar--l        29/10/2024     23:28                sine_t0.02_u6.00_LW3aiU-LW3aiU-LW3aiU
dar--l        29/10/2024     23:30                combi_t1.00_u1.00_LW3aiU-LW3aiU
#####dar--l        29/10/2024     23:30                combi_t1.00_u3.00_LW3aiU-LW3aiU
#####dar--l        29/10/2024     23:31                combi_t1.00_u6.00_LW3aiU-LW3aiU
#####dar--l        31/10/2024     08:46                combi_t1.00_u2.00_LW3aiU
#####dar--l        31/10/2024     08:47                combi_t1.00_u4.00_LW3aiU
#####dar--l        31/10/2024     08:47                combi_t1.00_u5.00_LW3aiU
dar--l        05/11/2024     12:54                sine_t0.0200_u2.0000_LW3aiU
dar--l        05/11/2024     12:55                sine_t0.0200_u4.0000_LW3aiU
dar--l        05/11/2024     12:55                sine_t0.0200_u5.0000_LW3aiU

I need to use u's for which nt is still a integer after 1 revolution.
u=1       nt=100      c=0.4
u=2       nt=50       c=0.8
u=2.5     nt=40       c=1.0
u=3.125   nt=32       c=1.25
u=4       nt=25       c=1.6
u=5       nt=20       c=2.0
u=6.25    nt=16       c=2.5

So actually used:
d-----        06/11/2024     09:57                combi_t0.1600_u6.2500_LW3aiU
d-----        06/11/2024     09:56                combi_t0.2000_u5.0000_LW3aiU
dar--l        06/11/2024     09:56                combi_t0.2500_u4.0000_LW3aiU
dar--l        06/11/2024     09:56                combi_t0.3200_u3.1250_LW3aiU
dar--l        06/11/2024     09:55                combi_t0.4000_u2.5000_LW3aiU
dar--l        06/11/2024     09:55                combi_t0.5000_u2.0000_LW3aiU
dar--l        06/11/2024     09:55                combi_t1.0000_u1.0000_LW3aiU
dar--l        06/11/2024     09:52                sine_t0.0200_u1.0000_LW3aiU
dar--l        06/11/2024     09:53                sine_t0.0200_u2.0000_LW3aiU
dar--l        06/11/2024     09:53                sine_t0.0200_u2.5000_LW3aiU
dar--l        06/11/2024     09:53                sine_t0.0200_u3.1250_LW3aiU
dar--l        06/11/2024     09:53                sine_t0.0200_u4.0000_LW3aiU
dar--l        06/11/2024     09:54                sine_t0.0200_u5.0000_LW3aiU
dar--l        06/11/2024     09:54                sine_t0.0200_u6.2500_LW3aiU
"""




import numpy as np
import matplotlib.pyplot as plt

def design_figure(filename, title, xlabel, ylabel, xlim1, xlim2, bool_ylim = False, ylim1=0.0, ylim2=0.0):
    if bool_ylim == True: plt.ylim(ylim1, ylim2)
    plt.xlim(xlim1, xlim2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()


# fields after 1 revolution
psi = np.array([[ 1.17921287e-02,  6.45211452e-02,  1.55466275e-01,  2.78923034e-01,
  4.24741128e-01,  5.79682036e-01,  7.27945934e-01,  8.52805131e-01,
  9.37448961e-01,  9.69933334e-01,  9.70215890e-01,  9.35008842e-01,
  8.50274586e-01,  7.26218181e-01,  5.79024963e-01,  4.24781603e-01,
  2.78537101e-01,  1.52636062e-01,  5.86117601e-02,  8.08223626e-03,
  1.82502312e-03,  3.23270339e-02,  1.66028282e-01,  3.80339065e-01,
  6.25533558e-01,  8.38085993e-01,  9.69280404e-01,  9.99967903e-01,
  1.00000000e+00,  9.72440228e-01,  8.38199646e-01,  6.22004252e-01,
  3.75073066e-01,  1.61734002e-01,  3.05112110e-02, -1.73472348e-18,
 -1.73472348e-18, -1.72761186e-18, -1.95156391e-18, -1.89735380e-18],
                [ 9.12882535e-03,  5.83795097e-02,  1.49099864e-01,  2.74181023e-01,
  4.21843800e-01,  5.77745281e-01,  7.26811082e-01,  8.53860073e-01,
  9.43181041e-01,  9.83430686e-01,  9.82232751e-01,  9.46636831e-01,
  8.56199782e-01,  7.27343451e-01,  5.77478193e-01,  4.22052708e-01,
  2.74990937e-01,  1.49680076e-01,  5.77650211e-02,  7.70964009e-03,
 -2.05829006e-18,  1.55043828e-03,  8.77378482e-02,  3.32083709e-01,
  6.57026283e-01,  9.21851149e-01,  1.00000000e+00,  1.00000000e+00,
  1.00000000e+00,  9.98515333e-01,  9.12418126e-01,  6.67953585e-01,
  3.42968986e-01,  7.81439695e-02, -3.86656929e-19, -4.33680869e-19,
 -1.88855338e-19, -5.15102512e-19, -8.67361738e-19, -8.67361738e-19],
                [6.15582970e-003, 5.44967379e-002, 1.46446609e-001, 2.73004750e-001,
 4.21782767e-001, 5.78217233e-001, 7.26995250e-001, 8.53553391e-001,
 9.45503262e-001, 9.93844170e-001, 9.93844170e-001, 9.45503262e-001,
 8.53553391e-001, 7.26995250e-001, 5.78217233e-001, 4.21782767e-001,
 2.73004750e-001, 1.46446609e-001, 5.44967379e-002, 6.15582970e-003,
 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
 8.86365947e-126, 1.99591868e-110, 4.49440930e-095, 1.01205100e-079,
 2.27893626e-064, 5.13170824e-049, 1.15555797e-033, 1.73472348e-018],
                [0.12756022, 0.18943763, 0.26958643, 0.36350238, 0.46476387, 0.56556135,
 0.65751152, 0.73259772, 0.78408864, 0.80731478, 0.8002155 , 0.7636033,
 0.70111981, 0.6188534 , 0.5245794 , 0.42665621, 0.33356425, 0.25472807,
 0.20111042, 0.18343639, 0.20821769, 0.27424259, 0.3721099 , 0.48713879,
 0.60309088, 0.70414345, 0.77560729, 0.8056042 , 0.78818411, 0.72587163,
 0.62943516, 0.51462913, 0.39760292, 0.29113605, 0.20299135, 0.13640223,
 0.09180761, 0.06871104, 0.06685303, 0.08642965],
                [0.25411514, 0.3012727,  0.35774031, 0.41947303, 0.48190144, 0.54029356,
 0.59016216, 0.62767262, 0.6500006 , 0.6555971 , 0.64434689, 0.61763501,
 0.57833221, 0.53067416, 0.47997485, 0.43212808, 0.39291901, 0.36724923,
 0.35841786, 0.36758638, 0.3935165 , 0.43263794, 0.47947069, 0.52737349,
 0.569514  , 0.59988382, 0.6141524 , 0.61019011, 0.58818183, 0.55036236,
 0.50049217, 0.44322902, 0.38353522, 0.32620764, 0.27555624, 0.23520721,
 0.20798405, 0.19582149, 0.19968643, 0.21950505],
                [0.3419044,  0.37495846, 0.41153644, 0.44897018, 0.48452271, 0.51561407,
 0.54004321, 0.55619087, 0.56318719, 0.56102359, 0.55058685, 0.53359789,
 0.51244925, 0.48995115, 0.46901057, 0.45228045, 0.44182418, 0.43884449,
 0.44352083, 0.4549855 , 0.47144573, 0.49043298, 0.50913733, 0.52477171,
 0.53490986, 0.53775296, 0.53229818, 0.51840257, 0.49675214, 0.46875649,
 0.43639247, 0.40201886, 0.36817933, 0.33740595, 0.31203214, 0.29402228,
 0.2848252,  0.28525987, 0.29544197, 0.31475969],
                [0.39832602, 0.41940404, 0.44069259, 0.46063301, 0.47813373, 0.49195605,
 0.50083935, 0.5069953 , 0.5069953 , 0.50393002, 0.49819256, 0.49105249,
 0.4837028 , 0.4774536 , 0.4732761 , 0.47179052, 0.47319289, 0.47723944,
 0.48328751, 0.49038383, 0.49738549, 0.50309614, 0.50640064, 0.50638416,
 0.50242474, 0.49425418, 0.48198483, 0.46611267, 0.4474518 , 0.42709382,
 0.40632492, 0.38659172, 0.36906595, 0.35513898, 0.34580454, 0.3417555,
 0.34331809, 0.35041733, 0.36257521, 0.37894213]])

print(psi.shape)
# rmse after nt=4,1,2 for each c (fine - coarse - reg)
rmse_orig = np.array([[0.00010997, 0.00367589, 0.00048265],
                 [9.59074536e-05, 2.46489770e-03, 6.45911332e-04],
                 [1.56230241e-15, 3.61044815e-16, 5.47344339e-16],
                 [0.00222002, 0.00873882, 0.00442224],
                 [0.00678537, 0.02601012, 0.01342036],
                 [0.0140221,  0.05168205, 0.02742764],
                 [0.02593903, 0.09019472, 0.04984593]])
print(rmse_orig.shape)

rmse = np.array([rmse_orig[:,0], rmse_orig[:,2], rmse_orig[:,1]])


# Analytic solution after one revolution (matches for the different u's and c's as they have different nt's)
analytic = np.array([0.00615583, 0.05449674, 0.14644661, 0.27300475, 0.42178277, 0.57821723,
                    0.72699525, 0.85355339, 0.94550326, 0.99384417, 0.99384417, 0.94550326,
                    0.85355339, 0.72699525, 0.57821723, 0.42178277, 0.27300475, 0.14644661,
                    0.05449674, 0.00615583, 0.        , 0.        , 0.        , 0.,
                    1.        , 1.        , 1.        , 1.        , 1.        , 1.,
                    1.        , 1.        , 0.        , 0.        , 0.        , 0.,
                    0.        , 0.        , 0.        , 0.        ])

x = np.linspace(0.,1.,40, endpoint=False)

c_str = ['0.4', '0.8', '1.0', '1.25', '1.6', '2.0', '2.5']
l_str = [':', ':', ':', '--', '--', '--', '-']
#clrs = ['#762a83', '#af8dc3', '#e7d4e8', '#f7f7f7', '#d9f0d3', '#7fbf7b', '#1b7837']
clrs = ['#810f7c', '#8856a7', '#8c96c6', '#810f7c', '#8856a7', '#8c96c6', '#810f7c']
#m_str = ['', '', '', '', '', '', '']

fig1, ax1 = plt.subplots(figsize=(len(c_str)+1,3))
box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.18,
                 box.width, box.height * 0.95])
ax1.plot(x, analytic, label='Analytic', color='k', marker='', linestyle='-')
for r in range(len(c_str)):
    ax1.plot(x, psi[r,:], label=f'$C$ = {c_str[r]}', color=clrs[r], marker='', linestyle=l_str[r])

#ax1.plot(x, psi04, label='$C$ = 0.4', color=clrs[0], marker='', linestyle='-')
#ax1.plot(x, psi08, label='$C$ = 0.8', color=clrs[1], marker='', linestyle='-')
#ax1.plot(x, psi12, label='$C$ = 1.2', color=clrs[2], marker='', linestyle='-')
#ax1.plot(x, psi16, label='$C$ = 1.6', color=clrs[2], marker='', linestyle='--')
#ax1.plot(x, psi20, label='$C$ = 2.0', color=clrs[1], marker='', linestyle='--')
#ax1.plot(x, psi24, label='$C$ = 2.4', color=clrs[0], marker='', linestyle='--')
ax1.set_xlabel('x')
ax1.set_ylabel('$\\Psi$')
ax1.set_xlim(0., 1.)
ax1.set_ylim(-0.1, 1.1)
#ax1.legend()
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),
          ncol=len(c_str)+1, fancybox=False, shadow=False, columnspacing=0.7, handletextpad=0.2, handlelength=1.5)
#plt.savefig('./field.jpg')
plt.savefig('./field.png', format="png", dpi=1200)



#plt.plot(xc, locals()[f'psi_{s}_reg'][nt], **plot_args[c])
#    ut.design_figure(plotname, f'$\\Psi$ at t={nt*dt}', \
#                     'x', '$\\Psi$', 0., xmax, True, -0.1, 1.1)

dx = [0.0125, 0.025, 0.05]
factor = 2
gridscale = np.logspace(0, np.log10(2*factor), num=10)
gridsizes = dx[0]*gridscale
firstorder = rmse[0,0]*gridscale
secondorder = rmse[0,0]*gridscale*gridscale
thirdorder = rmse[0,0]*gridscale*gridscale*gridscale

fig2, ax2 = plt.subplots(figsize=(len(c_str),3))
box = ax2.get_position()
ax2.set_position([box.x0, box.y0 + box.height * 0.18,
                 box.width, box.height * 0.95])
print(np.shape(dx))
print(np.shape(rmse[:,0]))
ax2.plot(gridsizes, firstorder, color='lightgray', linestyle='-')
ax2.plot(gridsizes, secondorder, color='lightgray', linestyle='-')
ax2.plot(gridsizes, thirdorder, color='lightgray', linestyle='-')
for r in range(len(c_str)):
    if c_str[r] == '1.0':
        continue
    ax2.plot(dx, rmse[:,r], label=f'$C$ = {c_str[r]}', color=clrs[r], marker='x', linestyle=l_str[r])
#ax2.scatter(dx, rmse04, label='$C$ = 0.4', color=clrs[0], marker='x')
#ax2.scatter(dx, rmse08, label='$C$ = 0.8', color=clrs[1], marker='x')
#ax2.scatter(dx, rmse12, label='$C$ = 1.2', color=clrs[2], marker='x')
#ax2.scatter(dx, rmse16, label='$C$ = 1.6', color=clrs[2], marker='+')
#ax2.scatter(dx, rmse20, label='$C$ = 2.0', color=clrs[1], marker='+')
#ax2.scatter(dx, rmse24, label='$C$ = 2.4', color=clrs[0], marker='+')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('dx')
ax2.set_ylabel('RMSE')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),
          ncol=len(c_str), fancybox=False, shadow=False, columnspacing=0.7, handletextpad=0.2, handlelength=1.5)
plt.savefig('./rmse.png', format="png", dpi=1200)


