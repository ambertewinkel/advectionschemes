"""
This code processes the data from data_plots_afespposter.txt to produce plots for the poster for the AFESP conference on Nov 4, 2024.
Date: 30-10-2024 (using combi and sine data from 29-10, 30-10, 31-10)
Author: Amber te Winkel
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

psi04 = np.array([ 1.17921287e-02,  6.45211452e-02,  1.55466275e-01,  2.78923034e-01,
  4.24741128e-01,  5.79682036e-01,  7.27945934e-01,  8.52805131e-01,
  9.37448961e-01,  9.69933334e-01,  9.70215890e-01,  9.35008842e-01,
  8.50274586e-01,  7.26218181e-01,  5.79024963e-01,  4.24781603e-01,
  2.78537101e-01,  1.52636062e-01,  5.86117601e-02,  8.08223626e-03,
  1.82502312e-03,  3.23270339e-02,  1.66028282e-01,  3.80339065e-01,
  6.25533558e-01,  8.38085993e-01,  9.69280404e-01,  9.99967903e-01,
  1.00000000e+00,  9.72440228e-01,  8.38199646e-01,  6.22004252e-01,
  3.75073066e-01,  1.61734002e-01,  3.05112110e-02, -1.73472348e-18,
 -1.73472348e-18, -1.72761186e-18, -1.95156391e-18, -1.89735380e-18])
psi08 = [ 1.11325070e-02,  6.19955421e-02,  1.52287637e-01,  2.75988399e-01,
  4.22483388e-01,  5.77775684e-01,  7.26168828e-01,  8.51684783e-01,
  9.38784587e-01,  9.77433974e-01,  9.76339541e-01,  9.42770042e-01,
  8.55384745e-01,  7.28410505e-01,  5.78740831e-01,  4.23112041e-01,
  2.76530514e-01,  1.51840850e-01,  5.95466191e-02,  8.82298620e-03,
  9.97420178e-04,  1.66814851e-02,  1.33675089e-01,  3.57981422e-01,
  6.32504587e-01,  8.69234616e-01,  9.91691379e-01,  1.00000000e+00,
  1.00000000e+00,  9.84691955e-01,  8.66776939e-01,  6.41982761e-01,
  3.67486166e-01,  1.30766307e-01,  8.29587253e-03, -1.73472348e-18,
 -1.66009434e-18, -1.77566214e-18, -1.95156391e-18, -1.95156391e-18]
psi12 = np.array([0.25004165, 0.29393354, 0.34747756, 0.40689261, 0.46787355, 0.52590374,
 0.57661661, 0.61617245, 0.64161135, 0.65114589, 0.64436588, 0.62233627,
 0.58757511, 0.54389935, 0.49612769, 0.44963902, 0.40980673, 0.38136006,
 0.36775413, 0.37064853, 0.38959249, 0.42199083, 0.46338249, 0.50800971,
 0.54959991, 0.58223602, 0.60116747, 0.60342238, 0.58812306, 0.55647196,
 0.51144516, 0.45728568, 0.39891365, 0.34136227, 0.2893151 , 0.24677578,
 0.21686251, 0.20169634, 0.20234791, 0.21881757])
psi16 = [0.41905437, 0.42570114, 0.43283073, 0.44014371, 0.44735943, 0.4542329,
 0.46056755, 0.46622288, 0.47111639, 0.47522011, 0.47855212, 0.48116435,
 0.48312804, 0.48451863, 0.48540166, 0.48582135, 0.48579314, 0.48530079,
 0.4842986 , 0.48271834, 0.4804801 , 0.47750601, 0.47373516, 0.46913819,
 0.46372989, 0.45757831, 0.45080937, 0.44360611, 0.43620255, 0.42887237,
 0.42191332, 0.41562851, 0.41030616, 0.40619961, 0.40350917, 0.40236752,
 0.40282992, 0.40487013, 0.40838231, 0.41318904]
psi20 =[0.44337286, 0.44503388, 0.4468126,  0.44866613, 0.45055007, 0.45241953,
 0.45423007, 0.4559386 , 0.45750439, 0.45888982, 0.46006134, 0.46099013,
 0.46165288, 0.46203238, 0.46211803, 0.46190623, 0.46140062, 0.46061212,
 0.45955884, 0.45826575, 0.45676416, 0.455091  , 0.4532879 , 0.45140016,
 0.44947552, 0.44756285, 0.44571084, 0.4439666 , 0.44237441, 0.44097442,
 0.43980164, 0.438885  , 0.43824662, 0.43790134, 0.43785638, 0.43811139,
 0.43865848, 0.43948267, 0.44056237, 0.44187001]
psi24 = np.array([0.44955861, 0.44995286, 0.45034828, 0.45073518, 0.45110401, 0.45144569,
 0.45175181, 0.45201479, 0.45222815, 0.45238662, 0.45248629, 0.45252471,
 0.45250091, 0.45241551, 0.45227059, 0.45206976, 0.45181795, 0.45152139,
 0.45118738, 0.45082417, 0.45044071, 0.45004644, 0.44965105, 0.44926427,
 0.44889561, 0.44855412, 0.44824823, 0.44798546, 0.44777229, 0.44761396,
 0.44751438, 0.447476  , 0.44749975, 0.44758505, 0.4477298 , 0.44793043,
 0.44818202, 0.44847837, 0.44881218, 0.44917524])
analytic = np.array([0.00615583, 0.05449674, 0.14644661, 0.27300475, 0.42178277, 0.57821723,
 0.72699525, 0.85355339, 0.94550326, 0.99384417, 0.99384417, 0.94550326,
 0.85355339, 0.72699525, 0.57821723, 0.42178277, 0.27300475, 0.14644661,
 0.05449674, 0.00615583, 0.        , 0.        , 0.        , 0.,
 1.        , 1.        , 1.        , 1.        , 1.        , 1.,
 1.        , 1.        , 0.        , 0.        , 0.        , 0.,
 0.        , 0.        , 0.        , 0.        ])

rmse04 = np.array([0.00010997, 0.00367589, 0.00048265])
rmse08 = np.array([0.00011472, 0.        , 0.00044208])
rmse12 = np.array([0.00170595, 0.00673598, 0.00340097])
rmse16 = np.array([0.00340269, 0.        , 0.00674938])
rmse20 = np.array([0.00705401, 0.        , 0.01387865])
rmse24 = np.array([0.02331069, 0.08205495, 0.04496879])

x = np.linspace(0.,1.,40, endpoint=False)

#clrs = ['#762a83', '#af8dc3', '#e7d4e8', '#f7f7f7', '#d9f0d3', '#7fbf7b', '#1b7837']
clrs = ['#810f7c', '#8856a7', '#8c96c6']

fig1, ax1 = plt.subplots(figsize=(7,3))
box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.18,
                 box.width, box.height * 0.95])
ax1.plot(x, analytic, label='Analytic', color='k', marker='', linestyle='-')
ax1.plot(x, psi04, label='$C$ = 0.4', color=clrs[0], marker='', linestyle='-')
ax1.plot(x, psi08, label='$C$ = 0.8', color=clrs[1], marker='', linestyle='-')
ax1.plot(x, psi12, label='$C$ = 1.2', color=clrs[2], marker='', linestyle='-')
ax1.plot(x, psi16, label='$C$ = 1.6', color=clrs[2], marker='', linestyle='--')
ax1.plot(x, psi20, label='$C$ = 2.0', color=clrs[1], marker='', linestyle='--')
ax1.plot(x, psi24, label='$C$ = 2.4', color=clrs[0], marker='', linestyle='--')
ax1.set_xlabel('x')
ax1.set_ylabel('$\\Psi$')
ax1.set_xlim(0., 1.)
ax1.set_ylim(-0.1, 1.1)
#ax1.legend()
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),
          ncol=7, fancybox=False, shadow=False, columnspacing=0.2)
#plt.savefig('./field.jpg')
plt.savefig('./field.png', format="png", dpi=1200)



#plt.plot(xc, locals()[f'psi_{s}_reg'][nt], **plot_args[c])
#    ut.design_figure(plotname, f'$\\Psi$ at t={nt*dt}', \
#                     'x', '$\\Psi$', 0., xmax, True, -0.1, 1.1)

dx = [0.0125, 0.05, 0.025]
factor = 2
gridscale = np.logspace(0, np.log10(2*factor), num=10)
gridsizes = dx[0]*gridscale
firstorder = rmse04[0]*gridscale
secondorder = rmse04[0]*gridscale*gridscale
thirdorder = rmse04[0]*gridscale*gridscale*gridscale

fig2, ax2 = plt.subplots(figsize=(6,3))
box = ax2.get_position()
ax2.set_position([box.x0, box.y0 + box.height * 0.18,
                 box.width, box.height * 0.95])
ax2.scatter(dx, rmse04, label='$C$ = 0.4', color=clrs[0], marker='x')
ax2.scatter(dx, rmse08, label='$C$ = 0.8', color=clrs[1], marker='x')
ax2.scatter(dx, rmse12, label='$C$ = 1.2', color=clrs[2], marker='x')
ax2.scatter(dx, rmse16, label='$C$ = 1.6', color=clrs[2], marker='+')
ax2.scatter(dx, rmse20, label='$C$ = 2.0', color=clrs[1], marker='+')
ax2.scatter(dx, rmse24, label='$C$ = 2.4', color=clrs[0], marker='+')
ax2.plot(gridsizes, firstorder, color='black', linestyle=':')
ax2.plot(gridsizes, secondorder, color='black', linestyle=':')
ax2.plot(gridsizes, thirdorder, color='black', linestyle=':')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('dx')
ax2.set_ylabel('RMSE')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),
          ncol=6, fancybox=False, shadow=False, columnspacing=0.2)
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#          fancybox=True, shadow=True, ncol=5)
#plt.tight_layout()
plt.savefig('./rmse.png', format="png", dpi=1200)
#plt.savefig("myImage.png", format="png", dpi=resolution_value)
print(f"Default Matplotlib DPI: {plt.rcParams['figure.dpi']}")

