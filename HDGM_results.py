import numpy as np
import pickle
import matplotlib.pyplot as plt

# IND = [10,15,20,35,50,80,100]
IND = [100, 500, 1000, 1500, 2000, 3000]
d_IND = [3, 5, 15, 20, 25, 30]
n_IND = len(IND)
Result_test_power_n = np.zeros([n_IND,6])
Result_test_n_std = np.zeros([n_IND,6])
Result_type_I_n = np.zeros([n_IND,6])
Result_type_I_n_std = np.zeros([n_IND,6])
Result_test_power_d = np.zeros([n_IND,6])
Result_test_d_std = np.zeros([n_IND,6])
Result_type_I_d = np.zeros([n_IND,6])
Result_type_I_d_std = np.zeros([n_IND,6])
J = np.zeros([n_IND,1000])
J0 = np.zeros([n_IND,1000])
J_d = np.zeros([n_IND,1000])
J0_d = np.zeros([n_IND,1000])
k=0
for ii in IND:
    Result_test_power_n[k,:]=(pickle.load(open('Results_HDGM_n'+str(ii)+'_d'+str(30)+'_H1.pckl', 'rb')))[0].mean(1)
    Result_test_n_std[k, :] = (pickle.load(open('Results_HDGM_n'+str(ii)+'_d'+str(30)+'_H1.pckl', 'rb')))[0].std(1)
    Result_type_I_n[k, :] = (pickle.load(open('Results_HDGM_n'+str(ii)+'_d'+str(30)+'_H0.pckl', 'rb')))[0].mean(1)
    Result_type_I_n_std[k, :] = (pickle.load(open('Results_HDGM_n'+str(ii)+'_d'+str(30)+'_H0.pckl', 'rb')))[0].std(1)
    # J[k, :] = (pickle.load(open('Results_HDGM_n'+str(ii)+'_d'+str(30)+'_H1.pckl', 'rb')))[1]
    # J0[k, :] = (pickle.load(open('Results_HDGM_n'+str(ii)+'_d'+str(30)+'_H1.pckl', 'rb')))[2]
    k=k+1

k=0
for ii in d_IND:
    Result_test_power_d[k,:]=(pickle.load(open('Results_HDGM_n'+str(1000)+'_d'+str(ii)+'_H1.pckl', 'rb')))[0].mean(1)
    Result_test_d_std[k, :] = (pickle.load(open('Results_HDGM_n'+str(1000)+'_d'+str(ii)+'_H1.pckl', 'rb')))[0].std(1)
    Result_type_I_d[k, :] = (pickle.load(open('Results_HDGM_n'+str(1000)+'_d'+str(ii)+'_H0.pckl', 'rb')))[0].mean(1)
    Result_type_I_d_std[k, :] = (pickle.load(open('Results_HDGM_n'+str(1000)+'_d'+str(ii)+'_H0.pckl', 'rb')))[0].std(1)
    # J_d[k, :] = (pickle.load(open('Results_HDGM_n'+str(1000)+'_d'+str(ii)+'_H1.pckl', 'rb')))[1]
    # J0_d[k, :] = (pickle.load(open('Results_HDGM_n'+str(1000)+'_d'+str(ii)+'_H1.pckl', 'rb')))[2]
    k=k+1

mycolor = np.array([[224,32,32],
                    [255,192,0],
                    [32,160,64],
                    [48,96,192],
                    [192,48,192]])/255.0
mylinewidth = 4
n_list = 4*np.array(IND)
plt.style.use('ggplot')
fig, axes = plt.subplots(figsize=(8,6.5))
MMD_DK, = axes.plot(n_list,Result_test_power_n[:,0],'-',
                 color=mycolor[0,:],linewidth=mylinewidth,label='MMD-DK')
MMD_OPT, = axes.plot(n_list,Result_test_power_n[:,2],'-',
                 color=mycolor[2,:],linewidth=mylinewidth,label='MMD-OPT')
# MMD, = axes.plot(n_list,Result_test_power[:,3],'--',
#                  color=mycolor[2,:],linewidth=mylinewidth,label='MMD')
ME, = axes.plot(n_list,Result_test_power_n[:,4],'-',
                 color=mycolor[1,:],linewidth=mylinewidth,label='ME')
SCF, = axes.plot(n_list,Result_test_power_n[:,5],'-',
                 color=mycolor[3,:],linewidth=mylinewidth,label='SCF')
axes.legend(handles=[MMD_DK,MMD_OPT,ME,SCF],fontsize=20,loc=0)
axes = plt.gca()
axes.set_xlabel('Number of samples',fontsize=20)
axes.set_ylabel('Average test power',fontsize=20)
axes.set_xlim([400,12000])
axes.set_ylim([-0.05,1.05])
axes.tick_params(axis='both', which='major', labelsize=16)
plt.show()
fig.savefig('Test_power_HDGM_n.pdf',bbox_inches='tight')

plt.style.use('ggplot')
fig, axes = plt.subplots(figsize=(8,6.5))
MMD_DK, = axes.plot(n_list,Result_test_n_std[:,0],'-',
                 color=mycolor[0,:],linewidth=mylinewidth,label='MMD-DK')
MMD_OPT, = axes.plot(n_list,Result_test_n_std[:,2],'-',
                 color=mycolor[2,:],linewidth=mylinewidth,label='MMD-OPT')
# MMD, = axes.plot(n_list,Result_test_std[:,3],'--',
#                  color=mycolor[2,:],linewidth=mylinewidth,label='MMD')
ME, = axes.plot(n_list,Result_test_n_std[:,4],'-',
                 color=mycolor[1,:],linewidth=mylinewidth,label='ME')
SCF, = axes.plot(n_list,Result_test_n_std[:,5],'-',
                 color=mycolor[3,:],linewidth=mylinewidth,label='SCF')
# axes.legend(handles=[MMD_DK,MMD_OPT,MMD,ME,SCF],fontsize=20,loc=4)

axes = plt.gca()
axes.set_xlabel('Number of samples',fontsize=20)
axes.set_ylabel('STD of test power',fontsize=20)
axes.set_xlim([400,12000])
axes.set_ylim([-0.05,0.5])
axes.tick_params(axis='both', which='major', labelsize=16)
plt.show()
fig.savefig('Test_power_std_HDGM_n.pdf',bbox_inches='tight')

plt.style.use('ggplot')
fig, axes = plt.subplots(figsize=(8,6.5))
MMD_DK, = axes.plot(n_list,Result_type_I_n[:,0],'-',
                 color=mycolor[0,:],linewidth=mylinewidth,label='MMD-DK')
MMD_OPT, = axes.plot(n_list,Result_type_I_n[:,2],'-',
                 color=mycolor[2,:],linewidth=mylinewidth,label='MMD-OPT')
# MMD, = axes.plot(n_list,Result_type_I[:,3],'--',
#                  color=mycolor[2,:],linewidth=mylinewidth,label='MMD')
ME, = axes.plot(n_list,Result_type_I_n[:,4],'-',
                 color=mycolor[1,:],linewidth=mylinewidth,label='ME')
SCF, = axes.plot(n_list,Result_type_I_n[:,5],'-',
                 color=mycolor[3,:],linewidth=mylinewidth,label='SCF')
# axes.legend(handles=[MMD_DK,MMD_OPT,MMD,ME,SCF],fontsize=20,loc=4)
axes = plt.gca()
axes.set_xlabel('Number of samples',fontsize=20)
axes.set_ylabel('Average type-I error',fontsize=20)
axes.set_xlim([400,12000])
axes.set_ylim([-0.01,0.1])
axes.tick_params(axis='both', which='major', labelsize=16)
plt.show()
fig.savefig('Type_I_HDGM_n.pdf',bbox_inches='tight')

# plt.style.use('ggplot')
# fig, axes = plt.subplots(figsize=(8,6.5))
# MMD_DK_J, = axes.plot(n_list,-1*J[:,499],'-',
#                  color=mycolor[0,:],linewidth=mylinewidth,label='MMD-DK')
# MMD_OPT_J, = axes.plot(n_list,-1*J0[:,499],'-',
#                  color=mycolor[2,:],linewidth=mylinewidth,label='MMD-OPT')
# axes.legend(handles=[MMD_DK_J,MMD_OPT_J],fontsize=20,loc=4)
# axes = plt.gca()
# axes.set_xlabel('Number of samples at each modal',fontsize=20)
# axes.set_ylabel('Optimized value of objective J',fontsize=20)
# axes.set_xlim([9,101])
# axes.set_ylim([0,0.4])
# axes.tick_params(axis='both', which='major', labelsize=16)
# plt.show()
# fig.savefig('Jvsn_HDGM.pdf',bbox_inches='tight')

n_list = d_IND
plt.style.use('ggplot')
fig, axes = plt.subplots(figsize=(8,6.5))
MMD_DK, = axes.plot(n_list,Result_test_power_d[:,0],'-',
                 color=mycolor[0,:],linewidth=mylinewidth,label='MMD-DK')
MMD_OPT, = axes.plot(n_list,Result_test_power_d[:,2],'-',
                 color=mycolor[2,:],linewidth=mylinewidth,label='MMD-OPT')
# MMD, = axes.plot(n_list,Result_test_power[:,3],'--',
#                  color=mycolor[2,:],linewidth=mylinewidth,label='MMD')
ME, = axes.plot(n_list,Result_test_power_d[:,4],'-',
                 color=mycolor[1,:],linewidth=mylinewidth,label='ME')
SCF, = axes.plot(n_list,Result_test_power_d[:,5],'-',
                 color=mycolor[3,:],linewidth=mylinewidth,label='SCF')
# axes.legend(handles=[MMD_DK,MMD_OPT,ME,SCF],fontsize=20,loc=0)
axes = plt.gca()
axes.set_xlabel('Dimension of samples',fontsize=20)
axes.set_ylabel('Average test power',fontsize=20)
axes.set_xlim([2,31])
axes.set_ylim([-0.05,1.05])
axes.tick_params(axis='both', which='major', labelsize=16)
plt.show()
fig.savefig('Test_power_HDGM_d.pdf',bbox_inches='tight')

plt.style.use('ggplot')
fig, axes = plt.subplots(figsize=(8,6.5))
MMD_DK, = axes.plot(n_list,Result_test_n_std[:,0],'-',
                 color=mycolor[0,:],linewidth=mylinewidth,label='MMD-DK')
MMD_OPT, = axes.plot(n_list,Result_test_n_std[:,2],'-',
                 color=mycolor[2,:],linewidth=mylinewidth,label='MMD-OPT')
# MMD, = axes.plot(n_list,Result_test_std[:,3],'--',
#                  color=mycolor[2,:],linewidth=mylinewidth,label='MMD')
ME, = axes.plot(n_list,Result_test_n_std[:,4],'-',
                 color=mycolor[1,:],linewidth=mylinewidth,label='ME')
SCF, = axes.plot(n_list,Result_test_n_std[:,5],'-',
                 color=mycolor[3,:],linewidth=mylinewidth,label='SCF')
# axes.legend(handles=[MMD_DK,MMD_OPT,MMD,ME,SCF],fontsize=20,loc=4)

axes = plt.gca()
axes.set_xlabel('Dimension of samples',fontsize=20)
axes.set_ylabel('STD of test power',fontsize=20)
axes.set_xlim([2,31])
axes.set_ylim([-0.05,0.5])
axes.tick_params(axis='both', which='major', labelsize=16)
plt.show()
fig.savefig('Test_power_std_HDGM_d.pdf',bbox_inches='tight')

plt.style.use('ggplot')
fig, axes = plt.subplots(figsize=(8,6.5))
MMD_DK, = axes.plot(n_list,Result_type_I_n[:,0],'-',
                 color=mycolor[0,:],linewidth=mylinewidth,label='MMD-DK')
MMD_OPT, = axes.plot(n_list,Result_type_I_n[:,2],'-',
                 color=mycolor[2,:],linewidth=mylinewidth,label='MMD-OPT')
# MMD, = axes.plot(n_list,Result_type_I[:,3],'--',
#                  color=mycolor[2,:],linewidth=mylinewidth,label='MMD')
ME, = axes.plot(n_list,Result_type_I_n[:,4],'-',
                 color=mycolor[1,:],linewidth=mylinewidth,label='ME')
SCF, = axes.plot(n_list,Result_type_I_n[:,5],'-',
                 color=mycolor[3,:],linewidth=mylinewidth,label='SCF')
# axes.legend(handles=[MMD_DK,MMD_OPT,MMD,ME,SCF],fontsize=20,loc=4)
axes = plt.gca()
axes.set_xlabel('Dimension of samples',fontsize=20)
axes.set_ylabel('Average type-I error',fontsize=20)
axes.set_xlim([2,31])
axes.set_ylim([-0.01,0.1])
axes.tick_params(axis='both', which='major', labelsize=16)
plt.show()
fig.savefig('Type_I_HDGM_d.pdf',bbox_inches='tight')

# plt.style.use('ggplot')
# fig, axes = plt.subplots(figsize=(8,6.5))
# MMD_DK_J, = axes.plot(n_list,-1*J[:,499],'-',
#                  color=mycolor[0,:],linewidth=mylinewidth,label='MMD-DK')
# MMD_OPT_J, = axes.plot(n_list,-1*J0[:,499],'-',
#                  color=mycolor[2,:],linewidth=mylinewidth,label='MMD-OPT')
# axes.legend(handles=[MMD_DK_J,MMD_OPT_J],fontsize=20,loc=4)
# axes = plt.gca()
# axes.set_xlabel('Number of samples at each modal',fontsize=20)
# axes.set_ylabel('Optimized value of objective J',fontsize=20)
# axes.set_xlim([9,101])
# axes.set_ylim([0,0.4])
# axes.tick_params(axis='both', which='major', labelsize=16)
# plt.show()
# fig.savefig('Jvsn_HDGM_d.pdf',bbox_inches='tight')