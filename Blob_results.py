import numpy as np
import pickle
import matplotlib.pyplot as plt

# IND = [10,15,20,35,50,80,100]
IND = [10,20,30,50,70,80,90,100]
n_IND = len(IND)
Result_test_power = np.zeros([n_IND,6])
Result_test_std = np.zeros([n_IND,6])
Result_type_I = np.zeros([n_IND,6])
Result_type_I_std = np.zeros([n_IND,6])
J = np.zeros([n_IND,500])
J0 = np.zeros([n_IND,500])
k=0
for ii in IND:
    Result_test_power[k,:]=(pickle.load(open('./Results_'+str(ii)+'_H1.pckl', 'rb')))[0].mean(1)
    Result_test_std[k, :] = (pickle.load(open('./Results_' + str(ii) + '_H1.pckl', 'rb')))[0].std(1)
    Result_type_I[k, :] = (pickle.load(open('./Results_' + str(ii) + '_H0.pckl', 'rb')))[0].mean(1)
    Result_type_I_std[k, :] = (pickle.load(open('./Results_' + str(ii) + '_H0.pckl', 'rb')))[0].std(1)
    J[k, :] = (pickle.load(open('./Results_' + str(ii) + '_H1.pckl', 'rb')))[1]
    J0[k, :] = (pickle.load(open('./Results_' + str(ii) + '_H1.pckl', 'rb')))[2]
    k=k+1

mycolor = np.array([[224,32,32],
                    [255,192,0],
                    [32,160,64],
                    [48,96,192],
                    [192,48,192]])/255.0
n_list = IND
mylinewidth = 4
plt.style.use('ggplot')
fig, axes = plt.subplots(figsize=(8,6.5))
MMD_DK, = axes.plot(n_list,Result_test_power[:,0],'-',
                 color=mycolor[0,:],linewidth=mylinewidth,label='MMD-DK')
MMD_OPT, = axes.plot(n_list,Result_test_power[:,2],'-',
                 color=mycolor[2,:],linewidth=mylinewidth,label='MMD-OPT')
# MMD, = axes.plot(n_list,Result_test_power[:,3],'--',
#                  color=mycolor[2,:],linewidth=mylinewidth,label='MMD')
ME, = axes.plot(n_list,Result_test_power[:,4],'-',
                 color=mycolor[1,:],linewidth=mylinewidth,label='ME')
SCF, = axes.plot(n_list,Result_test_power[:,5],'-',
                 color=mycolor[3,:],linewidth=mylinewidth,label='SCF')
axes.legend(handles=[MMD_DK,MMD_OPT,ME,SCF],fontsize=20,loc=4)
axes = plt.gca()
axes.set_xlabel('Number of samples at each modal',fontsize=20)
axes.set_ylabel('Average test power',fontsize=20)
axes.set_xlim([9,101])
axes.set_ylim([-0.05,1.05])
axes.tick_params(axis='both', which='major', labelsize=16)
plt.show()
fig.savefig('Test_power_blob.pdf',bbox_inches='tight')

plt.style.use('ggplot')
fig, axes = plt.subplots(figsize=(8,6.5))
MMD_DK, = axes.plot(n_list,Result_test_std[:,0],'-',
                 color=mycolor[0,:],linewidth=mylinewidth,label='MMD-DK')
MMD_OPT, = axes.plot(n_list,Result_test_std[:,2],'-',
                 color=mycolor[2,:],linewidth=mylinewidth,label='MMD-OPT')
# MMD, = axes.plot(n_list,Result_test_std[:,3],'--',
#                  color=mycolor[2,:],linewidth=mylinewidth,label='MMD')
ME, = axes.plot(n_list,Result_test_std[:,4],'-',
                 color=mycolor[1,:],linewidth=mylinewidth,label='ME')
SCF, = axes.plot(n_list,Result_test_std[:,5],'-',
                 color=mycolor[3,:],linewidth=mylinewidth,label='SCF')
# axes.legend(handles=[MMD_DK,MMD_OPT,MMD,ME,SCF],fontsize=20,loc=4)
axes = plt.gca()
axes.set_xlabel('Number of samples at each modal',fontsize=20)
axes.set_ylabel('STD of test power',fontsize=20)
axes.set_xlim([9,101])
axes.set_ylim([-0.05,0.5])
axes.tick_params(axis='both', which='major', labelsize=16)
plt.show()
fig.savefig('Test_power_std_blob.pdf',bbox_inches='tight')

plt.style.use('ggplot')
fig, axes = plt.subplots(figsize=(8,6.5))
MMD_DK, = axes.plot(n_list,Result_type_I[:,0],'-',
                 color=mycolor[0,:],linewidth=mylinewidth,label='MMD-DK')
MMD_OPT, = axes.plot(n_list,Result_type_I[:,2],'-',
                 color=mycolor[2,:],linewidth=mylinewidth,label='MMD-OPT')
# MMD, = axes.plot(n_list,Result_type_I[:,3],'--',
#                  color=mycolor[2,:],linewidth=mylinewidth,label='MMD')
ME, = axes.plot(n_list,Result_type_I[:,4],'-',
                 color=mycolor[1,:],linewidth=mylinewidth,label='ME')
SCF, = axes.plot(n_list,Result_type_I[:,5],'-',
                 color=mycolor[3,:],linewidth=mylinewidth,label='SCF')
# axes.legend(handles=[MMD_DK,MMD_OPT,MMD,ME,SCF],fontsize=20,loc=4)
axes = plt.gca()
axes.set_xlabel('Number of samples at each modal',fontsize=20)
axes.set_ylabel('Average type-I error',fontsize=20)
axes.set_xlim([9,101])
axes.set_ylim([-0.01,0.1])
axes.tick_params(axis='both', which='major', labelsize=16)
plt.show()
fig.savefig('Type_I_blob.pdf',bbox_inches='tight')

plt.style.use('ggplot')
fig, axes = plt.subplots(figsize=(8,6.5))
MMD_DK_J, = axes.plot(n_list,-1*J[:,499],'-',
                 color=mycolor[0,:],linewidth=mylinewidth,label='MMD-DK')
MMD_OPT_J, = axes.plot(n_list,-1*J0[:,499],'-',
                 color=mycolor[2,:],linewidth=mylinewidth,label='MMD-OPT')
# axes.legend(handles=[MMD_DK_J,MMD_OPT_J],fontsize=20,loc=4)
axes = plt.gca()
axes.set_xlabel('Number of samples at each modal',fontsize=20)
axes.set_ylabel('Optimized value of objective J',fontsize=20)
axes.set_xlim([9,101])
axes.set_ylim([0,0.4])
axes.tick_params(axis='both', which='major', labelsize=16)
plt.show()
fig.savefig('Jvsn.pdf',bbox_inches='tight')