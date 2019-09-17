
import pickle
import matplotlib.pyplot as plt

plt.style.use('ggplot')
fig, axes = plt.subplots(figsize=(8,6.5))
data = pickle.load(open('./Contour_motivition_fig_MMD.pckl','rb'))
X = data[0]
Y = data[1]
ZZ = data[2]
s1 = data[3]
s2 = data[4]
plt.plot(s2[:, 0], s2[:, 1], 'r.')
plt.plot(s1[:, 0], s1[:, 1], 'b.')
for i in range(9):
    contour = plt.contour(X, Y, ZZ[i], [0.7,0.8,0.9], colors='k')
axes = plt.gca()
axes.set_xlabel('Sample values at 1st dimension',fontsize=22)
# axes.set_ylabel('Sample values at 2nd dimension',fontsize=16)
axes.set_xlim([-0.75,2.75])
axes.set_ylim([-0.5,2.5])
axes.tick_params(axis='both', which='major', labelsize=16)
plt.show()
fig.savefig('Motivition_MMD.pdf',bbox_inches='tight')

plt.style.use('ggplot')
fig, axes = plt.subplots(figsize=(8,6.5))
data = pickle.load(open('./Contour_motivition_fig_DK2ST.pckl','rb'))
# data = pickle.load(open('./Contour_results50_3.pckl','rb'))
X = data[0]
Y = data[1]
ZZ = data[2]
s1 = data[3]
s2 = data[4]
plt.plot(s2[:, 0], s2[:, 1], 'r.')
plt.plot(s1[:, 0], s1[:, 1], 'b.')
for i in range(9):
    contour = plt.contour(X, Y, ZZ[i], [0.7,0.8,0.9], colors='k')
axes = plt.gca()
axes.set_xlabel('Sample values at 1st dimension',fontsize=22)
# axes.set_ylabel('Sample values at 2nd dimension',fontsize=16)
axes.set_xlim([-0.75,2.75])
axes.set_ylim([-0.5,2.5])
axes.tick_params(axis='both', which='major', labelsize=16)
plt.show()
fig.savefig('Motivition_DK2ST.pdf',bbox_inches='tight')

plt.style.use('ggplot')
fig, axes_list = plt.subplots(1,2,figsize=(16,6.5))
axes_list[0].plot(s1[:, 0], s1[:, 1], 'b.')
axes_list[0].set_xlabel('Sample values at 1st dimension',fontsize=22)
axes_list[0].set_ylabel('Sample values at 2nd dimension',fontsize=22)
axes_list[0].tick_params(axis='both', which='major', labelsize=16)
axes_list[1].plot(s2[:, 0], s2[:, 1], 'r.')
axes_list[1].set_xlabel('Sample values at 1st dimension',fontsize=22)
axes_list[1].tick_params(axis='both', which='major', labelsize=16)
# axes_list[1].set_ylabel('Sample values at 2nd dimension',fontsize=12)
plt.show()
fig.savefig('Motivition_TwoSamples.pdf',bbox_inches='tight')
