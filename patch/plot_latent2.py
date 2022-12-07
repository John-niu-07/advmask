import numpy as np
from matplotlib import pyplot as plt


z = np.load('latent_z_ep100.npy')
z_init = np.load('latent_z_init_ep100.npy')
z_c = np.load('latent_z_control_ep100.npy')

z_97 = np.load('latent_z_ep97.npy')
z_98 = np.load('latent_z_ep98.npy')
z_99 = np.load('latent_z_ep99.npy')
z_101 = np.load('latent_z_ep101.npy')
z_102 = np.load('latent_z_ep102.npy')
z_103 = np.load('latent_z_ep103.npy')
z_104 = np.load('latent_z_ep104.npy')
z_105 = np.load('latent_z_ep105.npy')
z_106 = np.load('latent_z_ep106.npy')


z_0 = np.load('latent_z_ep0.npy')
z_1 = np.load('latent_z_ep1.npy')
z_2 = np.load('latent_z_ep2.npy')
z_3 = np.load('latent_z_ep3.npy')
'''
z[np.abs(z)<1.0]=0
z_init[np.abs(z_init)<1.0]=0
z_c[np.abs(z_c)<1.0]=0
'''


z_diff0 = z_98 - z_97
z_diff1 = z_99 - z_98
z_diff2 = z - z_99

z_diff3 = z_101 - z
z_diff4 = z_102 - z_101
z_diff5 = z_103 - z_102
z_diff6 = z_104 - z_103
z_diff7 = z_105 - z_104
z_diff8 = z_106 - z_105


z_diff_a = z_1 - z_0
z_diff_b = z_2 - z_1
z_diff_c = z_3 - z_2


idx3 = np.where(np.abs(z_diff3)>0.025)[1]
idx4 = np.where(np.abs(z_diff4)>0.04)[1]
idx5 = np.where(np.abs(z_diff5)>0.04)[1]
idx6 = np.where(np.abs(z_diff6)>0.04)[1]
idx7 = np.where(np.abs(z_diff7)>0.04)[1]
idx8 = np.where(np.abs(z_diff8)>0.04)[1]
print(idx3)
print(idx4)
print(idx5)
print(idx6)
print(idx7)
print(idx8)

hist_lat = []
hist_lat.append(idx3.tolist())
hist_lat.append(idx4.tolist())
hist_lat.append(idx5.tolist())
hist_lat.append(idx6.tolist())
hist_lat.append(idx7.tolist())
hist_lat.append(idx8.tolist())
print(hist_lat)

a = []
for it in hist_lat:
    for i in it:
        a.append(i)


set_a = set(a)
dict_a = {item:a.count(item) for item in set_a}
sorted_a = sorted(dict_a.items(), key=lambda x:x[1], reverse=True)

print(sorted_a)

#plt.hist(a, 500)
#plt.xlim(0,5000)
#plt.show()


idxa = np.where(np.abs(z_diff_a)>0.03)[1]
idxb = np.where(np.abs(z_diff_b)>0.03)[1]
idxc = np.where(np.abs(z_diff_c)>0.03)[1]
print(idxa)
print(idxb)
print(idxc)

hist_lat = []
hist_lat.append(idxa.tolist())
hist_lat.append(idxb.tolist())
hist_lat.append(idxc.tolist())
print(hist_lat)

a = []
for it in hist_lat:
    for i in it:
        a.append(i)


set_a = set(a)
dict_a = {item:a.count(item) for item in set_a}
sorted_a = sorted(dict_a.items(), key=lambda x:x[1], reverse=True)

print(sorted_a)



x = np.arange(7168)

print(np.linalg.norm(z_diff0[0]))
print(np.linalg.norm(z_diff1[0]))
print(np.linalg.norm(z_diff2[0]))
print(np.linalg.norm(z_diff3[0]))
print(np.linalg.norm(z_diff4[0]))
print(np.linalg.norm(z_diff5[0]))
print(np.linalg.norm(z_diff6[0]))
print(np.linalg.norm(z_diff7[0]))
print(np.linalg.norm(z_diff8[0]))

print('---')
print(np.linalg.norm(z_diff_a[0]))
print(np.linalg.norm(z_diff_b[0]))
print(np.linalg.norm(z_diff_c[0]))



z_diff3[np.abs(z_diff3)<0.025]=0
z_diff4[np.abs(z_diff4)<0.04]=0
z_diff5[np.abs(z_diff5)<0.04]=0
z_diff6[np.abs(z_diff6)<0.04]=0
z_diff7[np.abs(z_diff7)<0.04]=0
z_diff8[np.abs(z_diff8)<0.04]=0


z_diff_a[np.abs(z_diff_a)<0.02]=0
z_diff_b[np.abs(z_diff_b)<0.02]=0




plt.subplot(8,1,1)
plt.plot(x, z_diff3[0])

plt.subplot(8,1,2)
plt.plot(x, z_diff4[0])


plt.subplot(8,1,3)
plt.plot(x, z_diff5[0])

plt.subplot(8,1,4)
plt.plot(x, z_diff6[0])

plt.subplot(8,1,5)
plt.plot(x, z_diff7[0])

plt.subplot(8,1,6)
plt.plot(x, z_diff8[0])

plt.subplot(8,1,7)
plt.plot(x, z_diff_a[0])

plt.subplot(8,1,8)
plt.plot(x, z_diff_b[0])


plt.show()
