import numpy as np
from matplotlib import pyplot as plt


z = np.load('latent_z_ep50.npy')
z_init = np.load('latent_z_init_ep50.npy')
z_c = np.load('latent_z_control_ep50.npy')

z_47 = np.load('latent_z_ep47.npy')
z_48 = np.load('latent_z_ep48.npy')
z_49 = np.load('latent_z_ep49.npy')
z_51 = np.load('latent_z_ep51.npy')
z_52 = np.load('latent_z_ep52.npy')
z_53 = np.load('latent_z_ep53.npy')
z_54 = np.load('latent_z_ep54.npy')
z_55 = np.load('latent_z_ep55.npy')
z_56 = np.load('latent_z_ep56.npy')
z_57 = np.load('latent_z_ep57.npy')
z_58 = np.load('latent_z_ep58.npy')
z_59 = np.load('latent_z_ep59.npy')
z_60 = np.load('latent_z_ep60.npy')
z_61 = np.load('latent_z_ep61.npy')
z_62 = np.load('latent_z_ep62.npy')
z_63 = np.load('latent_z_ep63.npy')


z_64 = np.load('latent_z_ep64.npy')
z_65 = np.load('latent_z_ep65.npy')
z_66 = np.load('latent_z_ep66.npy')
'''
z[np.abs(z)<1.0]=0
z_init[np.abs(z_init)<1.0]=0
z_c[np.abs(z_c)<1.0]=0
'''


z_diff0 = z_48 - z_47
z_diff1 = z_49 - z_48
z_diff2 = z - z_49

z_diff3 = z_51 - z
z_diff4 = z_52 - z_51
z_diff5 = z_53 - z_52
z_diff6 = z_54 - z_53
z_diff7 = z_55 - z_54
z_diff8 = z_56 - z_55
z_diff9 = z_57 - z_56
z_diff10 = z_58 - z_57
z_diff11 = z_59 - z_58
z_diff12 = z_60 - z_59
z_diff13 = z_61 - z_60
z_diff14 = z_62 - z_61
z_diff15 = z_63 - z_62


z_diff16 = z_64 - z_63
z_diff17 = z_65 - z_64
z_diff18 = z_66 - z_65


idx3 = np.where(np.abs(z_diff3)>0.05)[1]
idx4 = np.where(np.abs(z_diff4)>0.1)[1]
idx5 = np.where(np.abs(z_diff5)>0.15)[1]
idx6 = np.where(np.abs(z_diff6)>0.15)[1]
idx7 = np.where(np.abs(z_diff7)>0.15)[1]
idx8 = np.where(np.abs(z_diff8)>0.1)[1]
idx9 = np.where(np.abs(z_diff9)>0.1)[1]
idx10 = np.where(np.abs(z_diff10)>0.1)[1]
idx11 = np.where(np.abs(z_diff11)>0.1)[1]
print(idx3)
print(idx4)
print(idx5)
print(idx6)

hist_lat = []
hist_lat.append(idx3.tolist())
hist_lat.append(idx4.tolist())
hist_lat.append(idx5.tolist())
hist_lat.append(idx6.tolist())
hist_lat.append(idx7.tolist())
hist_lat.append(idx8.tolist())
hist_lat.append(idx9.tolist())
hist_lat.append(idx10.tolist())
hist_lat.append(idx11.tolist())
#print(hist_lat)

print('---')
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

print('---')
rst = []
for itm in sorted_a:
    #if itm[1] == 7:
    if itm[1] > 0:
        rst.append(itm[0])
print(rst)
np.save('advbit_0.npy', rst)


x = np.arange(7168)

print('---')
print(np.linalg.norm(z_diff0[0]))
print(np.linalg.norm(z_diff1[0]))
print(np.linalg.norm(z_diff2[0]))

print(np.linalg.norm(z_diff3[0]))
print(np.linalg.norm(z_diff4[0]))
print(np.linalg.norm(z_diff5[0]))
print(np.linalg.norm(z_diff6[0]))
print(np.linalg.norm(z_diff7[0]))
print(np.linalg.norm(z_diff8[0]))
print(np.linalg.norm(z_diff9[0]))
print(np.linalg.norm(z_diff10[0]))
print(np.linalg.norm(z_diff11[0]))

print(np.linalg.norm(z_diff12[0]))
print(np.linalg.norm(z_diff13[0]))
print(np.linalg.norm(z_diff14[0]))
print(np.linalg.norm(z_diff15[0]))
print(np.linalg.norm(z_diff16[0]))
print(np.linalg.norm(z_diff17[0]))
print(np.linalg.norm(z_diff18[0]))
print('---')



z_diff3[np.abs(z_diff3)<0.05]=0
z_diff4[np.abs(z_diff4)<0.1]=0
z_diff5[np.abs(z_diff5)<0.15]=0
z_diff6[np.abs(z_diff6)<0.15]=0
z_diff7[np.abs(z_diff7)<0.15]=0

z_diff8[np.abs(z_diff8)<0.1]=0
#z_diff9[np.abs(z_diff9)<0.1]=0
z_diff10[np.abs(z_diff10)<0.1]=0
z_diff11[np.abs(z_diff11)<0.1]=0




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
plt.plot(x, z_diff10[0])

plt.subplot(8,1,8)
plt.plot(x, z_diff11[0])


plt.show()
