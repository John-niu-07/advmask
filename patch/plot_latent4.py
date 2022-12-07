import numpy as np
from matplotlib import pyplot as plt



z_0 = np.load('latent_z_ep0_.npy')
z_1 = np.load('latent_z_ep1_.npy')
z_2 = np.load('latent_z_ep2_.npy')
z_3 = np.load('latent_z_ep3_.npy')
z_4 = np.load('latent_z_ep4_.npy')
z_5 = np.load('latent_z_ep5_.npy')
z_6 = np.load('latent_z_ep6_.npy')
z_7 = np.load('latent_z_ep7_.npy')
z_8 = np.load('latent_z_ep8_.npy')
z_9 = np.load('latent_z_ep9_.npy')
z_10 = np.load('latent_z_ep10_.npy')
z_11 = np.load('latent_z_ep11_.npy')
z_12 = np.load('latent_z_ep12_.npy')
z_13 = np.load('latent_z_ep13_.npy')
z_14 = np.load('latent_z_ep14_.npy')
z_15 = np.load('latent_z_ep15_.npy')
z_16 = np.load('latent_z_ep16_.npy')
z_17 = np.load('latent_z_ep17_.npy')

'''
z[np.abs(z)<1.0]=0
z_init[np.abs(z_init)<1.0]=0
z_c[np.abs(z_c)<1.0]=0
'''


z_diff0 = z_1 - z_0
z_diff1 = z_2 - z_1
z_diff2 = z_3 - z_2

z_diff3 = z_4 - z_3
z_diff4 = z_5 - z_4
z_diff5 = z_6 - z_5
z_diff6 = z_7 - z_6
z_diff7 = z_8 - z_7
z_diff8 = z_9 - z_8

z_diff9 = z_10 - z_9
z_diff10 = z_11 - z_10
z_diff11 = z_12 - z_11
z_diff12 = z_13 - z_12
z_diff13 = z_14 - z_13
z_diff14 = z_15 - z_14
z_diff15 = z_16 - z_15
z_diff16 = z_17 - z_16


idx0 = np.where(np.abs(z_diff0)>0.009)[1]
idx1 = np.where(np.abs(z_diff1)>0.009)[1]
idx2 = np.where(np.abs(z_diff2)>0.009)[1]
idx3 = np.where(np.abs(z_diff3)>0.009)[1]
idx4 = np.where(np.abs(z_diff4)>0.009)[1]
idx5 = np.where(np.abs(z_diff5)>0.009)[1]
idx6 = np.where(np.abs(z_diff6)>0.009)[1]
idx7 = np.where(np.abs(z_diff7)>0.009)[1]
idx8 = np.where(np.abs(z_diff8)>0.009)[1]

idx9 = np.where(np.abs(z_diff9)>0.008)[1]
idx10 = np.where(np.abs(z_diff10)>0.008)[1]
idx11 = np.where(np.abs(z_diff11)>0.008)[1]
idx12 = np.where(np.abs(z_diff12)>0.008)[1]
idx13 = np.where(np.abs(z_diff13)>0.008)[1]
idx14 = np.where(np.abs(z_diff14)>0.008)[1]
idx15 = np.where(np.abs(z_diff15)>0.008)[1]
idx16 = np.where(np.abs(z_diff16)>0.008)[1]
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
hist_lat.append(idx0.tolist())
hist_lat.append(idx1.tolist())
hist_lat.append(idx2.tolist())
#print(hist_lat)
hist_lat.append(idx9.tolist())
hist_lat.append(idx10.tolist())
hist_lat.append(idx11.tolist())
hist_lat.append(idx12.tolist())
hist_lat.append(idx13.tolist())
hist_lat.append(idx14.tolist())
hist_lat.append(idx15.tolist())
hist_lat.append(idx16.tolist())

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
np.save('Advbit_0_.npy', rst)


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
print('---')
print(np.linalg.norm(z_diff9[0]))
print(np.linalg.norm(z_diff10[0]))
print(np.linalg.norm(z_diff11[0]))
print(np.linalg.norm(z_diff12[0]))
print(np.linalg.norm(z_diff13[0]))
print(np.linalg.norm(z_diff14[0]))
print(np.linalg.norm(z_diff15[0]))
print(np.linalg.norm(z_diff16[0]))



z_diff3[np.abs(z_diff3)<0.009]=0
z_diff4[np.abs(z_diff4)<0.009]=0
z_diff5[np.abs(z_diff5)<0.009]=0
z_diff6[np.abs(z_diff6)<0.009]=0
z_diff7[np.abs(z_diff7)<0.009]=0
z_diff8[np.abs(z_diff8)<0.009]=0

z_diff0[np.abs(z_diff0)<0.009]=0
z_diff1[np.abs(z_diff1)<0.009]=0
z_diff2[np.abs(z_diff2)<0.009]=0


z_diff9[np.abs(z_diff9)<0.009]=0
z_diff10[np.abs(z_diff10)<0.009]=0
z_diff11[np.abs(z_diff11)<0.008]=0
z_diff12[np.abs(z_diff12)<0.009]=0
z_diff13[np.abs(z_diff13)<0.008]=0
z_diff14[np.abs(z_diff14)<0.009]=0
z_diff15[np.abs(z_diff15)<0.008]=0
z_diff16[np.abs(z_diff16)<0.009]=0


plt.subplot(8,1,1)
plt.plot(x, z_diff0[0])

plt.subplot(8,1,2)
plt.plot(x, z_diff1[0])


plt.subplot(8,1,3)
plt.plot(x, z_diff2[0])

plt.subplot(8,1,4)
plt.plot(x, z_diff3[0])

plt.subplot(8,1,5)
plt.plot(x, z_diff4[0])

plt.subplot(8,1,6)
plt.plot(x, z_diff11[0])


plt.subplot(8,1,7)
plt.plot(x, z_diff13[0])

plt.subplot(8,1,8)
plt.plot(x, z_diff15[0])


plt.show()
