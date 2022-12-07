import numpy as np
from matplotlib import pyplot as plt


z = np.load('latent_z.npy')
z_init = np.load('latent_z_init.npy')
z_c = np.load('latent_z_control.npy')
'''
z = np.load('latent_z2.npy')
z_init = np.load('latent_z_init2.npy')
z_c = np.load('latent_z_control2.npy')
'''

'''
z[np.abs(z)<1.0]=0
z_init[np.abs(z_init)<1.0]=0
z_c[np.abs(z_c)<1.0]=0
'''


z_diff = z-z_c
z_diff[np.abs(z_diff)<0.2]=0
idx = np.where(np.abs(z_diff)>0.2)
print(idx[1])

x = np.arange(7168)


plt.plot(x, z_diff[0], '*')

'''
plt.subplot(3,1,1)
plt.plot(x, z[0], '*')

#plt.subplot(3,1,2)
plt.plot(x, z_init[0])


plt.subplot(3,1,3)
plt.plot(x, z[0], '*')
plt.plot(x, z_c[0])
'''
plt.show()
