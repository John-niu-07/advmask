import numpy as np
import matplotlib.pyplot as plt 

latent_adv = np.load('/face/Mask/AdversarialMask/patch/experiments/October/23-10-2022_09-41-04/final_results/latent_z.npy')
latent_tar = np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test2/inverted_codes.npy').reshape((1, 7168))
latent_init = np.load('../datasets/inverted_codes.npy').reshape((1, 7168))



latent_seg = []
for i in range(14):
    latent_seg.append(latent_init[0][512*i:512*(i+1)])
    print(latent_seg[-1])



print(np.linalg.norm(latent_adv, 2))
print(np.linalg.norm(latent_tar, 2))
print(np.linalg.norm(latent_init, 2))

to_init = latent_adv - latent_init

to_init_seg = []
for i in range(14):
    #to_init_seg.append(to_init[0][512*i:512*(i+1)])
    to_init_seg.append( np.linalg.norm(to_init[0][512*i:512*(i+1)], 2) )

print(to_init_seg)

b = np.flipud( np.argsort(to_init[0]) )

val = []
for i in range(1000):
    #print(to_init[0][b[i]])
    val.append(to_init[0][b[i]])

'''
plt.plot(val)
plt.show()
'''
print(np.linalg.norm(to_init, 2))

to_tar = latent_adv - latent_tar
print(np.linalg.norm(to_tar, 2))
