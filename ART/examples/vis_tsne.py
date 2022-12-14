import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold 


#X = np.load('test5.npy')

#X = np.load('x0_tiny_woM_base.npy')
#X = np.load('x0_tiny_wM_base.npy')

#X = np.load('x0_tiny_woM.npy')
#X = np.load('x0_tiny_wM.npy')

#X = np.load('x0_tiny_woM_sp.npy')
#X = np.load('x0_tiny_wM_sp.npy')

#X = np.load('x0_cifar_woM_eps8.npy')
#X = np.load('x0_cifar_wM_eps8.npy')
#X = np.load('x0_cifar_woM_base_eps8.npy')
#X = np.load('x0_cifar_wM_base_eps8.npy')

#X = np.load('x0_cifar_woM_eps2.npy')
#X = np.load('x0_cifar_wM_eps2.npy')
#X = np.load('x0_cifar_woM_base_eps2.npy')
X = np.load('x0_cifar_wM_base_eps2.npy')
for kk in range(1, 16):
    print(kk)
    #x_t = np.load('x'+str(kk)+'_tiny_woM_base.npy')
    #x_t = np.load('x'+str(kk)+'_tiny_wM_base.npy')

    #x_t = np.load('x'+str(kk)+'_tiny_woM.npy')
    #x_t = np.load('x'+str(kk)+'_tiny_wM.npy')
    
    #x_t = np.load('x'+str(kk)+'_tiny_woM_sp.npy')
    #x_t = np.load('x'+str(kk)+'_tiny_wM_sp.npy')

    #x_t = np.load('x'+str(kk)+'_cifar_woM_eps8.npy')
    #x_t = np.load('x'+str(kk)+'_cifar_wM_eps8.npy')
    
    #x_t = np.load('x'+str(kk)+'_cifar_woM_base_eps8.npy')
    #x_t = np.load('x'+str(kk)+'_cifar_wM_base_eps8.npy')
    
    #x_t = np.load('x'+str(kk)+'_cifar_woM_eps2.npy')
    #x_t = np.load('x'+str(kk)+'_cifar_wM_eps2.npy')
    #x_t = np.load('x'+str(kk)+'_cifar_woM_base_eps2.npy')
    x_t = np.load('x'+str(kk)+'_cifar_wM_base_eps2.npy')
    print(x_t.shape)
    X = np.vstack((X, x_t))

print(X.shape)


#y_test = np.load('y_tiny_woM_base.npy')
#y_test = np.load('y_tiny_woM.npy')

y_test = np.load('y_cifar_woM_eps8.npy')
#y_test = np.load('y_cifar_woM_eps8.npy')
y = np.argmax(y_test, axis=1)
print(y)
print(y.shape)


tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
X_tsne = tsne.fit_transform(X)
print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
print(X_norm.shape)
plt.figure(figsize=(8, 8))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.show()
