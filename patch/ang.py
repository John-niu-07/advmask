import numpy as np


img7 = np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test7/inverted_codes.npy').reshape((7168))
img8 = np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test8/inverted_codes.npy').reshape((7168))
adv7 = np.load('latent_z_adv7_.npy')[0]
adv8 = np.load('latent_z_adv8_.npy')[0]
#adv8_2 = np.load('latent_z_adv8_2.npy')[0]
adv8_2 = np.load('latent_z_adv8_2_.npy')[0]
adv8_3 = np.load('latent_z_adv8_3.npy')[0]
adv8_4 = np.load('latent_z_adv8_4.npy')[0]
adv8_5 = np.load('latent_z_adv8_5.npy')[0]
adv8_6 = np.load('latent_z_adv8_6.npy')[0]
adv8_7 = np.load('latent_z_adv8_7.npy')[0]
adv8_8 = np.load('latent_z_adv8_8.npy')[0]
adv8_9 = np.load('latent_z_adv8_9.npy')[0]
adv8_10 = np.load('latent_z_adv8_10.npy')[0]
adv8_11 = np.load('latent_z_adv8_11.npy')[0]
adv8_12 = np.load('latent_z_adv8_12.npy')[0]
adv8_13 = np.load('latent_z_adv8_13.npy')[0]
adv8_14 = np.load('latent_z_adv8_14.npy')[0]
adv8_15 = np.load('latent_z_adv8_15.npy')[0]
adv8_16 = np.load('latent_z_adv8_16.npy')[0]
adv8_17 = np.load('latent_z_adv8_17.npy')[0]
adv8_18 = np.load('latent_z_adv8_18.npy')[0]
adv8_19 = np.load('latent_z_adv8_19.npy')[0]
adv8_20 = np.load('latent_z_adv8_20.npy')[0]
adv8_21 = np.load('latent_z_adv8_21.npy')[0]
adv8_22 = np.load('latent_z_adv8_22.npy')[0]
adv8_23 = np.load('latent_z_adv8_23.npy')[0]
adv8_24 = np.load('latent_z_adv8_24.npy')[0]
adv8_25 = np.load('latent_z_adv8_25.npy')[0]
adv8_26 = np.load('latent_z_adv8_26.npy')[0]
adv8_27 = np.load('latent_z_adv8_27.npy')[0]
adv8_28 = np.load('latent_z_adv8_28.npy')[0]
adv8_29 = np.load('latent_z_adv8_29.npy')[0]
adv8_30 = np.load('latent_z_adv8_30.npy')[0]
adv8_31 = np.load('latent_z_adv8_31.npy')[0]
adv8_32 = np.load('latent_z_adv8_32.npy')[0]
adv8_t = np.load('latent_z_adv8_33.npy')[0]
img10= np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test10/inverted_codes.npy').reshape((7168))
#img9= np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test9/inverted_codes.npy').reshape((7168))
img6= np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test6/inverted_codes.npy').reshape((7168))
adv6 = np.load('latent_z_adv6_.npy')[0]


vec_8_adv8 = adv8 - img8
vec_7_adv7 = adv7 - img7
vec_8_7 = img7 - img8
vec_7_8 = img8 - img7


def ang(a,b):
    cos_ = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    ang_ = np.degrees( np.arccos(cos_) )
    return ang_


#cos_adv8_img8_img7 = np.dot(vec_8_7, vec_8_adv8) / (np.linalg.norm(vec_8_7) * np.linalg.norm(vec_8_adv8))
#print(cos_adv8_img8_img7)
ang_adv8_img8_img7 = ang(vec_8_7, vec_8_adv8)
print('ang   adv8_img8_img7: ' + str(ang_adv8_img8_img7) )


#cos_adv8_o_adv7 = np.dot(vec_8_adv8, vec_7_adv7) / (np.linalg.norm(vec_8_adv8) * np.linalg.norm(vec_7_adv7))
#print(cos_adv8_o_adv7)


ang_adv8_o_adv7 = ang(vec_8_adv8, vec_7_adv7)
vec_6_adv6 = adv6 - img6
print('ang   adv8_o_adv7   : ' + str(ang_adv8_o_adv7) )
print('ang   adv8_o_adv6   : ' + str(ang(vec_8_adv8, vec_6_adv6)) )
print('ang   adv7_o_adv6   : ' + str(ang(vec_7_adv7, vec_6_adv6)) )

ang_adv7_img7_img8 = ang(vec_7_adv7, vec_7_8)
print('ang   adv7_img7_img8: ' + str(ang_adv7_img7_img8) )
ang_adv7_img8_img7 = ang(vec_7_adv7, vec_8_7)
print('ang   adv7_img8_img7: ' + str(ang_adv7_img8_img7) )



vec_7_10 = img10 - img7
ang_adv7_img7_img10 = ang(vec_7_adv7, vec_7_10)
print('ang   adv7_img7_img10: ' + str(ang_adv7_img7_img10) )

ang_img8_img7_img10 = ang(vec_7_8, vec_7_10)
print('ang   img8_img7_img10: ' + str(ang_img8_img7_img10) )




vec_8_10 = img10 - img8
ang_adv8_img8_img10 = ang(vec_8_10, vec_8_adv8)
print('ang   adv8_img8_img10: ' + str(ang_adv8_img8_img10) )

ang_img10_img8_img7 = ang(vec_8_10, vec_8_7)
print('ang   img10_img8_img7: ' + str(ang_img10_img8_img7) )


vec_6_10 = img10 - img6
vec_6_7 = img7 - img6
ang_img10_img6_img7 = ang(vec_6_10, vec_6_7)
print('ang   img10_img6_img7: ' + str(ang_img10_img6_img7) )

vec_6_8 = img8 - img6
ang_img8_img6_img7 = ang(vec_6_8, vec_6_7)
print('ang   img8_img6_img7: ' + str(ang_img8_img6_img7) )


ang_img8_img6_img10 = ang(vec_6_8, vec_6_10)
print('ang   img8_img6_img10: ' + str(ang_img8_img6_img10) )

vec_adv8_8 = img8 - adv8
vec_adv8_10 = img10 - adv8
ang_img8_adv8_img10 = ang(vec_adv8_8, vec_adv8_10)
print('ang   img8_adv8_img10: ' + str(ang_img8_adv8_img10) )


dist_adv8_adv7 = np.linalg.norm(adv8 - adv7)
dist_adv8_img10 = np.linalg.norm(adv8 - img10)
dist_adv7_img10 = np.linalg.norm(adv7 - img10)
dist_img8_img7 = np.linalg.norm(img8 - img7)
dist_img7_img10 = np.linalg.norm(img7 - img10)
dist_img8_img10 = np.linalg.norm(img8 - img10)

dist_adv8_img8 = np.linalg.norm(img8 - adv8)
dist_adv8t_img10 = np.linalg.norm(adv8_t - img10)
dist_adv8t_img8 = np.linalg.norm(img8 - adv8_t)
dist_adv8t_img7 = np.linalg.norm(img7 - adv8_t)
dist_adv7_img7 = np.linalg.norm(img7 - adv7)
dist_adv6_img6 = np.linalg.norm(img6 - adv6)

print('dist   adv8_adv7: ' + str(dist_adv8_adv7) )
print('dist   adv8_img10: ' + str(dist_adv8_img10) )
print('dist   adv7_img10: ' + str(dist_adv7_img10) )
print('norm   adv7: ' + str(dist_adv7_img7) )
print('norm   adv6: ' + str(dist_adv6_img6) )
print('norm   adv8: ' + str(dist_adv8_img8) )
print('norm   adv82: ' + str( np.linalg.norm(img8 - adv8_2) ) )
print('norm   adv83: ' + str( np.linalg.norm(img8 - adv8_3) ) )
print('norm   adv84: ' + str( np.linalg.norm(img8 - adv8_4) ) )
print('norm   adv85: ' + str( np.linalg.norm(img8 - adv8_5) ) )
print('norm   adv86: ' + str( np.linalg.norm(img8 - adv8_6) ) )
print('norm   adv87: ' + str( np.linalg.norm(img8 - adv8_7) ) )
print('norm   adv88: ' + str( np.linalg.norm(img8 - adv8_8) ) )
print('norm   adv89: ' + str( np.linalg.norm(img8 - adv8_9) ) )
print('norm   adv810: ' + str( np.linalg.norm(img8 - adv8_10) ) )
print('norm   adv811: ' + str( np.linalg.norm(img8 - adv8_11) ) )
print('norm   adv812: ' + str( np.linalg.norm(img8 - adv8_12) ) )
print('norm   adv813: ' + str( np.linalg.norm(img8 - adv8_13) ) )
print('norm   adv814: ' + str( np.linalg.norm(img8 - adv8_14) ) )
print('norm   adv815: ' + str( np.linalg.norm(img8 - adv8_15) ) )
print('norm   adv816: ' + str( np.linalg.norm(img8 - adv8_16) ) )
print('norm   adv817: ' + str( np.linalg.norm(img8 - adv8_17) ) )
print('norm   adv818: ' + str( np.linalg.norm(img8 - adv8_18) ) )
print('norm   adv819: ' + str( np.linalg.norm(img8 - adv8_19) ) )
print('norm   adv820: ' + str( np.linalg.norm(img8 - adv8_20) ) )
print('norm   adv821: ' + str( np.linalg.norm(img8 - adv8_21) ) )
print('norm   adv822: ' + str( np.linalg.norm(img8 - adv8_22) ) )
print('norm   adv823: ' + str( np.linalg.norm(img8 - adv8_23) ) )
print('norm   adv824: ' + str( np.linalg.norm(img8 - adv8_24) ) )
print('norm   adv825: ' + str( np.linalg.norm(img8 - adv8_25) ) )
print('norm   adv826: ' + str( np.linalg.norm(img8 - adv8_26) ) )
print('norm   adv827: ' + str( np.linalg.norm(img8 - adv8_27) ) )
print('norm   adv828: ' + str( np.linalg.norm(img8 - adv8_28) ) )
print('norm   adv829: ' + str( np.linalg.norm(img8 - adv8_29) ) )
print('norm   adv830: ' + str( np.linalg.norm(img8 - adv8_30) ) )
print('norm   adv831: ' + str( np.linalg.norm(img8 - adv8_31) ) )
print('norm   adv832: ' + str( np.linalg.norm(img8 - adv8_32) ) )

print('norm   adv8t: ' + str(dist_adv8t_img8) )

print('dist   adv8t_adv8: ' + str( np.linalg.norm(adv8 - adv8_t) ) )
print('dist   adv8t_adv82: ' + str( np.linalg.norm(adv8_2 - adv8_t) ) )
print('dist   adv8t_adv83: ' + str( np.linalg.norm(adv8_3 - adv8_t) ) )
print('dist   adv8t_adv84: ' + str( np.linalg.norm(adv8_4 - adv8_t) ) )
print('dist   adv8t_adv85: ' + str( np.linalg.norm(adv8_5 - adv8_t) ) )
print('dist   adv8t_adv86: ' + str( np.linalg.norm(adv8_6 - adv8_t) ) )
print('dist   adv8t_adv87: ' + str( np.linalg.norm(adv8_7 - adv8_t) ) )
print('dist   adv8t_adv88: ' + str( np.linalg.norm(adv8_8 - adv8_t) ) )
print('dist   adv8t_adv89: ' + str( np.linalg.norm(adv8_9 - adv8_t) ) )
print('dist   adv8t_adv810: ' + str( np.linalg.norm(adv8_10 - adv8_t) ) )
print('dist   adv8t_adv811: ' + str( np.linalg.norm(adv8_11 - adv8_t) ) )
print('dist   adv8t_adv812: ' + str( np.linalg.norm(adv8_12 - adv8_t) ) )
print('dist   adv8t_adv813: ' + str( np.linalg.norm(adv8_13 - adv8_t) ) )
print('dist   adv8t_adv814: ' + str( np.linalg.norm(adv8_14 - adv8_t) ) )
print('dist   adv8t_adv815: ' + str( np.linalg.norm(adv8_15 - adv8_t) ) )

print(np.max(adv8_t))
print(np.min(adv8_t))
print('---')
print('dist   adv8_img10: ' + str(dist_adv8_img10) )
print('dist   adv82_img10: ' + str( np.linalg.norm(adv8_2 - img10) ) )
print('dist   adv83_img10: ' + str( np.linalg.norm(adv8_3 - img10) ) )
print('dist   adv84_img10: ' + str( np.linalg.norm(adv8_4 - img10) ) )
print('dist   adv85_img10: ' + str( np.linalg.norm(adv8_5 - img10) ) )
print('dist   adv86_img10: ' + str( np.linalg.norm(adv8_6 - img10) ) )
print('dist   adv87_img10: ' + str( np.linalg.norm(adv8_7 - img10) ) )
print('dist   adv88_img10: ' + str( np.linalg.norm(adv8_8 - img10) ) )
print('dist   adv89_img10: ' + str( np.linalg.norm(adv8_9 - img10) ) )
print('dist   adv810_img10: ' + str( np.linalg.norm(adv8_10 - img10) ) )
print('dist   adv811_img10: ' + str( np.linalg.norm(adv8_11 - img10) ) )
print('dist   adv812_img10: ' + str( np.linalg.norm(adv8_12 - img10) ) )
print('dist   adv813_img10: ' + str( np.linalg.norm(adv8_13 - img10) ) )
print('dist   adv814_img10: ' + str( np.linalg.norm(adv8_14 - img10) ) )
print('dist   adv815_img10: ' + str( np.linalg.norm(adv8_15 - img10) ) )
print('dist   adv816_img10: ' + str( np.linalg.norm(adv8_16 - img10) ) )
print('dist   adv817_img10: ' + str( np.linalg.norm(adv8_17 - img10) ) )
print('dist   adv818_img10: ' + str( np.linalg.norm(adv8_18 - img10) ) )
print('dist   adv819_img10: ' + str( np.linalg.norm(adv8_19 - img10) ) )
print('dist   adv820_img10: ' + str( np.linalg.norm(adv8_20 - img10) ) )
print('dist   adv821_img10: ' + str( np.linalg.norm(adv8_21 - img10) ) )
print('dist   adv822_img10: ' + str( np.linalg.norm(adv8_22 - img10) ) )
print('dist   adv823_img10: ' + str( np.linalg.norm(adv8_23 - img10) ) )
print('dist   adv824_img10: ' + str( np.linalg.norm(adv8_24 - img10) ) )
print('dist   adv825_img10: ' + str( np.linalg.norm(adv8_25 - img10) ) )
print('dist   adv826_img10: ' + str( np.linalg.norm(adv8_26 - img10) ) )
print('dist   adv827_img10: ' + str( np.linalg.norm(adv8_27 - img10) ) )
print('dist   adv828_img10: ' + str( np.linalg.norm(adv8_28 - img10) ) )
print('dist   adv829_img10: ' + str( np.linalg.norm(adv8_29 - img10) ) )
print('dist   adv830_img10: ' + str( np.linalg.norm(adv8_30 - img10) ) )
print('dist   adv831_img10: ' + str( np.linalg.norm(adv8_31 - img10) ) )
print('dist   adv832_img10: ' + str( np.linalg.norm(adv8_32 - img10) ) )

print('dist   adv8t_img10: ' + str(dist_adv8t_img10) )

print('dist   adv8_adv8: ' + str( np.linalg.norm(adv8 - adv8) ) )
print('dist   adv82_: ' + str( np.linalg.norm(adv8_2 - adv8) ) )
print('dist   adv83_: ' + str( np.linalg.norm(adv8_3 - adv8) ) )
print('dist   adv84_: ' + str( np.linalg.norm(adv8_4 - adv8) ) )
print('dist   adv85_: ' + str( np.linalg.norm(adv8_5 - adv8) ) )
print('dist   adv86_: ' + str( np.linalg.norm(adv8_6 - adv8) ) )
print('dist   adv87_: ' + str( np.linalg.norm(adv8_7 - adv8) ) )
print('dist   adv88_: ' + str( np.linalg.norm(adv8_8 - adv8) ) )
print('dist   adv89_: ' + str( np.linalg.norm(adv8_9 - adv8) ) )
print('dist   adv810_: ' + str( np.linalg.norm(adv8_10 - adv8) ) )
print('dist   adv811_: ' + str( np.linalg.norm(adv8_11 - adv8) ) )
print('dist   adv812_: ' + str( np.linalg.norm(adv8_12 - adv8) ) )
print('dist   adv813_: ' + str( np.linalg.norm(adv8_13 - adv8) ) )
print('dist   adv814_: ' + str( np.linalg.norm(adv8_14 - adv8) ) )
print('dist   adv815_: ' + str( np.linalg.norm(adv8_15 - adv8) ) )

print('dist   adv8_adv82: ' + str( np.linalg.norm(adv8 - adv8_2) ) )
print('dist   adv82_: ' + str( np.linalg.norm(adv8_2 - adv8_2) ) )
print('dist   adv83_: ' + str( np.linalg.norm(adv8_3 - adv8_2) ) )
print('dist   adv84_: ' + str( np.linalg.norm(adv8_4 - adv8_2) ) )
print('dist   adv85_: ' + str( np.linalg.norm(adv8_5 - adv8_2) ) )
print('dist   adv86_: ' + str( np.linalg.norm(adv8_6 - adv8_2) ) )
print('dist   adv87_: ' + str( np.linalg.norm(adv8_7 - adv8_2) ) )
print('dist   adv88_: ' + str( np.linalg.norm(adv8_8 - adv8_2) ) )
print('dist   adv89_: ' + str( np.linalg.norm(adv8_9 - adv8_2) ) )
print('dist   adv810_: ' + str( np.linalg.norm(adv8_10 - adv8_2) ) )
print('dist   adv811_: ' + str( np.linalg.norm(adv8_11 - adv8_2) ) )
print('dist   adv812_: ' + str( np.linalg.norm(adv8_12 - adv8_2) ) )
print('dist   adv813_: ' + str( np.linalg.norm(adv8_13 - adv8_2) ) )
print('dist   adv814_: ' + str( np.linalg.norm(adv8_14 - adv8_2) ) )
print('dist   adv815_: ' + str( np.linalg.norm(adv8_15 - adv8_2) ) )

print('dist   adv8_adv83: ' + str( np.linalg.norm(adv8 - adv8_3) ) )
print('dist   adv82_: ' + str( np.linalg.norm(adv8_2 - adv8_3) ) )
print('dist   adv83_: ' + str( np.linalg.norm(adv8_3 - adv8_3) ) )
print('dist   adv84_: ' + str( np.linalg.norm(adv8_4 - adv8_3) ) )
print('dist   adv85_: ' + str( np.linalg.norm(adv8_5 - adv8_3) ) )
print('dist   adv86_: ' + str( np.linalg.norm(adv8_6 - adv8_3) ) )
print('dist   adv87_: ' + str( np.linalg.norm(adv8_7 - adv8_3) ) )
print('dist   adv88_: ' + str( np.linalg.norm(adv8_8 - adv8_3) ) )

print('dist   adv8_adv84: ' + str( np.linalg.norm(adv8 - adv8_4) ) )
print('dist   adv82_: ' + str( np.linalg.norm(adv8_2 - adv8_4) ) )
print('dist   adv83_: ' + str( np.linalg.norm(adv8_3 - adv8_4) ) )
print('dist   adv84_: ' + str( np.linalg.norm(adv8_4 - adv8_4) ) )
print('dist   adv85_: ' + str( np.linalg.norm(adv8_5 - adv8_4) ) )
print('dist   adv86_: ' + str( np.linalg.norm(adv8_6 - adv8_4) ) )
print('dist   adv87_: ' + str( np.linalg.norm(adv8_7 - adv8_4) ) )
print('dist   adv88_: ' + str( np.linalg.norm(adv8_8 - adv8_4) ) )

print('dist   adv8_adv85: ' + str( np.linalg.norm(adv8 - adv8_5) ) )
print('dist   adv82_: ' + str( np.linalg.norm(adv8_2 - adv8_5) ) )
print('dist   adv83_: ' + str( np.linalg.norm(adv8_3 - adv8_5) ) )
print('dist   adv84_: ' + str( np.linalg.norm(adv8_4 - adv8_5) ) )
print('dist   adv85_: ' + str( np.linalg.norm(adv8_5 - adv8_5) ) )
print('dist   adv86_: ' + str( np.linalg.norm(adv8_6 - adv8_5) ) )
print('dist   adv87_: ' + str( np.linalg.norm(adv8_7 - adv8_5) ) )
print('dist   adv88_: ' + str( np.linalg.norm(adv8_8 - adv8_5) ) )

print('dist   adv8_adv86: ' + str( np.linalg.norm(adv8 - adv8_6) ) )
print('dist   adv82_: ' + str( np.linalg.norm(adv8_2 - adv8_6) ) )
print('dist   adv83_: ' + str( np.linalg.norm(adv8_3 - adv8_6) ) )
print('dist   adv84_: ' + str( np.linalg.norm(adv8_4 - adv8_6) ) )
print('dist   adv85_: ' + str( np.linalg.norm(adv8_5 - adv8_6) ) )
print('dist   adv86_: ' + str( np.linalg.norm(adv8_6 - adv8_6) ) )
print('dist   adv87_: ' + str( np.linalg.norm(adv8_7 - adv8_6) ) )
print('dist   adv88_: ' + str( np.linalg.norm(adv8_8 - adv8_6) ) )

print('dist   adv8_adv87: ' + str( np.linalg.norm(adv8 - adv8_7) ) )
print('dist   adv82_: ' + str( np.linalg.norm(adv8_2 - adv8_7) ) )
print('dist   adv83_: ' + str( np.linalg.norm(adv8_3 - adv8_7) ) )
print('dist   adv84_: ' + str( np.linalg.norm(adv8_4 - adv8_7) ) )
print('dist   adv85_: ' + str( np.linalg.norm(adv8_5 - adv8_7) ) )
print('dist   adv86_: ' + str( np.linalg.norm(adv8_6 - adv8_7) ) )
print('dist   adv87_: ' + str( np.linalg.norm(adv8_7 - adv8_7) ) )
print('dist   adv88_: ' + str( np.linalg.norm(adv8_8 - adv8_7) ) )

print('dist   adv8_adv88: ' + str( np.linalg.norm(adv8 - adv8_8) ) )
print('dist   adv82_: ' + str( np.linalg.norm(adv8_2 - adv8_8) ) )
print('dist   adv83_: ' + str( np.linalg.norm(adv8_3 - adv8_8) ) )
print('dist   adv84_: ' + str( np.linalg.norm(adv8_4 - adv8_8) ) )
print('dist   adv85_: ' + str( np.linalg.norm(adv8_5 - adv8_8) ) )
print('dist   adv86_: ' + str( np.linalg.norm(adv8_6 - adv8_8) ) )
print('dist   adv87_: ' + str( np.linalg.norm(adv8_7 - adv8_8) ) )
print('dist   adv88_: ' + str( np.linalg.norm(adv8_8 - adv8_8) ) )
print('dist   adv89_: ' + str( np.linalg.norm(adv8_9 - adv8_8) ) )
print('dist   adv810_: ' + str( np.linalg.norm(adv8_10 - adv8_8) ) )
print('dist   adv811_: ' + str( np.linalg.norm(adv8_11 - adv8_8) ) )
print('dist   adv812_: ' + str( np.linalg.norm(adv8_12 - adv8_8) ) )
print('dist   adv813_: ' + str( np.linalg.norm(adv8_13 - adv8_8) ) )
print('dist   adv814_: ' + str( np.linalg.norm(adv8_14 - adv8_8) ) )
print('dist   adv815_: ' + str( np.linalg.norm(adv8_15 - adv8_8) ) )

print('dist   adv8_adv15: ' + str( np.linalg.norm(adv8 - adv8_15) ) )
print('dist   adv82_: ' + str( np.linalg.norm(adv8_2 - adv8_15) ) )
print('dist   adv83_: ' + str( np.linalg.norm(adv8_3 - adv8_15) ) )
print('dist   adv84_: ' + str( np.linalg.norm(adv8_4 - adv8_15) ) )
print('dist   adv85_: ' + str( np.linalg.norm(adv8_5 - adv8_15) ) )
print('dist   adv86_: ' + str( np.linalg.norm(adv8_6 - adv8_15) ) )
print('dist   adv87_: ' + str( np.linalg.norm(adv8_7 - adv8_15) ) )
print('dist   adv88_: ' + str( np.linalg.norm(adv8_8 - adv8_15) ) )
print('dist   adv89_: ' + str( np.linalg.norm(adv8_9 - adv8_15) ) )
print('dist   adv810_: ' + str( np.linalg.norm(adv8_10 - adv8_15) ) )
print('dist   adv811_: ' + str( np.linalg.norm(adv8_11 - adv8_15) ) )
print('dist   adv812_: ' + str( np.linalg.norm(adv8_12 - adv8_15) ) )
print('dist   adv813_: ' + str( np.linalg.norm(adv8_13 - adv8_15) ) )
print('dist   adv814_: ' + str( np.linalg.norm(adv8_14 - adv8_15) ) )
print('dist   adv815_: ' + str( np.linalg.norm(adv8_15 - adv8_15) ) )
print('dist   adv816_: ' + str( np.linalg.norm(adv8_16 - adv8_15) ) )


print('dist   adv8t_img7: ' + str(dist_adv8t_img7) )
print('dist   img8_img7: ' + str(dist_img8_img7) )
print('dist   img7_img10: ' + str(dist_img7_img10) )
print('dist   img8_img10: ' + str(dist_img8_img10) )


print('norm   img8: ' + str(np.linalg.norm(img8)) )
print('norm   img7: ' + str(np.linalg.norm(img7)) )
print('norm   img10: ' + str(np.linalg.norm(img10)) )
print('norm   img6: ' + str(np.linalg.norm(img6)) )
print('dist   img6_img10: ' + str(np.linalg.norm(img6 - img10)) )
print('dist   img6_img8: ' + str(np.linalg.norm(img6 - img8)) )
print('dist   img6_img7: ' + str(np.linalg.norm(img6 - img7)) )



print('ang   img8_o_img10: ' + str(ang(img8, img10)) )
print('ang   img8_o_img7: ' + str(ang(img8, img7)) )
print('ang   img7_o_img10: ' + str(ang(img10, img7)) )

print('ang   img6_o_img10: ' + str(ang(img10, img6)) )

#print('dist   adv8_0: ' + str(np.linalg.norm(adv8)) )
#print('dist   adv7_0: ' + str(np.linalg.norm(adv7)) )

vec_6_10 = img10 - adv6
ang_adv6_img6_img10 = ang(vec_6_adv6, vec_6_10)
print('ang   adv6_img6_img10: ' + str(ang_adv6_img6_img10) )

vec_6_7 = img7 - adv6
ang_adv6_img6_img7 = ang(vec_6_adv6, vec_6_7)
print('ang   adv6_img6_img7: ' + str(ang_adv6_img6_img7) )

vec_6_8 = img8 - adv6
ang_adv6_img6_img8 = ang(vec_6_adv6, vec_6_8)
print('ang   adv6_img6_img8: ' + str(ang_adv6_img6_img8) )
