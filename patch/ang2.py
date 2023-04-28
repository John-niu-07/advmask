import numpy as np


img8 = np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test8/inverted_codes.npy').reshape((7168))
adv7 = np.load('latent_z_adv7_.npy')[0]
adv8C1 = np.load('latent_z_adv8C_1.npy')[0]
adv8C2 = np.load('latent_z_adv8C_2.npy')[0]
adv8C3 = np.load('latent_z_adv8C_3.npy')[0]
adv8C4 = np.load('latent_z_adv8C_4.npy')[0]
adv8C5 = np.load('latent_z_adv8C_5.npy')[0]
adv8C6 = np.load('latent_z_adv8C_6.npy')[0]
adv8C7 = np.load('latent_z_adv8C_7.npy')[0]
adv8C8 = np.load('latent_z_adv8C_8.npy')[0]
adv8C9 = np.load('latent_z_adv8C_9.npy')[0]
adv8C10 = np.load('latent_z_adv8C_10.npy')[0]
adv8C11 = np.load('latent_z_adv8C_11.npy')[0]
adv8C12 = np.load('latent_z_adv8C_12.npy')[0]
adv8C13 = np.load('latent_z_adv8C_13.npy')[0]
adv8C14 = np.load('latent_z_adv8C_14.npy')[0]
adv8C15 = np.load('latent_z_adv8C_15.npy')[0]
adv8C16 = np.load('latent_z_adv8C_16.npy')[0]
adv8C17 = np.load('latent_z_adv8C_17.npy')[0]
adv8C18 = np.load('latent_z_adv8C_18.npy')[0]

adv8C19 = np.load('latent_z_adv8C_19.npy')[0]
adv8C20 = np.load('latent_z_adv8C_20.npy')[0]
adv8C21 = np.load('latent_z_adv8C_21.npy')[0]
adv8C22 = np.load('latent_z_adv8C_22.npy')[0]
adv8C23 = np.load('latent_z_adv8C_23.npy')[0]
adv8C24 = np.load('latent_z_adv8C_24.npy')[0]
adv8C25 = np.load('latent_z_adv8C_25.npy')[0]
adv8C26 = np.load('latent_z_adv8C_26.npy')[0]
adv8C27 = np.load('latent_z_adv8C_27.npy')[0]
adv8C28 = np.load('latent_z_adv8C_28.npy')[0]
adv8C29 = np.load('latent_z_adv8C_29.npy')[0]
adv8C30 = np.load('latent_z_adv8C_30.npy')[0]
adv8C31 = np.load('latent_z_adv8C_31.npy')[0]
adv8C32 = np.load('latent_z_adv8C_32.npy')[0]
adv8C33 = np.load('latent_z_adv8C_33.npy')[0]

img10= np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test10/inverted_codes.npy').reshape((7168))


print('norm   adv8C1: ' + str( np.linalg.norm(img8 - adv8C1) ) )
print('norm   adv8C2: ' + str( np.linalg.norm(img8 - adv8C2) ) )
print('norm   adv8C3: ' + str( np.linalg.norm(img8 - adv8C3) ) )
print('norm   adv8C4: ' + str( np.linalg.norm(img8 - adv8C4) ) )
print('norm   adv8C5: ' + str( np.linalg.norm(img8 - adv8C5) ) )
print('norm   adv8C6: ' + str( np.linalg.norm(img8 - adv8C6) ) )
print('norm   adv8C7: ' + str( np.linalg.norm(img8 - adv8C7) ) )
print('norm   adv8C8: ' + str( np.linalg.norm(img8 - adv8C8) ) )
print('norm   adv8C9: ' + str( np.linalg.norm(img8 - adv8C9) ) )
print('norm   adv8C10: ' + str( np.linalg.norm(img8 - adv8C10) ) )
print('norm   adv8C11: ' + str( np.linalg.norm(img8 - adv8C11) ) )
print('norm   adv8C12: ' + str( np.linalg.norm(img8 - adv8C12) ) )
print('norm   adv8C13: ' + str( np.linalg.norm(img8 - adv8C13) ) )
print('norm   adv8C14: ' + str( np.linalg.norm(img8 - adv8C14) ) )
print('norm   adv8C15: ' + str( np.linalg.norm(img8 - adv8C15) ) )
print('norm   adv8C16: ' + str( np.linalg.norm(img8 - adv8C16) ) )
print('norm   adv8C17: ' + str( np.linalg.norm(img8 - adv8C17) ) )
print('norm   adv8C18: ' + str( np.linalg.norm(img8 - adv8C18) ) )

print('norm   adv8C19: ' + str( np.linalg.norm(img8 - adv8C19) ) )
print('norm   adv8C20: ' + str( np.linalg.norm(img8 - adv8C20) ) )
print('norm   adv8C21: ' + str( np.linalg.norm(img8 - adv8C21) ) )
print('norm   adv8C22: ' + str( np.linalg.norm(img8 - adv8C22) ) )
print('norm   adv8C23: ' + str( np.linalg.norm(img8 - adv8C23) ) )
print('norm   adv8C24: ' + str( np.linalg.norm(img8 - adv8C24) ) )
print('norm   adv8C25: ' + str( np.linalg.norm(img8 - adv8C25) ) )
print('norm   adv8C26: ' + str( np.linalg.norm(img8 - adv8C26) ) )
print('norm   adv8C27: ' + str( np.linalg.norm(img8 - adv8C27) ) )
print('norm   adv8C28: ' + str( np.linalg.norm(img8 - adv8C28) ) )
print('norm   adv8C29: ' + str( np.linalg.norm(img8 - adv8C29) ) )
print('norm   adv8C30: ' + str( np.linalg.norm(img8 - adv8C30) ) )
print('norm   adv8C31: ' + str( np.linalg.norm(img8 - adv8C31) ) )
print('norm   adv8C32: ' + str( np.linalg.norm(img8 - adv8C32) ) )
print('norm   adv8C33: ' + str( np.linalg.norm(img8 - adv8C33) ) )


print('---')
print('dist   img8_img10: ' + str( np.linalg.norm(img8 - img10) ) )
print('dist   adv8C1_img10: ' + str( np.linalg.norm(adv8C1 - img10) ) )
print('dist   adv8C2_img10: ' + str( np.linalg.norm(adv8C2 - img10) ) )
print('dist   adv8C3_img10: ' + str( np.linalg.norm(adv8C3 - img10) ) )
print('dist   adv8C4_img10: ' + str( np.linalg.norm(adv8C4 - img10) ) )
print('dist   adv8C5_img10: ' + str( np.linalg.norm(adv8C5 - img10) ) )
print('dist   adv8C6_img10: ' + str( np.linalg.norm(adv8C6 - img10) ) )
print('dist   adv8C7_img10: ' + str( np.linalg.norm(adv8C7 - img10) ) )
print('dist   adv8C8_img10: ' + str( np.linalg.norm(adv8C8 - img10) ) )
print('dist   adv8C9_img10: ' + str( np.linalg.norm(adv8C9 - img10) ) )
print('dist   adv8C10_img10: ' + str( np.linalg.norm(adv8C10 - img10) ) )
print('dist   adv8C11_img10: ' + str( np.linalg.norm(adv8C11 - img10) ) )
print('dist   adv8C12_img10: ' + str( np.linalg.norm(adv8C12 - img10) ) )
print('dist   adv8C13_img10: ' + str( np.linalg.norm(adv8C13 - img10) ) )
print('dist   adv8C14_img10: ' + str( np.linalg.norm(adv8C14 - img10) ) )
print('dist   adv8C15_img10: ' + str( np.linalg.norm(adv8C15 - img10) ) )
print('dist   adv8C16_img10: ' + str( np.linalg.norm(adv8C16 - img10) ) )
print('dist   adv8C17_img10: ' + str( np.linalg.norm(adv8C17 - img10) ) )
print('dist   adv8C18_img10: ' + str( np.linalg.norm(adv8C18 - img10) ) )

print('dist   adv8C19_img10: ' + str( np.linalg.norm(adv8C19 - img10) ) )
print('dist   adv8C20_img10: ' + str( np.linalg.norm(adv8C20 - img10) ) )
print('dist   adv8C21_img10: ' + str( np.linalg.norm(adv8C21 - img10) ) )
print('dist   adv8C22_img10: ' + str( np.linalg.norm(adv8C22 - img10) ) )
print('dist   adv8C23_img10: ' + str( np.linalg.norm(adv8C23 - img10) ) )

print('dist   adv8C30_img10: ' + str( np.linalg.norm(adv8C30 - img10) ) )
print('dist   adv8C31_img10: ' + str( np.linalg.norm(adv8C31 - img10) ) )
print('dist   adv8C32_img10: ' + str( np.linalg.norm(adv8C32 - img10) ) )
print('dist   adv8C33_img10: ' + str( np.linalg.norm(adv8C33 - img10) ) )


print('---')
print('dist   adv8C1_adv8C2: ' + str( np.linalg.norm(adv8C1 - adv8C2) ) )
print('dist   adv8C1_adv8C3: ' + str( np.linalg.norm(adv8C1 - adv8C3) ) )
print('dist   adv8C1_adv8C4: ' + str( np.linalg.norm(adv8C1 - adv8C4) ) )
print('dist   adv8C1_adv8C5: ' + str( np.linalg.norm(adv8C1 - adv8C5) ) )
print('dist   adv8C1_adv8C6: ' + str( np.linalg.norm(adv8C1 - adv8C6) ) )
print('dist   adv8C1_adv8C7: ' + str( np.linalg.norm(adv8C1 - adv8C7) ) )
print('dist   adv8C1_adv8C8: ' + str( np.linalg.norm(adv8C1 - adv8C8) ) )
print('dist   adv8C1_adv8C9: ' + str( np.linalg.norm(adv8C1 - adv8C9) ) )
print('dist   adv8C1_adv8C10: ' + str( np.linalg.norm(adv8C1 - adv8C10) ) )
print('dist   adv8C1_adv8C11: ' + str( np.linalg.norm(adv8C1 - adv8C11) ) )
print('dist   adv8C1_adv8C12: ' + str( np.linalg.norm(adv8C1 - adv8C12) ) )
print('dist   adv8C1_adv8C13: ' + str( np.linalg.norm(adv8C1 - adv8C13) ) )
print('dist   adv8C1_adv8C14: ' + str( np.linalg.norm(adv8C1 - adv8C14) ) )
print('dist   adv8C1_adv8C15: ' + str( np.linalg.norm(adv8C1 - adv8C15) ) )

print('dist   adv8C2_adv8C3: ' + str( np.linalg.norm(adv8C2 - adv8C3) ) )
print('dist   adv8C2_adv8C4: ' + str( np.linalg.norm(adv8C2 - adv8C4) ) )
print('dist   adv8C2_adv8C5: ' + str( np.linalg.norm(adv8C2 - adv8C5) ) )
print('dist   adv8C2_adv8C6: ' + str( np.linalg.norm(adv8C2 - adv8C6) ) )
print('dist   adv8C2_adv8C7: ' + str( np.linalg.norm(adv8C2 - adv8C7) ) )
print('dist   adv8C2_adv8C8: ' + str( np.linalg.norm(adv8C2 - adv8C8) ) )
print('dist   adv8C2_adv8C9: ' + str( np.linalg.norm(adv8C2 - adv8C9) ) )
print('dist   adv8C2_adv8C10: ' + str( np.linalg.norm(adv8C2 - adv8C10) ) )


print('dist   adv8C3_adv8C4: ' + str( np.linalg.norm(adv8C3 - adv8C4) ) )
print('dist   adv8C3_adv8C5: ' + str( np.linalg.norm(adv8C3 - adv8C5) ) )
print('dist   adv8C3_adv8C6: ' + str( np.linalg.norm(adv8C3 - adv8C6) ) )
print('dist   adv8C3_adv8C7: ' + str( np.linalg.norm(adv8C3 - adv8C7) ) )
print('dist   adv8C3_adv8C8: ' + str( np.linalg.norm(adv8C3 - adv8C8) ) )
print('dist   adv8C3_adv8C9: ' + str( np.linalg.norm(adv8C3 - adv8C9) ) )
print('dist   adv8C3_adv8C10: ' + str( np.linalg.norm(adv8C3 - adv8C10) ) )

print('dist   adv8C4_adv8C5: ' + str( np.linalg.norm(adv8C4 - adv8C5) ) )
print('dist   adv8C4_adv8C6: ' + str( np.linalg.norm(adv8C4 - adv8C6) ) )
print('dist   adv8C4_adv8C7: ' + str( np.linalg.norm(adv8C4 - adv8C7) ) )
print('dist   adv8C4_adv8C8: ' + str( np.linalg.norm(adv8C4 - adv8C8) ) )
print('dist   adv8C4_adv8C9: ' + str( np.linalg.norm(adv8C4 - adv8C9) ) )
print('dist   adv8C4_adv8C10: ' + str( np.linalg.norm(adv8C4 - adv8C10) ) )

print('dist   adv8C5_adv8C6: ' + str( np.linalg.norm(adv8C5 - adv8C6) ) )
print('dist   adv8C5_adv8C7: ' + str( np.linalg.norm(adv8C5 - adv8C7) ) )
print('dist   adv8C5_adv8C8: ' + str( np.linalg.norm(adv8C5 - adv8C8) ) )
print('dist   adv8C5_adv8C9: ' + str( np.linalg.norm(adv8C5 - adv8C9) ) )
print('dist   adv8C5_adv8C10: ' + str( np.linalg.norm(adv8C5 - adv8C10) ) )

print('dist   adv8C6_adv8C7: ' + str( np.linalg.norm(adv8C6 - adv8C7) ) )
print('dist   adv8C6_adv8C8: ' + str( np.linalg.norm(adv8C6 - adv8C8) ) )
print('dist   adv8C6_adv8C9: ' + str( np.linalg.norm(adv8C6 - adv8C9) ) )
print('dist   adv8C6_adv8C10: ' + str( np.linalg.norm(adv8C6 - adv8C10) ) )

print('dist   adv8C7_adv8C8: ' + str( np.linalg.norm(adv8C7 - adv8C8) ) )
print('dist   adv8C7_adv8C9: ' + str( np.linalg.norm(adv8C7 - adv8C9) ) )
print('dist   adv8C7_adv8C10: ' + str( np.linalg.norm(adv8C7 - adv8C10) ) )

print('dist   adv8C8_adv8C9: ' + str( np.linalg.norm(adv8C8 - adv8C9) ) )
print('dist   adv8C8_adv8C10: ' + str( np.linalg.norm(adv8C8 - adv8C10) ) )

print('dist   adv8C9_adv8C10: ' + str( np.linalg.norm(adv8C9 - adv8C10) ) )


print('dist   adv8C10_adv8C11: ' + str( np.linalg.norm(adv8C10 - adv8C11) ) )
print('dist   adv8C10_adv8C12: ' + str( np.linalg.norm(adv8C10 - adv8C12) ) )
print('dist   adv8C10_adv8C13: ' + str( np.linalg.norm(adv8C10 - adv8C13) ) )
print('dist   adv8C10_adv8C14: ' + str( np.linalg.norm(adv8C10 - adv8C14) ) )
print('dist   adv8C10_adv8C15: ' + str( np.linalg.norm(adv8C10 - adv8C15) ) )



print('dist   adv8C30_adv8C31: ' + str( np.linalg.norm(adv8C30 - adv8C31) ) )
print('dist   adv8C30_adv8C32: ' + str( np.linalg.norm(adv8C30 - adv8C32) ) )
print('dist   adv8C31_adv8C32: ' + str( np.linalg.norm(adv8C31 - adv8C32) ) )

