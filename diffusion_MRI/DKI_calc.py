# 此程序用于计算kurtosis的相关参数
from dipy.reconst.dti import planarity
import numpy as np
import matplotlib.pyplot as plt
import dipy.reconst.dki as dki
import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.segment.mask import median_otsu
from numpy.core.fromnumeric import amax
from scipy.ndimage.filters import gaussian_filter
import scipy.signal as signal
import os


dataFolder = 'F:\lab\HIE\ROI_analysis_atlas\ROI_analysis_atlas_diffeomap'
for file in os.listdir(dataFolder):
    fdwi = os.path.join(dataFolder, file, 'DKI')
    
    data, affine, img = load_nifti(os.path.join(fdwi, 'dwi.nii.gz'), return_img=True)
    bvals, bvecs = read_bvals_bvecs(os.path.join(fdwi, 'bvals.bval'), os.path.join(fdwi, 'bvecs.bvec'))
    gtab = gradient_table(bvals, bvecs)
    maskdata, mask = median_otsu(data, vol_idx=[0, 1], median_radius=4, numpass=2, autocrop=False, dilate=1)

    '''
    #高斯滤波
    fwhm = 1.25
    gauss_std = fwhm / np.sqrt(8 * np.log(2))  # converting fwhm to Gaussian std
    data_smooth = np.zeros(data.shape)
    for v in range(data.shape[-1]):
        data_smooth[..., v] = gaussian_filter(data[..., v], sigma=gauss_std)
    '''
    data_smooth = data

    # 奇偶层之间取均值平滑
    for v in range(data.shape[-1]):
        for u in range(1, data.shape[-2]-1):
            data_smooth[:,:,u,v] = (data_smooth[:,:,u,v]+data_smooth[:,:,u+1,v]+data_smooth[:,:,u-1,v]) / 3


    dkimodel = dki.DiffusionKurtosisModel(gtab)
    dkifit = dkimodel.fit(data_smooth, mask=mask)

    #FA = dkifit.fa
    #MD = dkifit.md
    #AD = dkifit.ad
    #RD = dkifit.rd
    MK = dkifit.mk(0, 3)
    #AK = dkifit.ak(0, 3)
    #RK = dkifit.rk(0, 3)
    #MKT = dkifit.mkt(0, 3)
    #KFA = dkifit.kfa


    save_nifti(os.path.join(fdwi, 'MK.nii'), MK, affine)
    # save_nifti(os.path.join(fdwi, 'AK.nii'), AK, affine)
    # save_nifti(os.path.join(fdwi, 'RK.nii'), RK, affine)
    # save_nifti(os.path.join(fdwi, 'MKT.nii'), MKT, affine)
    # save_nifti(os.path.join(fdwi, 'KFA.nii'), KFA, affine)



'''
# 中值滤波试一下
# 7.30试了一下，不太行，很糊
MK_slice = signal.medfilt(MK[:,:,axial_slice], kernel_size=3)
AK_slice = signal.medfilt(AK[:,:,axial_slice], kernel_size=3)
RK_slice = signal.medfilt(RK[:,:,axial_slice], kernel_size=3)
MKT_slice = signal.medfilt(MKT[:,:,axial_slice], kernel_size=3)
KFA_slice = signal.medfilt(KFA[:,:,axial_slice], kernel_size=3)
fig2, ax = plt.subplots(1, 5, figsize=(12, 6), subplot_kw={'xticks': [], 'yticks': []})
fig2.subplots_adjust(hspace=0.3, wspace=0.05)
ax.flat[0].imshow(MK_slice.T, cmap='gray', vmin=0, vmax=1.5, origin='lower')
ax.flat[0].set_title('MK(blur)')
ax.flat[1].imshow(AK_slice.T, cmap='gray', vmin=0, vmax=1.5, origin='lower')
ax.flat[1].set_title('AK(blur)')
ax.flat[2].imshow(RK_slice.T, cmap='gray', vmin=0, vmax=1.5, origin='lower')
ax.flat[2].set_title('RK(blur)')
ax.flat[3].imshow(MKT_slice.T, cmap='gray', vmin=0, vmax=1.5, origin='lower')
ax.flat[3].set_title('MKT(blur)')
ax.flat[4].imshow(KFA_slice.T, cmap='gray', vmin=0, vmax=1.5, origin='lower')
ax.flat[4].set_title('KFA(blur)')
plt.show()
'''

# 以下均为图像展示的程序
#axial_slice = 41
#sagittal_slice = 70
#coronal_slice = 70
'''
fig0, ax = plt.subplots(1, 4, figsize=(12, 6), subplot_kw={'xticks': [], 'yticks': []})
fig0.subplots_adjust(hspace=0.3, wspace=0.05)
ax.flat[0].imshow(maskdata[:, :, axial_slice, 0].T, cmap='gray', vmin=0, vmax=0.7, origin='lower')
ax.flat[0].set_title('maskdata')
ax.flat[1].imshow(mask[:, :, axial_slice].T, cmap='gray', vmin=0, vmax=2.0e-3, origin='lower')
ax.flat[1].set_title('mask')
data_mask = maskdata * data
ax.flat[2].imshow(data_mask[:, :, axial_slice, 0].T, cmap='gray', origin='lower')
ax.flat[2].set_title('mask*data')
ax.flat[3].imshow(data[:, :, axial_slice, 0].T, cmap='gray', origin='lower')
ax.flat[3].set_title('data')
plt.show()
'''

'''
fig1, ax = plt.subplots(1, 4, figsize=(12, 6), subplot_kw={'xticks': [], 'yticks': []})
fig1.subplots_adjust(hspace=0.3, wspace=0.05)
ax.flat[0].imshow(FA[:, :, axial_slice].T, cmap='gray', vmin=0, vmax=0.7, origin='lower')
ax.flat[0].set_title('FA (DKI)')
ax.flat[1].imshow(MD[:, :, axial_slice].T, cmap='gray', vmin=0, vmax=2.0e-3, origin='lower')
ax.flat[1].set_title('MD (DKI)')
ax.flat[2].imshow(AD[:, :, axial_slice].T, cmap='gray', vmin=0, vmax=2.0e-3, origin='lower')
ax.flat[2].set_title('AD (DKI)')
ax.flat[3].imshow(RD[:, :, axial_slice].T, cmap='gray', vmin=0, vmax=2.0e-3, origin='lower')
ax.flat[3].set_title('RD (DKI)')
plt.show()
'''
'''
fig2, ax = plt.subplots(3, 3, figsize=(12, 6), subplot_kw={'xticks': [], 'yticks': []})
fig2.subplots_adjust(hspace=0.3, wspace=0.05)
ax.flat[0].imshow(MK[:, :, axial_slice].T, cmap='gray', vmin=0, vmax=1.5, origin='lower')
ax.flat[0].set_title('MK(axial)')
ax.flat[1].imshow(AK[:, :, axial_slice].T, cmap='gray', vmin=0, vmax=1.5, origin='lower')
ax.flat[1].set_title('AK(axial)')
ax.flat[2].imshow(RK[:, :, axial_slice].T, cmap='gray', vmin=0, vmax=1.5, origin='lower')
ax.flat[2].set_title('RK(axial)')
ax.flat[3].imshow(MK[:, sagittal_slice, :].T, cmap='gray', vmin=0, vmax=1.5, origin='lower')
ax.flat[3].set_title('MK(sagittal)')
ax.flat[4].imshow(AK[:, sagittal_slice, :].T, cmap='gray', vmin=0, vmax=1.5, origin='lower')
ax.flat[4].set_title('AK(sagittal)')
ax.flat[5].imshow(RK[:, sagittal_slice, :].T, cmap='gray', vmin=0, vmax=1.5, origin='lower')
ax.flat[5].set_title('RK(sagittal)')
ax.flat[6].imshow(MK[coronal_slice, :, :].T, cmap='gray', vmin=0, vmax=1.5, origin='lower')
ax.flat[6].set_title('MK(coronal)')
ax.flat[7].imshow(AK[coronal_slice, :, :].T, cmap='gray', vmin=0, vmax=1.5, origin='lower')
ax.flat[7].set_title('AK(coronal)')
ax.flat[8].imshow(RK[coronal_slice, :, :].T, cmap='gray', vmin=0, vmax=1.5, origin='lower')
ax.flat[8].set_title('RK(coronal)')
plt.show()

fig3, ax = plt.subplots(3, 2, figsize=(10, 6), subplot_kw={'xticks': [], 'yticks': []})
fig3.subplots_adjust(hspace=0.3, wspace=0.05)
ax.flat[0].imshow(MKT[:, :, axial_slice].T, cmap='gray', vmin=0, vmax=1.5, origin='lower')
ax.flat[0].set_title('MKT(axial)')
ax.flat[1].imshow(KFA[:, :, axial_slice].T, cmap='gray', vmin=0, vmax=1, origin='lower')
ax.flat[1].set_title('KFA(axial)')
ax.flat[2].imshow(MKT[:, sagittal_slice, :].T, cmap='gray', vmin=0, vmax=1.5, origin='lower')
ax.flat[2].set_title('MKT(sagittal)')
ax.flat[3].imshow(KFA[:, sagittal_slice, :].T, cmap='gray', vmin=0, vmax=1, origin='lower')
ax.flat[3].set_title('KFA(sagittal)')
ax.flat[4].imshow(MKT[coronal_slice, :, :].T, cmap='gray', vmin=0, vmax=1.5, origin='lower')
ax.flat[4].set_title('MKT(coronal)')
ax.flat[5].imshow(KFA[coronal_slice, :, :].T, cmap='gray', vmin=0, vmax=1, origin='lower')
ax.flat[5].set_title('KFA(coronal)')
plt.show()
'''
