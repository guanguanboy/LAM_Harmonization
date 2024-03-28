import torch, cv2, os, sys, numpy as np, matplotlib.pyplot as plt
from PIL import Image

from ModelZoo.utils import load_as_tensor, Tensor2PIL, PIL2Tensor, _add_batch_one
from ModelZoo import get_model, load_model, print_network
from SaliencyModel.utils import vis_saliency, vis_saliency_kde, click_select_position, grad_abs_norm, grad_norm, prepare_images, make_pil_grid, blend_input
from SaliencyModel.utils import cv2_to_pil, pil_to_cv2, gini
from SaliencyModel.attributes import attr_grad
from SaliencyModel.BackProp import I_gradient, attribution_objective, Path_gradient
from SaliencyModel.BackProp import saliency_map_PG as saliency_map
from SaliencyModel.BackProp import GaussianBlurPath
from SaliencyModel.utils import grad_norm, IG_baseline, interpolation, isotropic_gaussian_kernel

#ESRGAN
#RCAN
#RNAN
#SAN
#CARN
model_name = 'SPANET'
model = load_model(model_name + '@Base')  # You can Change the model name to load different models
#model = load_model('RCAN@Base')  # You can Change the model name to load different models
window_size = 16  # Define windoes_size of D

"""
fig_name = 'a0040.jpg'
fig_path = './test_images/harmonization/' + fig_name

lr_name = 'a0040_1_2.jpg'
lr_path = './test_images/harmonization/' + lr_name

mask_name = 'a0040_1.png'
mask_path = './test_images/harmonization/' + mask_name

fig_name = 'a0038.jpg'
fig_path = './test_images/harmonization/' + fig_name

lr_name = 'a0038_1_2.jpg'
lr_path = './test_images/harmonization/' + lr_name

mask_name = 'a0038_1.png'
mask_path = './test_images/harmonization/' + mask_name
"""

fig_name = 'a1035.jpg'
fig_path = './test_images/harmonization/' + fig_name

lr_name = 'a1035_1_5.jpg'
lr_path = './test_images/harmonization/' + lr_name

mask_name = 'a1035_1.png'
mask_path = './test_images/harmonization/' + mask_name

img_hr = Image.open(fig_path)
img_lr = Image.open(lr_path)
img_mask = Image.open(mask_path).convert('L') 




#img_lr, img_hr = prepare_images(fig_path)  # Change this image name
#img_lr, img_hr = prepare_images('./test_images/1.png')  # Change this image name

tensor_lr = PIL2Tensor(img_lr)[:3] ; tensor_hr = PIL2Tensor(img_hr)[:3]
tensor_mask = PIL2Tensor(img_lr)[:1]

cv2_lr = np.moveaxis(tensor_lr.numpy(), 0, 2) ; cv2_hr = np.moveaxis(tensor_hr.numpy(), 0, 2)

plt.figure(1)
plt.imshow(cv2_hr)

w = 90  # The x coordinate of your select patch, 125 as an example
h = 100  # The y coordinate of your select patch, 160 as an example
         # And check the red box
         # Is your selected patch this one? If not, adjust the `w` and `h`.

tensor_lr = torch.cat([tensor_lr, tensor_mask], dim=0)

draw_img = pil_to_cv2(img_hr)
cv2.rectangle(draw_img, (w, h), (w + window_size, h + window_size), (0, 0, 255), 2)
position_pil = cv2_to_pil(draw_img)

plt.figure(2)
plt.imshow(position_pil)

sigma = 1.2 ; fold = 50 ; l = 9 ; alpha = 0.5
attr_objective = attribution_objective(attr_grad, h, w, window=window_size)
gaus_blur_path_func = GaussianBlurPath(sigma, fold, l)
interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(tensor_lr.numpy(), model, attr_objective, gaus_blur_path_func, cuda=True)
grad_numpy, result = saliency_map(interpolated_grad_numpy[:,:3,:,:], result_numpy)
abs_normed_grad_numpy = grad_abs_norm(grad_numpy)
saliency_image_abs = vis_saliency(abs_normed_grad_numpy,zoomin=1)

"""
saliency_image_kde = vis_saliency_kde(abs_normed_grad_numpy,zoomin=1)
blend_abs_and_input = cv2_to_pil(pil_to_cv2(saliency_image_abs) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
blend_kde_and_input = cv2_to_pil(pil_to_cv2(saliency_image_kde) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
pil = make_pil_grid(
    [position_pil,
     saliency_image_abs,
     blend_abs_and_input,
     blend_kde_and_input,
     Tensor2PIL(torch.clamp(result, min=0., max=1.))]
)
"""
saliency_save_path = './test_images/harmonization/output/' + model_name + fig_name
saliency_image_abs.save(saliency_save_path)
#plt.figure(3)
#plt.imshow(pil)
gini_index = gini(abs_normed_grad_numpy)
diffusion_index = (1 - gini_index) * 100
print(f"The DI of this case is {diffusion_index}")