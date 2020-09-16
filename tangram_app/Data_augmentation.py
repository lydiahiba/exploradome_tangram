import os
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np   
import scipy
from scipy.ndimage import rotate
from skimage.util import random_noise
from skimage import data, img_as_float
from skimage import exposure
from PIL import Image, ImageOps


def gaussian_noise(img, mean=0, sigma=0.03):
    img = img.copy()
    noise = np.random.normal(mean, sigma, img.shape)
    mask_overflow_upper = img+noise >= 1.0
    mask_overflow_lower = img+noise < 0.0
    noise[mask_overflow_upper] = 1.0
    noise[mask_overflow_lower] = 0.0
    img += noise # does not work
    return img

def cutout(image_origin, mask_size, mask_value='mean'):
    image = np.copy(image_origin)
    if mask_value == 'mean':
        mask_value = image.mean()
    elif mask_value == 'random':
        mask_value = np.random.randint(0, 256)
        
def rotate_img(img, angle, bg_patch=(5,5)):
    assert len(img.shape) <= 3, "Incorrect image shape"
    rgb = len(img.shape) == 3
    if rgb:
        
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0,1)) 
    else:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])
    img = img.copy()
    img = rotate(img, angle, reshape=False)
    mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]
    img[mask] = bg_color
    return img

def noisy(img):
    # Add salt-and-pepper noise to the image.
    noise_img = random_noise(img, mode='s&p',amount=0.15)
    # The above function returns a floating-point image
    # on the range [0, 1], thus we changed it to 'uint8'
    # and from [0,255]
    noise_img = np.array(255*noise_img, dtype = 'uint8')
    return noise_img

def change_contrast(img, factor):
    def contrast(pixel):
        return 128 + factor * (pixel - 128)
    return img.point(contrast)

def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    '''
    p : the probability that random erasing is performed
    s_l, s_h : minimum / maximum proportion of erased area against input image
    r_1, r_2 : minimum / maximum aspect ratio of erased area
    v_l, v_h : minimum / maximum value for erased area
    pixel_level : pixel-level randomization for erased area
    from github: yu4u/cutout-random-erasing
    '''
    def eraser(input_img):
#         img_h, img_w, img_c = input_img.shape # color
        img_h, img_w = input_img.shape # B & W
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
#             c = np.random.uniform(v_l, v_h, (h, w, img_c)) # color
            c = np.random.uniform(v_l, v_h, (h, w)) # BW
        else:
            c = np.random.uniform(v_l, v_h)

#         input_img[top:top + h, left:left + w, :] = c # color
        input_img[top:top + h, left:left + w] = c # BW

        return input_img

    return eraser

def equal(img):
    equ = cv2.equalizeHist(img)
    return equ

# FONCTION POUR SAUVEGARDER LES IMAGES
def action(count, img, dest, action="show"):
    count += 1
    imge = img.copy()
    '''
    count: counter to name images afterwards
    img: np.array image
    dest: path to save if action == "save"
    action: if "show" only display image, if "save" create a file and save image
    returns counter + 1 for naming images
    '''
    if action == "show":
        plt.imshow(imge, cmap='gray')
        plt.show()         
    else :
        cv2.imwrite(dest, imge)
        print(count)
        cv2.imwrite(dest, imge)        
    return count





img = None
count = 0
for (root,dirs,files) in os.walk('path_to_origin_folder_with_source_images_in_folders_by_class', topdown=True):
    
    # load random eraser ("poke")
    eraser = get_random_eraser(p=1, s_l=0.08, s_h=0.1, r_1=0.3, r_2=1/0.3,
                  v_l=0, v_h=255, pixel_level=False)

    for ele in files[:10]: # 10 de chaque classe
        
        path = os.path.join(root, ele) # source
        label = [ele for ele in Labels if ele in path] # label
        
        # Global action after each filter : "save" or "show" the images
        global_action = "save"
        
        print(path, label)
        img = cv2.imread(path) # original shape : (2160, 1920, 3)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # GREYSCALE
        img = np.array(img) 
        
        # Détecte si c'est le secteur gauche ou droit par la valeur des pixels 
        # (clairs:table, sombres:dehors)
        # découpe (crop) la zone d'intérêt
        if np.mean(img[:10,:10]) > 100 : 
            img = img[300:1700,:1400] # crop
        else:
            img = img[300:1700,250:1650]
        
        # Resize à la taille d'entrée dans le modèle
        img = cv2.resize(img, (250,250))
    
        # ROTATION
        for ang in np.linspace(0,180, 6): 
            
            img = rotate_img(img, ang, bg_patch=(10,10))            
            count = action(count, img.copy(), f'target_folder/{label[0]}/{label[0]}_{count}.jpg', action=global_action)
            
            pok = eraser(img.copy())
            count = action(count, pok, f'C:/Users/utilisateur/Dropbox/SIMPLON/eploradom/sda/{label[0]}/{label[0]}_{count}.jpg', action=global_action)

            eq = equal(img)            
            count = action(count, eq, f'C:/Users/utilisateur/Dropbox/SIMPLON/eploradom/sda/{label[0]}/{label[0]}_{count}.jpg', action=global_action)

            pok = eraser(eq.copy()) 
            count = action(count, pok, f'C:/Users/utilisateur/Dropbox/SIMPLON/eploradom/sda/{label[0]}/{label[0]}_{count}.jpg', action=global_action)            

            # Salt & pepper noise
            noisy_image = noisy(img)
            count = action(count, noisy_image, f'C:/Users/utilisateur/Dropbox/SIMPLON/eploradom/sda/{label[0]}/{label[0]}_{count}.jpg', action=global_action)
  
            pok = eraser(noisy_image) 
            count = action(count, pok, f'C:/Users/utilisateur/Dropbox/SIMPLON/eploradom/sda/{label[0]}/{label[0]}_{count}.jpg', action=global_action)
 
            eq = equal(noisy_image)  
            count = action(count, eq, f'C:/Users/utilisateur/Dropbox/SIMPLON/eploradom/sda/{label[0]}/{label[0]}_{count}.jpg', action=global_action)

            pok = eraser(eq.copy()) 
            count = action(count, pok, f'C:/Users/utilisateur/Dropbox/SIMPLON/eploradom/sda/{label[0]}/{label[0]}_{count}.jpg', action=global_action)
    
            # negative effect
            invert = ImageOps.invert(Image.fromarray(img.copy()))
            count = action(count, np.array(invert), f'C:/Users/utilisateur/Dropbox/SIMPLON/eploradom/sda/{label[0]}/{label[0]}_{count}.jpg', action=global_action)

            pok = eraser(np.array(invert).copy()) 
            count = action(count, pok, f'C:/Users/utilisateur/Dropbox/SIMPLON/eploradom/sda/{label[0]}/{label[0]}_{count}.jpg', action=global_action)

            eq = equal(np.array(invert))
            count = action(count, eq, f'C:/Users/utilisateur/Dropbox/SIMPLON/eploradom/sda/{label[0]}/{label[0]}_{count}.jpg', action=global_action)

            pok = eraser(eq.copy()) 
            count = action(count, pok, f'C:/Users/utilisateur/Dropbox/SIMPLON/eploradom/sda/{label[0]}/{label[0]}_{count}.jpg', action=global_action)

            # negative effect sur s&p noise image
            noisy_invert = noisy(np.array(invert))
            count = action(count, noisy_invert, f'C:/Users/utilisateur/Dropbox/SIMPLON/eploradom/sda/{label[0]}/{label[0]}_{count}.jpg', action=global_action)

            pok = eraser(noisy_invert.copy())  
            count = action(count, pok, f'C:/Users/utilisateur/Dropbox/SIMPLON/eploradom/sda/{label[0]}/{label[0]}_{count}.jpg', action=global_action)

            eq = equal(noisy_invert) 
            count = action(count, eq, f'C:/Users/utilisateur/Dropbox/SIMPLON/eploradom/sda/{label[0]}/{label[0]}_{count}.jpg', action=global_action)

            pok = eraser(eq.copy()) 
            count = action(count, pok, f'C:/Users/utilisateur/Dropbox/SIMPLON/eploradom/sda/{label[0]}/{label[0]}_{count}.jpg', action=global_action)                  