import sys
import os
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import rotate
sns.set(color_codes=True)


new_dirname = sys.argv[1]
data_dir = sys.argv[2]

try:
    os.mkdir(new_dirname)
except OSError:
    print ("Creation of the directory %s failed" % new_dirname)
else:
    print ("Successfully created the directory %s " % new_dirname)


def show_img(img, ax):
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img)

    
def plot_grid(imgs, nrows, ncols, figsize=(10, 10)):
    assert len(imgs) == nrows*ncols, "Number of images should be {nrows}x{ncols}"
    _, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        show_img(img, ax)


def translate(img, shift=10, direction='right', roll=True):
    assert direction in ['right', 'left', 'down', 'up'], 'Directions should be top|up|left|right'
    img = img.copy()
    if direction == 'right':
        right_slice = img[:, -shift:].copy()
        img[:, shift:] = img[:, :-shift]
        if roll:
            img[:,:shift] = np.fliplr(right_slice)
    if direction == 'left':
        left_slice = img[:, :shift].copy()
        img[:, :-shift] = img[:, shift:]
        if roll:
            img[:, -shift:] = left_slice
    if direction == 'down':
        down_slice = img[-shift:, :].copy()
        img[shift:, :] = img[:-shift,:]
        if roll:
            img[:shift, :] = down_slice
    if direction == 'up':
        upper_slice = img[:shift, :].copy()
        img[:-shift, :] = img[shift:, :]
        if roll:
            img[-shift:,:] = upper_slice
    return img


def random_crop(img, crop_size=(10, 10)):
    assert crop_size[0] <= img.shape[0] and crop_size[1] <= img.shape[1], "Crop size should be less than image size"
    img = img.copy()
    w, h = img.shape[:2]
    x, y = np.random.randint(h-crop_size[0]), np.random.randint(w-crop_size[1])
    img = img[y:y+crop_size[0], x:x+crop_size[1]]
    return img

def rotate_img(img, angle, bg_patch=(5,5)):
    assert len(img.shape) <= 3, "Incorrect image shape"
    rgb = len(img.shape) == 3
    if rgb:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0,1))
    else:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])
    img = rotate(img, angle, reshape=False)
    mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]
    img[mask] = bg_color
    return img

def gaussian_noise(img, mean=0, sigma=0.03):
    img = img.copy()
    noise = np.random.normal(mean, sigma, img.shape)
    mask_overflow_upper = img+noise >= 1.0
    mask_overflow_lower = img+noise < 0
    noise[mask_overflow_upper] = 1.0
    noise[mask_overflow_lower] = 0
    np.add(img, noise, out=img, casting="unsafe")
    return img


def change_channel_ratio(img, channel='r', ratio=0.5):
    assert channel in 'rgb', "Value for channel: r|g|b"
    img = img.copy()
    ci = 'rgb'.index(channel)
    np.multiply(img[:, :, ci], ratio, out=img[:, :, ci], casting="unsafe") 
    return img

os.mkdir(new_dirname + '/' + 'translate')
os.mkdir(new_dirname + '/' + 'random_crop')
os.mkdir(new_dirname + '/' + 'rotate_img')
os.mkdir(new_dirname + '/' + 'gaussian_noise')
os.mkdir(new_dirname + '/' + 'change_channel_ratio')

for filename in os.listdir(data_dir):
    print(filename)
    img = np.array(plt.imread(data_dir + '/' + filename))
    sep = '.'
    filename = filename.split(sep, 1)[0]
    
    #translate
    t_img = translate(img, direction='up', shift=50)
    matplotlib.image.imsave(new_dirname + '/' + 'translate/' + filename + '_1' + '.png', t_img)

    t_img = translate(img, direction='down', shift=100)
    matplotlib.image.imsave(new_dirname + '/' + 'translate/' + filename + '_2' + '.png', t_img)
    
    t_img = translate(img, direction='left', shift=150)
    matplotlib.image.imsave(new_dirname + '/' + 'translate/' + filename + '_3' + '.png', t_img)

    t_img = translate(img, direction='right', shift=200)
    matplotlib.image.imsave(new_dirname + '/' + 'translate/' + filename + '_4' + '.png', t_img)
    
    #random_crop
    c_img = random_crop(img, crop_size = (3000,3000))
    matplotlib.image.imsave(new_dirname + '/' + 'random_crop/' + filename + '_1' + '.png', c_img)

    c_img = random_crop(img, crop_size = (3000,3000))
    matplotlib.image.imsave(new_dirname + '/' + 'random_crop/' + filename + '_2' + '.png', c_img)
 
    c_img = random_crop(img, crop_size = (3000,3000))
    matplotlib.image.imsave(new_dirname + '/' + 'random_crop/' + filename + '_3' + '.png', c_img)
    
    c_img = random_crop(img, crop_size = (3000,3000))
    matplotlib.image.imsave(new_dirname + '/' + 'random_crop/' + filename + '_4' + '.png', c_img)
    
    #rotate_img
    r_img = rotate_img(img, 5)
    matplotlib.image.imsave(new_dirname + '/' + 'rotate_img/' + filename + '_1' + '.png', r_img)

    r_img = rotate_img(img, 25)
    matplotlib.image.imsave(new_dirname + '/' + 'rotate_img/' + filename + '_2' + '.png', r_img)

    r_img = rotate_img(img, 45)
    matplotlib.image.imsave(new_dirname + '/' + 'rotate_img/' + filename + '_3' + '.png', r_img)

    r_img = rotate_img(img, 60)
    matplotlib.image.imsave(new_dirname + '/' + 'rotate_img/' + filename + '_4' + '.png', r_img)  

    #gaussian_noise
    g_img = gaussian_noise(img, mean = 0, sigma=0.03)
    matplotlib.image.imsave(new_dirname + '/' + 'gaussian_noise/' + filename + '_1' + '.png', g_img)

    g_img = gaussian_noise(img, mean = 3, sigma=1.0)
    matplotlib.image.imsave(new_dirname + '/' + 'gaussian_noise/' + filename + '_2' + '.png', g_img) 

    g_img = gaussian_noise(img, mean = -3, sigma=3.0)
    matplotlib.image.imsave(new_dirname + '/' + 'gaussian_noise/' + filename + '_3' + '.png', g_img)

    g_img = gaussian_noise(img, mean = 10, sigma=5.0)
    matplotlib.image.imsave(new_dirname + '/' + 'gaussian_noise/' + filename + '_4' + '.png', g_img)

    #change_channel_ratio
    cc_img = change_channel_ratio(img, ratio=0.3)
    matplotlib.image.imsave(new_dirname + '/' + 'change_channel_ratio/' + filename + '_1' + '.png', cc_img)

    cc_img = change_channel_ratio(img, ratio=0.6) 
    matplotlib.image.imsave(new_dirname + '/' + 'change_channel_ratio/' + filename + '_2' + '.png', cc_img)

    cc_img = change_channel_ratio(img, ratio=0.9)
    matplotlib.image.imsave(new_dirname + '/' + 'change_channel_ratio/' + filename + '_3' + '.png', cc_img)

