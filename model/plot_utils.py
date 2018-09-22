import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave
from scipy.misc import imresize
import matplotlib.patches as mpatches

class Plot_Reproduce_Performance():
    def __init__(self, DIR, n_img_x=8, n_img_y=8, img_w=28, img_h=28, resize_factor=1.0):
        self.DIR = DIR

        assert n_img_x > 0 and n_img_y > 0

        self.n_img_x = n_img_x
        self.n_img_y = n_img_y
        self.n_tot_imgs = n_img_x * n_img_y

        assert img_w > 0 and img_h > 0

        self.img_w = img_w
        self.img_h = img_h

        assert resize_factor > 0

        self.resize_factor = resize_factor

    def save_images(self, images, name='result.jpg'):
        images = images.reshape(self.n_img_x*self.n_img_y, self.img_h, self.img_w)
        imsave(self.DIR + "/"+name, self._merge(images, [self.n_img_y, self.n_img_x]))

    def _merge(self, images, size):
        h, w = images.shape[1], images.shape[2]

        h_ = int(h * self.resize_factor)
        w_ = int(w * self.resize_factor)

        img = np.zeros((h_ * size[0], w_ * size[1]))

        for idx, image in enumerate(images):
            i = int(idx % size[1])
            j = int(idx / size[1])

            image_ = imresize(image, size=(w_,h_), interp='bicubic')

            img[j*h_:j*h_+h_, i*w_:i*w_+w_] = image_

        return img

class Plot_Manifold_Learning_Result():
    def __init__(self, DIR, n_img_x=20, n_img_y=20, img_w=28, img_h=28, resize_factor=1.0, z_range=4):
        self.DIR = DIR

        assert n_img_x > 0 and n_img_y > 0

        self.n_img_x = n_img_x
        self.n_img_y = n_img_y
        self.n_tot_imgs = n_img_x * n_img_y

        assert img_w > 0 and img_h > 0

        self.img_w = img_w
        self.img_h = img_h

        assert resize_factor > 0

        self.resize_factor = resize_factor

        assert z_range > 0
        self.z_range = z_range

        self._set_latent_vectors()

    def _set_latent_vectors(self):

        # z1 = np.linspace(-self.z_range, self.z_range, self.n_img_y)
        # z2 = np.linspace(-self.z_range, self.z_range, self.n_img_x)
        #
        # z = np.array(np.meshgrid(z1, z2))
        # z = z.reshape([-1, 2])

        # borrowed from https://github.com/fastforwardlabs/vae-tf/blob/master/plot.py
        z = np.rollaxis(np.mgrid[self.z_range:-self.z_range:self.n_img_y * 1j, self.z_range:-self.z_range:self.n_img_x * 1j], 0, 3)
        # z1 = np.rollaxis(np.mgrid[1:-1:self.n_img_y * 1j, 1:-1:self.n_img_x * 1j], 0, 3)
        # z = z1**2
        # z[z1<0] *= -1
        #
        # z = z*self.z_range

        self.z = z.reshape([-1, 2])

    def save_images(self, images, name='result.jpg'):
        images = images.reshape(self.n_img_x*self.n_img_y, self.img_h, self.img_w)
        imsave(self.DIR + "/"+name, self._merge(images, [self.n_img_y, self.n_img_x]))

    def _merge(self, images, size):
        h, w = images.shape[1], images.shape[2]

        h_ = int(h * self.resize_factor)
        w_ = int(w * self.resize_factor)

        img = np.zeros((h_ * size[0], w_ * size[1]))

        for idx, image in enumerate(images):
            i = int(idx % size[1])
            j = int(idx / size[1])

            image_ = imresize(image, size=(w_, h_), interp='bicubic')

            img[j * h_:j * h_ + h_, i * w_:i * w_ + w_] = image_

        return img

    # borrowed from https://github.com/ykwon0407/variational_autoencoder/blob/master/variational_bayes.ipynb
    def save_scattered_image(self, z, id, name='scattered_image.jpg'):
        N = 4
        plt.figure(figsize=(8, 7))
        plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), s=10, marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
        patch1 = mpatches.Patch(color='#FF0000', label='Ang')
        patch2 = mpatches.Patch(color='#0000FF', label='Hap')
        patch3 = mpatches.Patch(color='#009900', label='Neu')
        patch4 = mpatches.Patch(color='#9900CC', label='Neu')
        plt.legend(handles=[patch1,patch2,patch3,patch4])
        axes = plt.gca()
        axes.set_xlim([-5, 5])
        axes.set_ylim([-5, 5])
        plt.grid(False)
        plt.savefig(self.DIR + "/" + name)

# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 10, N))
    cmap_name = base.name + str(N)
    color_list=['#FF0000','#0000FF','#009900','#9900CC']
    return base.from_list(cmap_name, color_list, N)