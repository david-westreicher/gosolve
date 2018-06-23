import visdom
import numpy as np

class Vis:
    def __init__(self, unnorm):
        self.vis = visdom.Visdom()
        self.unnorm = unnorm
        self.window = None

    def showimg(self, img, unnorm=True):
        if unnorm:
            return self.vis.image(self.unnorm(img))
        else:
            return self.vis.image(img)

    def showbatch(self, batch):
        # if self.window is not None:
        # self.vis.close(self.window)
        new_imgs = []
        for img in batch.cpu():
            img = self.unnorm(img).numpy()
            if len(img.shape) == 2:
                # grayscale
                img = np.stack([img, img, img])
            new_imgs.append(img)
        self.window = self.vis.images(np.asarray(new_imgs), opts=dict(title='batch'))

    def showvideo(self, batch):
        new_imgs = []
        for img in batch:
            new_imgs.append(self.unnorm(img).numpy())
        self.window = self.vis.video(np.asarray(new_imgs))
