
import numpy as np
import matplotlib.pyplot as plt


def Scroller(X):

    MAX = max(np.concatenate(X,axis=None))

    class IndexTracker(object):
        def __init__(self, ax, X):
            self.ax = ax
            ax.set_title('use scroll wheel to navigate images')

            self.X = X
            rows, cols, self.slices = X.shape
            self.ind = self.slices//2

            self.im = ax.imshow(self.X[:,:, self.ind], vmax=MAX,aspect = 'equal')
            self.update()

        def onscroll(self, event):
            if event.button == 'up':
                self.ind = (self.ind + 1) % self.slices
            else:
                self.ind = (self.ind - 1) % self.slices
            self.update()

        def update(self):
            self.im.set_data(self.X[:, :, self.ind])
            ax.set_ylabel('slice %s' % self.ind)
            self.im.axes.figure.canvas.draw()


    fig, ax = plt.subplots(1, 1)

    tracker = IndexTracker(ax, X)

    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)

    plt.show()


def ScrollerMulti(X,num,name):

    MAX = max(np.concatenate(X,axis=None))

    class IndexTracker(object):
        def __init__(self, ax, X, num,name):
            
            self.ax = ax
            self.X = X
            self.im = self.ax
            for i in range(num):         
                ax[i].set_title(name[i])

                rows, cols, self.slices = X[i].shape
                self.ind = self.slices//2

                self.im[i] = ax[i].imshow(self.X[i][:,:, self.ind], vmax=MAX, aspect = 'equal')
            self.update()

        def onscroll(self, event):
            if event.button == 'up':
                self.ind = (self.ind + 1) % self.slices
            else:
                self.ind = (self.ind - 1) % self.slices
            self.update()

        def update(self):
            for i in range(num):
                self.im[i].set_data(self.X[i][:, :, self.ind])
                self.im[i].axes.set_ylabel('slice %s' % self.ind)

            for i in range(num):    #To allow for pause and avoid stutter
                self.im[i].axes.figure.canvas.draw()


    fig, ax = plt.subplots(1, num)

    tracker = IndexTracker(ax, X, num,name)

    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)

    plt.show()


print('Debug')