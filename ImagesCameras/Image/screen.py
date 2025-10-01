from typing import Literal
import cv2
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import patches
from einops import rearrange
import multiprocessing as mp


# ---------- Utility helpers ----------
def add_padding(img, pad=2):
    """Add black border around an image."""
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)


def concat_with_space(imgs, direction="h", pad=2):
    """Concatenate images with black space between them."""
    imgs_padded = [add_padding(im, pad) for im in imgs]
    if direction == "h":
        return cv2.hconcat(imgs_padded)
    else:
        return cv2.vconcat(imgs_padded)


def find_best_grid(param):
    srt = int(np.floor(np.sqrt(param)))
    i = 0
    while srt * (srt + i) < param:
        i += 1
    return srt, srt + i


# ---------- Worker process for async opencv ----------
def _opencv_display_loop(screen, **kwargs):
    img = None
    while True:
        # fetch newest available image
        try:
            while not screen.queue.empty():
                img = screen.queue.get_nowait()
        except:
            pass
        if img is not None:
            screen = screen.show(**kwargs)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC = quit
            break

    cv2.destroyAllWindows()


# ---------- The Screen class ----------
class Screen:
    def __init__(self, images):
        """
        tensor: torch.Tensor or np.ndarray
            Expected shape: (b, c, h, w)
        """
        self.images = images.permute('b', 'c', 'h', 'w').detach().cpu()
        self.windowName = 'Screen'
        self._async = False
        self._viewer_proc = None
        self._queue = None

    @property
    def queue(self):
        return self._queue

    @queue.setter
    def queue(self, value):
        self._queue = value

    @property
    def async_mode(self):
        return self._async

    @async_mode.setter
    def async_mode(self, value):
        self._async = value

    @property
    def viewer_proc(self):
        return self._viewer_proc

    @viewer_proc.setter
    def viewer_proc(self, value):
        self._viewer_proc = value

    def show(self,
             backend=Literal["matplotlib", "opencv"],
             name: str | None = None,
             cmap: str = 'gray',
             roi: list = None,
             point: torch.Tensor | list = None,
             save: str = '',
             split_batch: bool = False,
             split_channel: bool = False,
             pad: int = 2,
             asyncr: bool = False):
        """
        Show the tensor using matplotlib or opencv with optional sliders.
        """
        self.windowName = name if name else self.windowName
        if backend == "matplotlib":
            if self.images.modality == 'Multimodal' or self.images.batch_size > 1 or split_batch or split_channel:
                return self._multiple_show_matplot(cmap=cmap,
                                                   split_batch=split_batch,
                                                   split_channel=split_channel)
            else:
                return self._single_show_matplot(cmap=cmap,
                                                 roi=roi, point=point, save=save,
                                                 split_channel=split_channel)

        elif backend == "opencv":
            if asyncr and not self.async_mode:
                return self._start_async_opencv(name, split_batch, split_channel, pad, roi, point)
            if self.images.modality == 'Multimodal' or self.images.batch_size > 1 or split_batch:
                return self._multiple_show_opencv(split_batch=split_batch,
                                                  split_channel=split_channel,
                                                  pad=pad)
            else:
                return self._single_show_opencv(roi=roi, point=point, save=save,
                                                split_channel=split_channel,
                                                pad=pad)

    # ---------- Matplotlib implementations ----------
    def _single_show_matplot(self, cmap, roi, point, save, split_channel):
        matplotlib.use('TkAgg')
        num = self.images.name if self.windowName is None else self.windowName
        channels_names = self.images.channel_names if self.images.channel_names else np.arange(0,
                                                                                               self.images.channel_num).tolist()

        if split_channel:
            im_display = self.images.permute(['b', 'c', 'h', 'w']).numpy().squeeze()
            fig, axe = plt.subplots(1, 1, num=num)
            plt.subplots_adjust(left=0.15)
            axe_channel = plt.axes((0.03, 0.05, 0.05, 0.8))
            channel_slider = Slider(ax=axe_channel, label='Channel', valmin=0,
                                    valmax=self.images.channel_num - 1, valstep=1, valinit=0,
                                    orientation="vertical")

            def update(i):
                cmap_ = None if self.images.p_modality != 'Any' else cmap
                axe.clear()
                axe.imshow(im_display[int(i)], cmap_)
                axe.set_title(f"Channel {channels_names[int(channel_slider.val)]}")
                if point is not None:
                    for center in np.array(point).squeeze():
                        circle = patches.Circle(center, 5, linewidth=2, edgecolor='r', facecolor='none')
                        axe.add_patch(circle)
                if roi is not None:
                    for r, color in zip(roi, ['r', 'g', 'b']):
                        rect = patches.Rectangle((r[0], r[2]), r[1] - r[0], r[3] - r[2],
                                                 linewidth=2, edgecolor=color, facecolor='none')
                        axe.add_patch(rect)
                axe.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                fig.canvas.draw_idle()

            channel_slider.on_changed(update)
            update(0)
            plt.show()
        else:
            im_display = self.images.permute(['b', 'h', 'w', 'c']).numpy().squeeze()
            fig, axe = plt.subplots(ncols=1, nrows=1, num=num, squeeze=True)
            axe.imshow(im_display, cmap=None if self.images.p_modality not in ['Any', 'Depth'] else cmap)
            if point is not None:
                for center in np.array(point).squeeze():
                    circle = patches.Circle(center, 5, linewidth=2, edgecolor='r', facecolor='none')
                    axe.add_patch(circle)
            if roi is not None:
                for r, color in zip(roi, ['r', 'g', 'b']):
                    rect = patches.Rectangle((r[0], r[2]), r[1] - r[0], r[3] - r[2],
                                             linewidth=2, edgecolor=color, facecolor='none')
                    axe.add_patch(rect)
            axe.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            if save:
                fig.savefig(f'{save}.png', bbox_inches='tight', dpi=300)
            plt.show()
        return self

    def _multiple_show_matplot(self, cmap, split_batch, split_channel):
        num = self.images.name if self.windowName is None else self.windowName
        channels_names = self.images.channel_names if self.images.channel_names else np.arange(0,
                                                                                               self.images.channel_num).tolist()
        # plt.ion()
        if split_batch and split_channel:
            im_display = self.images.permute(['b', 'c', 'h', 'w']).to_numpy().squeeze()
            fig, axes = plt.subplots(1, 1, num=num)
            plt.subplots_adjust(left=0.15, bottom=0.15)
            # Make a horizontal slider to control the batch.
            axe_batch = plt.axes((0.15, 0.05, 0.75, 0.05))
            batch_slider = Slider(
                ax=axe_batch,
                label='Batch number',
                valmin=0,
                valmax=self.images.batch_size - 1,
                valstep=1,
                valinit=0)
            # Make a vertical slider to control the channel.
            axe_channel = plt.axes((0.03, 0.15, 0.05, 0.8))
            channel_slider = Slider(
                ax=axe_channel,
                label='Channel',
                valmin=0,
                valmax=self.images.channel_num - 1,
                valstep=1,
                valinit=0,
                orientation="vertical")

            def update(i):
                match self.images.colorspace:
                    case 'RGB':
                        cmap_ = ['Reds', 'Greens', 'Blues'][int(channel_slider.val)]
                    case _:
                        cmap_ = None if self.images.p_modality != 'Any' else cmap
                axes.imshow(im_display[int(batch_slider.val), int(channel_slider.val)], cmap_)
                axes.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                axes.set_title(f" Image {i} from batch, Channel {channels_names[int(channel_slider.val)]}")
                plt.show()

            batch_slider.on_changed(update)
            channel_slider.on_changed(update)
            update(0)
        elif split_batch:
            im_display = self.images.permute(['b', 'h', 'w', 'c']).to_numpy().squeeze()
            if self.images.channel_num == 3 or self.images.channel_num == 1:
                fig, axes = plt.subplots(1, 1, num=num)
                im_display = im_display[:, None]
            else:
                rows, cols = find_best_grid(self.images.channel_num)
                fig, axes = plt.subplots(rows, cols, num=num)
                axes = axes.flatten()
                im_display = im_display.moveaxis(-1, 1)
            plt.subplots_adjust(bottom=0.15)
            # Make a horizontal slider to control the batch.
            axe_batch = plt.axes((0.15, 0.05, 0.75, 0.05))
            batch_slider = Slider(
                ax=axe_batch,
                label='Batch number',
                valmin=0,
                valmax=self.images.batch_size - 1,
                valstep=1,
                valinit=0)

            def update(i):
                for j, axe in enumerate(axes):
                    if j < im_display.shape[1]:
                        axe.imshow(im_display[int(batch_slider.val), j], cmap)
                        axe.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                    else:
                        if axe.axes is not None:
                            axe.remove()
                fig.suptitle(f" Image {i} from batch, Channel {channels_names[int(channel_slider.val)]}")
                plt.show()

            batch_slider.on_changed(update)
            update(0)
        elif split_channel:
            im_display = self.images.permute(['b', 'c', 'h', 'w']).to_numpy()
            rows, cols = find_best_grid(self.images.batch_size)
            fig, axes = plt.subplots(rows, cols, num=num, squeeze=False)
            axes = axes.flatten()
            plt.subplots_adjust(left=0.15)
            # Make a vertical slider to control the channel.
            axe_channel = plt.axes((0.03, 0.05, 0.05, 0.8))
            channel_slider = Slider(
                ax=axe_channel,
                label='Channel',
                valmin=0,
                valmax=self.images.channel_num - 1,
                valstep=1,
                valinit=0,
                orientation="vertical")

            def update(i):
                match self.images.colorspace:
                    case 'RGB':
                        cmap_ = ['Reds', 'Greens', 'Blues'][int(channel_slider.val)]
                    case _:
                        cmap_ = None if self.images.p_modality != 'Any' else cmap
                for j, axe in enumerate(axes):
                    if j < im_display.shape[0]:
                        axe.imshow(im_display[j, int(i)], cmap_)
                        axe.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                    else:
                        if axe.axes is not None:
                            axe.remove()
                fig.suptitle(f"Channel {channels_names[int(channel_slider.val)]}")
                plt.show()

            channel_slider.on_changed(update)
            update(0)
        else:
            im = self.images.permute(['b', 'c', 'h', 'w'])
            if (im.channel_num == 3 and im.colorspace == 'RGB') or (im.channel_num == 4 and im.colorspace == 'RGBA'):
                im_display = rearrange(im.to_tensor(), 'b c h w -> b h w c').detach().cpu().numpy()
            else:
                im_display = rearrange(im.to_tensor(), 'b c h w -> (b c) h w').detach().cpu().numpy()
            rows, cols = find_best_grid(im_display.shape[0])
            fig, axes = plt.subplots(rows, cols, num=num)
            axes = axes.flatten()
            for j, axe in enumerate(axes):
                if j < im_display.shape[0]:
                    axe.imshow(im_display[j], cmap)
                    axe.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                else:
                    if axe.axes is not None:
                        axe.remove()
            plt.show()
        plt.ioff()
        return self

    # ---------- OpenCV implementations ----------

    def _single_show_opencv(self, roi, point, save, split_channel, pad):
        win_name = self.images.name if self.windowName is None else self.windowName
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        if split_channel and self.images.channel_num > 1:
            im_display = self.images.permute(['b', 'c', 'h', 'w']).numpy().squeeze()
            nb_channels = im_display.shape[0]

            def nothing(x):
                pass

            cv2.createTrackbar("Channel", win_name, 0, nb_channels - 1, nothing)

            while True:
                if self.async_mode and self.queue is not None:
                    # fetch newest available image
                    try:
                        while not self.queue.empty():
                            im_display, win_name = self.queue.get_nowait()
                            im_display = im_display.numpy().squeeze()
                            cv2.destroyAllWindows()
                            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                    except:
                        pass
                ch = cv2.getTrackbarPos("Channel", win_name)
                img = im_display[ch]
                img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                if roi is not None:
                    for r, color in zip(roi, [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                        cv2.rectangle(img_norm, (r[0], r[2]), (r[1], r[3]), color, 2)
                if point is not None:
                    for center in np.array(point).squeeze():
                        cv2.circle(img_norm, tuple(center), 5, (0, 0, 255), 2)

                img_norm = add_padding(img_norm, pad)
                cv2.imshow(win_name, img_norm)
                key = cv2.waitKey(50) & 0xFF
                if key == 27 or key == 0:
                    break
        else:
            im_display = self.images.permute(['b', 'h', 'w', 'c']).numpy().squeeze()
            while True:
                if self.async_mode and self.queue is not None:
                    # fetch newest available image
                    try:
                        while not self.queue.empty():
                            im_display, win_name = self.queue.get_nowait()
                            im_display = im_display.permute(['b', 'h', 'w', 'c']).numpy().squeeze()
                            cv2.destroyAllWindows()
                            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                    except:
                        pass
                img = cv2.normalize(im_display, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                if img.ndim == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if roi is not None:
                    for r, color in zip(roi, [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                        cv2.rectangle(img, (r[0], r[2]), (r[1], r[3]), color, 2)
                if point is not None:
                    for center in np.array(point).squeeze():
                        cv2.circle(img, tuple(center), 5, (0, 0, 255), 2)

                img = add_padding(img, pad)
                cv2.imshow(win_name, img)
                key = cv2.waitKey(50) & 0xFF
                if key == 27 or key == 0:
                    break
            if save:
                cv2.imwrite(f"{save}.png", img)

        cv2.destroyAllWindows()
        return self

    def _multiple_show_opencv(self, split_batch, split_channel, pad):
        """
                Version OpenCV du _multiple_show_matplot()
                avec sliders interactifs.
                """
        win_name = self.images.name if self.windowName is None else self.windowName
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        channels_names = self.images.channel_names if self.images.channel_names \
            else np.arange(0, self.images.channel_num).tolist()

        # --- Cas split_batch ET split_channel ---
        if split_batch and split_channel:
            im_display = self.images.permute(['b', 'c', 'h', 'w']).numpy().squeeze()

            def nothing(x):
                pass

            cv2.createTrackbar("Batch", win_name, 0, self.images.batch_size - 1, nothing)
            cv2.createTrackbar("Channel", win_name, 0, self.images.channel_num - 1, nothing)

            while True:
                if self.async_mode and self.queue is not None:
                    # fetch newest available image
                    try:
                        while not self.queue.empty():
                            im_display = self.queue.get_nowait().numpy().squeeze()
                            cv2.setWindowTitle(win_name, self.windowName)
                    except:
                        pass
                b = cv2.getTrackbarPos("Batch", win_name)
                ch = cv2.getTrackbarPos("Channel", win_name)
                img = im_display[b, ch]

                img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                img_norm = add_padding(img_norm, pad)
                cv2.imshow(win_name, img_norm)
                key = cv2.waitKey(50) & 0xFF
                if key == 27 or key == 0:  # ESC or ENTER
                    break

        # --- Cas split_batch seulement ---
        elif split_batch:
            im_display = self.images.permute(['b', 'h', 'w', 'c']).to_numpy().squeeze()

            def nothing(x):
                pass

            cv2.createTrackbar("Batch", win_name, 0, self.images.batch_size - 1, nothing)

            while True:
                if self.async_mode and self.queue is not None:
                    # fetch newest available image
                    try:
                        while not self.queue.empty():
                            im_display = self.queue.get_nowait().to_numpy().squeeze()
                            cv2.setWindowTitle(win_name, self.windowName)
                    except:
                        pass
                b = cv2.getTrackbarPos("Batch", win_name)
                img = im_display[b]

                img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                if img_norm.ndim == 3 and img_norm.shape[2] == 3:
                    img_norm = cv2.cvtColor(img_norm, cv2.COLOR_RGB2BGR)

                img_norm = add_padding(img_norm, pad)
                cv2.imshow(win_name, img_norm)
                key = cv2.waitKey(50) & 0xFF
                if key == 27 or key == 0:
                    break

        # --- Cas split_channel seulement ---
        elif split_channel:
            im_display = self.images.permute(['b', 'c', 'h', 'w']).to_numpy()

            def nothing(x):
                pass

            cv2.createTrackbar("Channel", win_name, 0, self.images.channel_num - 1, nothing)

            while True:
                if self.async_mode and self.queue is not None:
                    # fetch newest available image
                    try:
                        while not self.queue.empty():
                            im_display = self.queue.get_nowait().to_numpy()
                            cv2.setWindowTitle(win_name, self.windowName)
                    except:
                        pass
                ch = cv2.getTrackbarPos("Channel", win_name)
                # On affiche toutes les images batch pour le canal choisi (grille concaténée)
                imgs = []
                for b in range(self.images.batch_size):
                    img = im_display[b, ch]
                    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    img_norm = add_padding(img_norm, pad)
                    imgs.append(img_norm)

                # Concaténation horizontale (ou verticale si trop large)
                concat = cv2.hconcat(imgs) if len(imgs[0].shape) == 2 else np.hstack(imgs)
                cv2.imshow(win_name, concat)

                key = cv2.waitKey(50) & 0xFF
                if key == 27 or key == 0:
                    break

        # --- Cas simple (pas de split) ---
        else:
            im = self.images.permute(['b', 'c', 'h', 'w']).to_numpy()
            while True:
                if self.async_mode and self.queue is not None:
                    # fetch newest available image
                    try:
                        while not self.queue.empty():
                            im = self.queue.get_nowait().to_numpy()
                            cv2.setWindowTitle(win_name, self.windowName)
                    except:
                        pass
                if (im.channel_num == 3 and im.colorspace == 'RGB') or (im.channel_num == 4 and im.colorspace == 'RGBA'):
                    im_display = rearrange(im.to_tensor(), 'b c h w -> b h w c').detach().cpu().numpy()
                else:
                    im_display = rearrange(im.to_tensor(), 'b c h w -> (b c) h w').detach().cpu().numpy()

                imgs = []
                for i in range(im_display.shape[0]):
                    img_norm = cv2.normalize(im_display[i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    if img_norm.ndim == 3 and img_norm.shape[2] == 3:
                        img_norm = cv2.cvtColor(img_norm, cv2.COLOR_RGB2BGR)
                    img_norm = add_padding(img_norm, pad)
                    imgs.append(img_norm)

                # Grille automatique (concaténation en ligne)
                concat = cv2.hconcat(imgs) if imgs[0].ndim == 2 else np.hstack(imgs)
                cv2.imshow(win_name, concat)
                key = cv2.waitKey(50) & 0xFF
                if key == 27 or key == 0:
                    break

        cv2.destroyAllWindows()
        return self

    # ---------- Update method ----------
    def update(self, new_tensor, name=None):
        """Update tensor for async OpenCV mode"""
        if name is not None:
            self.windowName = name
        self.images = new_tensor.detach().cpu()
        if self.async_mode and self.queue is not None:
            self.queue.put((self.images.permute(['b', 'c', 'h', 'w']), self.windowName))

    def close(self):
        if self._async and self._viewer_proc is not None:
            self._viewer_proc.terminate()
            self._viewer_proc.join()
            self._async = False
            self._viewer_proc = None
            self._queue = None

    def _start_async_opencv(self, name, split_batch, split_channel, pad, roi, point,):
        self._queue = mp.Queue()
        self._async = True
        self._queue.put(self.images.permute(['b', 'c', 'h', 'w']).numpy())
        self._viewer_proc = mp.Process(target=self.show,
                                       kwargs={'backend': 'opencv', 'name': name, 'split_batch': split_batch,
                                               'split_channel': split_channel, 'pad': pad, 'roi': roi, 'point': point})
        self._viewer_proc.daemon = True
        self._viewer_proc.start()
