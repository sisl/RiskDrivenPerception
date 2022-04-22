import pymap3d as pm
from data_generation.xpc3 import *
import time
import cv2
import mss
import os
import torch
import numpy as np
from yolov5.models.yolo import Model
from yolov5.models.common import *
from yolov5.utils.torch_utils import fuse_conv_and_bn

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.patches as patches
import imageio

screen_shot = mss.mss()

####################################################
# X-Plane 11 functions
####################################################

# client=XPlaneConnect()
# model = load_model("models/traffic_detector_v2.pt")
# save_img_w_bb(model, get_screenshot(), "out.jpg")

def get_bounding_box(client, model, e0, n0, u0, psi0, e1, n1, u1, psi1, save):
    client.sendDREF("sim/time/zulu_time_sec", 9.0*3600 + 8*3600)
    set_position(client, 0, e0, n0, u0, -psi0, 0, 0)
    set_position(client, 1, e1, n1, u1, -psi1, 0, 0)
    # time.sleep(0.1)
    return save_img_w_bb(model, get_screenshot(), save)
    

def set_position(client, ac, e, n, u, psi, pitch=-998, roll=-998):
    ref = [37.46358871459961, -122.11750030517578, 1578.909423828125]
    p = pm.enu2geodetic(e, n, u, ref[0], ref[1], ref[2])
    client.sendPOSI([*p, pitch, roll, psi], ac)


def get_screenshot(screen_shot=screen_shot):
    
    # monitor = screen_shot.monitors[1]
    monitor = screen_shot.monitors[1] = {'left':1920, 'top':0, 'width':1920, 'height':1080}
    ss = cv2.cvtColor(np.array(screen_shot.grab(monitor)), cv2.COLOR_BGRA2BGR)[12:-12, :, ::-1]

    # Deal with screen tearing
    ss_sum = np.reshape(np.sum(ss, axis=-1), -1)
    ind = 0
    while np.min(ss_sum) == 0 and ind < 10:
        # print("Screen tearing detected. Trying again...")
        ss = cv2.cvtColor(np.array(screen_shot.grab(
            monitor)), cv2.COLOR_BGRA2BGR)[12:-12, :, ::-1]
        ss_sum = np.reshape(np.sum(ss, axis=-1), -1)
        ind += 1

    return ss


def generate_test_traj(start_point, end_point, rel_alt, dt=0.2, speed=50):
    x0, y0 = start_point
    xf, yf = end_point

    total_dist = np.sqrt((xf - x0) ** 2 + (yf - y0) ** 2)
    total_time = total_dist / speed
    nsteps = int(np.rint(total_time / dt))

    theta = np.arctan2(yf - y0, xf - x0)

    xs = np.zeros(nsteps + 1)
    ys = np.zeros(nsteps + 1)

    xs[0] = x0
    ys[0] = y0

    for i in range(1, nsteps):
        xs[i] = xs[i - 1] + speed * np.cos(theta) * dt
        ys[i] = ys[i - 1] + speed * np.sin(theta) * dt

    zs = rel_alt * np.ones(nsteps + 1)

    return xs, ys, zs, theta


def simulate_test_traj(xs, ys, zs, theta):
    imgs = []

    # Make sure ownship is at origin
    set_position(client, 0, 0, 0, 0, 0, roll=0, pitch=0)

    theta_deg = theta * 180 / np.pi

    for i in range(len(xs)):
        # Position intruder
        set_position(client, 1, xs[i], ys[i], zs[i],
                     theta_deg, roll=0, pitch=0)
        time.sleep(0.2)
        # Take screen shot
        img = get_screenshot(screen_shot)
        imgs.append(img)

    return imgs

####################################################
# yolo model helpers
####################################################

def load_model(ckpt_file):
    ckpt = torch.load(ckpt_file)

    # Create model architecture
    model = Model(ckpt['model'].yaml)

    # Load the weights
    model.load_state_dict(ckpt['model'].state_dict())

    # Fuse and autoshape
    print("Fusing...")
    for m in model.modules():
        if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
            delattr(m, 'bn')  # remove batchnorm
            m.forward = m.forward_fuse  # update forward

    model = AutoShape(model)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    return model


def show_img_w_bb(model, im):
    f, ax = plt.subplots()
    f.set_figwidth(14)
    f.set_figheight(14)

    ax.imshow(im)

    df = model(im).pandas().xyxy[0]

    if not df.empty:
        xmin = np.rint(df['xmin'][0])
        xmax = np.rint(df['xmax'][0])
        ymin = np.rint(df['ymin'][0])
        ymax = np.rint(df['ymax'][0])

        xp = xmin
        yp = ymin
        w = xmax - xmin
        h = ymax - ymin

        conf = np.round(df['confidence'][0], decimals=2)

        rect = patches.Rectangle((xp, yp),
                                 w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xp + (w / 2), yp - 10, "Conf: " + str(conf), horizontalalignment='center',
                verticalalignment='center', color='r')

    plt.xlim(0, 1920)
    plt.ylim(1056, 0)
    plt.axis('off')


def save_img_w_bb(model, im, save):
    
    df = model(im).pandas().xyxy[0]
    
    if save > -1:
        f, ax = plt.subplots()
        f.set_figwidth(14)
        f.set_figheight(14)

        ax.imshow(im)
        
        if not df.empty:
            xmin = np.rint(df['xmin'][0])
            xmax = np.rint(df['xmax'][0])
            ymin = np.rint(df['ymin'][0])
            ymax = np.rint(df['ymax'][0])

            xp = xmin
            yp = ymin
            w = xmax - xmin
            h = ymax - ymin

            conf = np.round(df['confidence'][0], decimals=2)

            rect = patches.Rectangle((xp, yp),
                                     w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(xp + (w / 2), yp - 10, "Conf: " + str(conf), horizontalalignment='center',
                    verticalalignment='center', color='r')

        plt.xlim(0, 1920)
        plt.ylim(1056, 0)
        plt.axis('off')

        filename = 'imgs/' + str(save) + '.png'
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close(f)
        
    return not df.empty

def create_gif(nsteps, dir='imgs/', gifname='out.gif'):
    frames = []
    for i in range(1, nsteps+1):
        filename = dir + str(i) + '.png'
        frames.append(imageio.imread(filename))
        
    imageio.mimsave(gifname, frames, 'GIF', fps=5)
    
    # Remove files
    for i in range(1, nsteps+1):
        filename = dir + str(i) + '.png'
        os.remove(filename)

def create_bounding_gif(imgs, gifname):
    # Create the images
    filenames = []
    for (i, img) in enumerate(imgs):
        filename = str(i) + ".png"
        save_img_w_bb(img, filename)
        filenames.append(filename)

    # Build gif
    frames = []
    for filename in filenames:
        frames.append(imageio.imread(filename))

    imageio.mimsave(gifname, frames, 'GIF', fps=5)

    # Remove files
    for filename in set(filenames):
        os.remove(filename)
