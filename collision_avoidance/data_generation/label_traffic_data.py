import pandas as pd
import numpy as np

def cosd(x):
    return np.cos(x * np.pi / 180)

def sind(x):
    return np.sin(x * np.pi / 180)

def tand(x):
    return np.tan(x * np.pi / 180)

def get_bounding_box(e0, n0, u0, h0, e1, n1, u1, hfov=80, vfov=49.5, offset=0, tilt=-1.20, sw=1920, sh=1056, aw0=0, daw=17000):
    # Make ownship be the origin
    x = n1 - n0
    y = -(e1 - e0)  # right-handed coordinates
    z = u1 - u0

    # Rotate x and y according to ownship heading
    xrot = x * cosd(h0) - y * sind(h0)
    yrot = -(x * sind(h0) + y * cosd(h0))

    # Account for offset
    z = z + offset

    # Rotate z according to tilt angle
    xcam = xrot * cosd(tilt) - z * sind(tilt)
    ycam = yrot
    zcam = xrot * sind(tilt) + z * cosd(tilt)

    # https://www.youtube.com/watch?v=LhQ85bPCAJ8
    xp = ycam / (xcam * tand(hfov / 2))
    yp = zcam / (xcam * tand(vfov / 2))

    # Get xp and yp between 0 and 1
    xp = (xp + 1) / 2
    yp = (yp + 1) / 2

    # Map to pixel location
    xp = xp * sw
    yp = (1 - yp) * sh

    # Get height and width of bounding box
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    w = daw * (1 / r) + aw0
    h = (3 / 8) * w

    return xp, yp, w, h

def gen_labels(outdir):
    # Load in the positions
    data_file = outdir + "state_data.csv"
    df = pd.read_csv(data_file)

    # Start new file with labels
    label_file = outdir + "bounding_boxes.csv"
    with open(label_file, 'w') as fd:
        fd.write("filename,xp,yp,w,h\n")

        for i in range(len(df)):
            xp, yp, w, h = get_bounding_box(
                df['e0'][i], df['n0'][i], df['u0'][i], df['h0'][i], df['e1'][i], df['n1'][i], df['u1'][i])
            fd.write("%d,%f,%f,%f,%f,\n" %
                     (i, xp, yp, w, h))

def gen_labels_yolo(outdir):
    # Load in the positions
    data_file = outdir + "state_data.csv"
    df = pd.read_csv(data_file)

    for i in range(len(df)):
        xp, yp, w, h = get_bounding_box(
            df['e0'][i], df['n0'][i], df['u0'][i], df['h0'][i], df['e1'][i], df['n1'][i], df['u1'][i])

        file_name = outdir + "imgs/" + str(i) + ".txt"
        with open(file_name, 'w') as fd:
            fd.write("0 %f %f %f %f\n" %
                     (xp  / 1920, yp / 1056, w / 1920, h / 1080))
    
    label_name = file_name = outdir + "imgs/darket.labels"
    with open(label_name, 'w') as fd:
        fd.write("aircraft")


outdir = "/home/smkatz/Documents/RiskSensitivePerception/collision_avoidance/data_files/traffic_data/"
gen_labels_yolo(outdir)
