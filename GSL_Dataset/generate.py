from blender_for_gsl import *
import os
import argparse
import shutil
import getpass
import json
import random
import sys


def extract_args(input_argv=None):
    """
  Pull out command-line arguments after "--". Blender ignores command-line flags
  after --, so this lets us forward command line arguments from the blender
  invocation to our own script.
  """
    if input_argv is None:
        input_argv = sys.argv
    output_argv = []
    if '--' in input_argv:
        idx = input_argv.index('--')
        output_argv = input_argv[(idx + 1):]
    return output_argv


def u_rgb(rgb):
    r = float(rgb[0]) / 255.0
    g = float(rgb[1]) / 255.0
    b = float(rgb[2]) / 255.0
    return (r, g, b, 1.0)


def u_getitem(js, box):
    """
    :param js: json file
    :param box: list name use to store value
    :return:
    """
    for key, value in js.items():
        box.append(value)


def u_diffitem(sets, item):
    """

    :param sets:
    :param item:
    :return:
    """
    if item == sets[0]:
        return False
    return True


def u_getun(set, ori):
    """
    :param set:  Group value
    :param ori: Possible values
    :return:
    """
    temp = random.choice(ori)
    if not set:
        return temp
    while True:
        # print(temp)
        if u_diffitem(set, temp):
            break
        temp = random.choice(ori)
    return temp


def u_mkdir(path):
    """
    :param path:
    :return: null
    """
    if not os.path.exists(path):
        os.makedirs(path)


def p_gen(num, objs, colors, sizes, speeds, moves, backs, rotations, resolution, out_dir):
    # Step 1: generate setting file name and store location
    video_dir = os.path.join(out_dir, "Group" + str(num))
    u_mkdir(video_dir)
    doc = os.path.join(video_dir, "Description")

    # Step 2: Generate setting for the group
    obj = []
    col = []
    size = []
    speed = []
    move = []
    back = []
    rotation = []
    name = []
    frames = []
    common = ["/", "Object", "Color", "Size", "Speed", "Movement", "Background", "Rotation"]
    for i in range(0, 8):
        if i == 0:
            print("center")
            name.append("center.avi")
            obj.append(u_getun(obj, objs))
            col.append(u_getun(col, colors))
            size.append(u_getun(size, sizes))
            speed.append(u_getun(speed, speeds))
            move.append(u_getun(move, moves))
            back.append(u_getun(back, backs))
            rotation.append(u_getun(rotation, rotations))

        else:
            name.append("%06d.avi" % i)
            if i == 1:
                print("01")
                # Common Object
                obj.append(obj[0])
                col.append(u_getun(col, colors))
                size.append(u_getun(size, sizes))
                speed.append(u_getun(speed, speeds))
                move.append(u_getun(move, moves))
                back.append(u_getun(back, backs))
                rotation.append(u_getun(rotation, rotations))
            elif i == 2:
                print("02")
                # Common Color
                obj.append(u_getun(obj, objs))
                col.append(col[0])
                size.append(u_getun(size, sizes))
                speed.append(u_getun(speed, speeds))
                move.append(u_getun(move, moves))
                back.append(u_getun(back, backs))
                rotation.append(u_getun(rotation, rotations))
            elif i == 3:
                print("03")
                # Common Size
                obj.append(u_getun(obj, objs))
                col.append(u_getun(col, colors))
                size.append(size[0])
                speed.append(u_getun(speed, speeds))
                move.append(u_getun(move, moves))
                back.append(u_getun(back, backs))
                rotation.append(u_getun(rotation, rotations))
            elif i == 4:
                print("04")
                # Common Speed
                obj.append(u_getun(obj, objs))
                col.append(u_getun(col, colors))
                size.append(u_getun(size, sizes))
                speed.append(speed[0])
                move.append(u_getun(move, moves))
                back.append(u_getun(back, backs))
                rotation.append(u_getun(rotation, rotations))
            elif i == 5:
                print("05")
                # Common Movement
                obj.append(u_getun(obj, objs))
                col.append(u_getun(col, colors))
                size.append(u_getun(size, sizes))
                speed.append(u_getun(speed, speeds))
                move.append(move[0])
                back.append(u_getun(back, backs))
                rotation.append(u_getun(rotation, rotations))
            elif i == 6:
                print("06")
                # Common Background
                obj.append(u_getun(obj, objs))
                col.append(u_getun(col, colors))
                size.append(u_getun(size, sizes))
                speed.append(u_getun(speed, speeds))
                move.append(u_getun(move, moves))
                back.append(back[0])
                rotation.append(u_getun(rotation, rotations))
            else:
                print("07")
                # Common Rotation
                obj.append(u_getun(obj, objs))
                col.append(u_getun(col, colors))
                size.append(u_getun(size, sizes))
                speed.append(u_getun(speed, speeds))
                move.append(u_getun(move, moves))
                back.append(u_getun(back, backs))
                rotation.append(rotation[0])

    # Step 3: Render the videos
    for i in range(0, 8):
        temp = move[i]
        temp = temp[1:len(temp) - 1].split(" ")
        mm = []
        for h in temp:
            temp1 = h[1:len(h) - 1].split(",")
            temp2 = [float(c) for c in temp1]
            mm.append(temp2)
        # print(len(mm[0]))
        # print(mm)
        frame = render(
            path=os.path.join(video_dir, name[i]),
            object=obj[i],
            color=u_rgb(col[i]),
            background=u_rgb(back[i]),
            sizes=size[i],
            speeds=speed[i],
            movement=mm,
            rot=[rotation[i][0], math.radians(rotation[i][1])],
            x=resolution[0],
            y=resolution[1]
        )
        frame = 0.75*frame
        frames.append(frame)

    # Step 4: Generate doc
    with open(doc, 'w') as f:
        f.write("Group" + str(num) + "\n")
        f.write(format("Name", '<10') + "\t")
        f.write(format("Object", '<10') + "\t")
        f.write(format("Color", '<15') + "\t")
        f.write(format("Size", '<5') + "\t")
        f.write(format("Speed", '<5') + "\t")
        f.write(format("Movement", '<70') + "\t")
        f.write(format("Background", '<20') + "\t")
        f.write(format("Rotation", '<8') + "\t")
        f.write(format("Common", '<10') + "\t")
        f.write(format("Frame#", '<10') + "\n")
        for i in range(0, 8):
            f.write(format(name[i], '<10') + "\t")
            f.write(format(obj[i], '<10') + "\t")
            f.write(format(str(col[i]), '<15') + "\t")
            f.write(format(str(size[i]), '<5') + "\t")
            f.write(format(str(speed[i]), '<5') + "\t")
            f.write(format(move[i], '<70') + "\t")
            f.write(format(str(back[i]), '<20') + "\t")
            f.write(format(str(rotation[i]), '<8') + "\t")
            f.write(format(common[i], '<10') + "\t")
            f.write(format(str(frames[i]), '<10') + "\n")
    print("Finish generating doc")


def main(args):
    # print(args.out_dir)
    # print(args.set_dir)
    # print(args.num)
    # print(args.gpu)

    # Step 1: Read basic info from Setting.json
    objs = []
    colors = []
    sizes = []
    speeds = []
    moves = []
    backs = []
    rotations = []
    resolution = []
    fps = []
    with open(os.path.join(args.set_dir, "Setting.json"), 'r') as f:
        setting = json.load(f)
        u_getitem(setting['Object'], objs)
        u_getitem(setting['Color'], colors)
        u_getitem(setting['Size'], sizes)
        u_getitem(setting['Speed'], speeds)
        u_getitem(setting['Movement'], moves)
        u_getitem(setting['Background'], backs)
        u_getitem(setting['Rotation'], rotations)
        u_getitem(setting['Resolution'], resolution)
        u_getitem(setting['FPS'], fps)

    f.close()
    print("Finish reading setting")

    # Step 2: Generate Readme for groups data
    with open(os.path.join(args.out_dir, "Groupinfo.md"), 'w') as f:
        f.write("#### Attribute\n")
        f.write("|Attribute|Possible Values|Dynamic/Static|\n"
                "|---|---|---|\n"
                "|Object|")
        for i in objs:
            f.write(i + " ")
        f.write("|S|\n|Color|RGB ")
        for i in colors:
            f.write(str(i) + " ")
        f.write("|S|\n|Size|")
        for i in sizes:
            f.write(str(i) + " ")
        f.write("|S|\n|Speed|")
        for i in speeds:
            f.write(str(i) + " ")
        f.write("|D|\n|Movement|control_0, control_1, control_2, control_3, control_4, control_5|D|\n|Background|RGB ")
        for i in backs:
            f.write(str(i) + " ")
        f.write("|S|\n|Rotation|")
        for i in rotations:
            f.write(str(i) + " ")
        f.write("|D|\n")
        f.write("&nbsp;\n")
        f.write("##### Resolution: " + str(resolution[0]) + " x " + str(resolution[1]) + "\n")
        f.write("##### FPS: "+str(fps[0])+"\n")
        f.write("##### Group Number: " + str(args.num) + "\n")

    f.close()

    # Step 3: Generate the Group Videos
    if args.gpu:
        use_gpu()
    for i in range(0, args.num):
        p_gen(i, objs, colors, sizes, speeds, moves, backs, rotations, resolution, args.out_dir)

    # print("shana")


if __name__ == '__main__':
    user = os.path.join("/home", getpass.getuser())
    out = os.path.join("/home", user, "GSL_Video")
    parser = argparse.ArgumentParser(description='gsldataset')
    parser.add_argument('--out_dir', type=str, default=out,
                        help="Output dir for group data")
    parser.add_argument('--set_dir', type=str, default=out, help="setting file location")
    parser.add_argument('--num', type=int, default=1, help="number of groups")
    parser.add_argument('--gpu', type=bool, default=True, help="use gpu or not")
    argv = extract_args()
    args = parser.parse_args(argv)
    main(args)
    # use_gpu()
    # render(
    #            path="/home/shana/te.avi",
    #            object='sphere',
    #            color=(0.8,0.2,0.15,1.0),
    #            background=(0.25,0.9,0.8,1.0),
    #            sizes=8,
    #            speeds=5,
    #            movement=[(-15,-15,10),(-15,15,10),(15,15,10),(15,-15,10)],
    #            rot=[0,math.radians(-45)]
    #     )
