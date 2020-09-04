import os
import subprocess
import argparse
import shutil
import getpass
import json
import random


def u_mkdir(path):
    """
    :param path:
    :return: null
    """
    if not os.path.exists(path):
        os.makedirs(path)


def u_getitem(js, box):
    """
    :param js: json file
    :param box: list name use to store value
    :return:
    """
    for key, value in js.items():
        box.append(value)


def u_rgba():
    """
    generate random color
    :return:  a str
    """
    r = format(random.uniform(0, 1), '.6f')
    g = format(random.uniform(0, 1), '.6f')
    b = format(random.uniform(0, 1), '.6f')
    a = format(1, '.6f')
    re = str(r) + ", " + str(g) + ", " + str(b) + ", " + str(a)
    return re


def u_diffco(cols, col):
    """
    judge if the color is similar with exists color
    :param cols: Color set
    :param col: New color
    :return: True if different
    """
    if col in cols:
        return False
    return True


def u_diffite(set, item):
    """

    :param set:
    :param item:
    :return:
    """
    if item in set:
        return False
    return True


def u_getun(set, max, min, ori, flag):
    """

    :param set:
    :param max:
    :param min:
    :param ori:
    :param flag:
    :return:
    """
    c = 10
    if flag == 0:  # range
        temp = random.uniform(min, max)
    elif flag == 1:  # choice
        temp = random.choice(ori)
    else:  # color
        temp = u_rgba()
    while True:
        # print(temp)
        if u_diffite(set, temp):
            break
        if flag == 0:  # range
            temp = random.uniform(min, max)
        elif flag == 1:  # choice
            temp = random.choice(ori)
        else:  # color
            temp = u_rgba()
    return temp


def u_mkdir(path):
    """
    :param path:
    :return: null
    """
    if not os.path.exists(path):
        os.makedirs(path)


def p_gen(num, objs, sizes, speeds, moves, backs, out_dir, cle_dir, temp, width, height):
    print(cle_dir)
    # Step 1: generate setting file name and store location
    video_dir = os.path.join(out_dir, "Group" + str(num))
    u_mkdir(video_dir)
    doc = os.path.join(video_dir, "Description")
    print(objs)
    print(sizes)
    print(speeds)
    print(moves)
    print(backs)
    # Step 2: generate setting for each video
    print("Start generating setting for group data")
    lens = len(objs)
    # objs = copy.deepcopy(objs_o)
    # moves = copy.deepcopy(moves_o)
    # backs = copy.deepcopy(backs_o)
    obj = []
    col = []
    size = []
    speed = []
    move = []
    back = []
    name = []
    common = ["/", "Object", "Color", "Size", "Speed", "Movement", "Background"]

    for i in range(0, lens + 1):
        if i == 0:
            print("center")
            name.append("center.avi")
            # o = random.choice(objs)
            # obj.append(o)
            # c = u_rgba()
            # col.append(c)
            # se = random.uniform(sizes[1],sizes[0])
            # size.append(se)
            # sd = random.uniform(speeds[1],speeds[0])
            # speed.append(sd)
            # m = random.choice(moves)
            # move.append(m)
            # b = random.choice(backs)
            # back.append(b)
            obj.append(u_getun(obj, 0, 0, objs, 1))
            col.append(u_getun(col, 0, 0, 0, 2))
            size.append(u_getun(size, sizes[0], sizes[1], 0, 0))
            speed.append(u_getun(speed, speeds[0], speeds[1], 0, 0))
            move.append(u_getun(move, 0, 0, moves, 1))
            back.append(u_getun(back, 0, 0, backs, 1))
        else:
            name.append("%06d.avi" % i)
            if i == 1:
                print("01")
                # Common Object
                obj.append(obj[0])
                col.append(u_getun(col, 0, 0, 0, 2))
                size.append(u_getun(size, sizes[0], sizes[1], 0, 0))
                speed.append(u_getun(speed, speeds[0], speeds[1], 0, 0))
                move.append(u_getun(move, 0, 0, moves, 1))
                back.append(u_getun(back, 0, 0, backs, 1))
            elif i == 2:
                print("02")
                # Common Color
                obj.append(u_getun(obj, 0, 0, objs, 1))
                col.append(col[0])
                size.append(u_getun(size, sizes[0], sizes[1], 0, 0))
                speed.append(u_getun(speed, speeds[0], speeds[1], 0, 0))
                move.append(u_getun(move, 0, 0, moves, 1))
                back.append(u_getun(back, 0, 0, backs, 1))
            elif i == 3:
                print("03")
                # Common Size
                obj.append(u_getun(obj, 0, 0, objs, 1))
                col.append(u_getun(col, 0, 0, 0, 2))
                size.append(size[0])
                speed.append(u_getun(speed, speeds[0], speeds[1], 0, 0))
                move.append(u_getun(move, 0, 0, moves, 1))
                back.append(u_getun(back, 0, 0, backs, 1))
            elif i == 4:
                print("04")
                # Common Speed
                obj.append(u_getun(obj, 0, 0, objs, 1))
                col.append(u_getun(col, 0, 0, 0, 2))
                size.append(u_getun(size, sizes[0], sizes[1], 0, 0))
                speed.append(speed[0])
                move.append(u_getun(move, 0, 0, moves, 1))
                back.append(u_getun(back, 0, 0, backs, 1))
            elif i == 5:
                print("05")
                # Common Movement
                obj.append(u_getun(obj, 0, 0, objs, 1))
                col.append(u_getun(col, 0, 0, 0, 2))
                size.append(u_getun(size, sizes[0], sizes[1], 0, 0))
                speed.append(u_getun(speed, speeds[0], speeds[1], 0, 0))
                move.append(move[0])
                back.append(u_getun(back, 0, 0, backs, 1))
            else:
                print("06")
                ## Common Background
                obj.append(u_getun(obj, 0, 0, objs, 1))
                col.append(u_getun(col, 0, 0, 0, 2))
                size.append(u_getun(size, sizes[0], sizes[1], 0, 0))
                speed.append(u_getun(speed, speeds[0], speeds[1], 0, 0))
                move.append(u_getun(move, 0, 0, moves, 1))
                back.append(back[0])
        # print(obj)
        # print(col)
        # print(size)
        # print(speed)
        # print(move)
        # print(back)
        # print(name)

    # Step 3: generate doc
    with open(doc, 'w') as f:
        f.write("Group" + str(num) + "\n")
        f.write(format("Name", '<10') + "\t")
        f.write(format("Object", '<10') + "\t")
        f.write(format("Color", '<40') + "\t")
        f.write(format("Size", '<20') + "\t")
        f.write(format("Speed", '<20') + "\t")
        f.write(format("Movement", '<10') + "\t")
        f.write(format("Background", '<10') + "\t")
        f.write(format("Common", '<10') + "\n")
        for i in range(0, lens + 1):
            f.write(format(name[i], '<10') + "\t")
            f.write(format(obj[i], '<10') + "\t")
            f.write(format(col[i], '<40') + "\t")
            f.write(format(str(size[i]), '<20') + "\t")
            f.write(format(str(speed[i]), '<20') + "\t")
            f.write(format(move[i], '<10') + "\t")
            f.write(format(back[i], '<10') + "\t")
            f.write(format(common[i], '<10') + "\n")
    print("Finish generating doc")

    # Step 4: generate videos
    for i in range(0, lens + 1):
        temp_frame = os.path.join(video_dir, "images", name[i])
        scene = "data/" + str(back[i]) + ".blend"
        # generate frames
        print("Start generate frames for " + name[i])
        subprocess.run([
            "blender",
            "--background",
            "--python", os.path.join(cle_dir, "render_images.py"),
            "--",
            "--use_gpu", "1",
            "--output_image_dir", temp_frame,
            "--Objname", obj[i],
            "--Objco", col[i],
            "--Objsize", str(size[i]),
            "--max_len", str(speed[i]),
            "--control", move[i],
            "--base_scene_blendfile", scene,
            "--width", width,
            "--height", height
        ],
            cwd=cle_dir)
        # generate videos
        print("Start generate videos for" + name[i])
        subprocess.run([
            "ffmpeg",
            "-f", "image2",
            "-i", os.path.join(temp_frame, "CLEVR_new_%06d.png"),
            os.path.join(video_dir, name[i])
        ]
        )
        # delete temp files
    if not temp:
        shutil.rmtree(os.path.join(video_dir, "images"))


def main(args):
    # print(args.out_dir)
    # print(args.cle_dir)
    # print(args.set_dir)  # print(args.num)

    # Step 1: Reading Setting.json to get basic info
    objs = []
    sizes = []
    speeds = []
    moves = []
    backs = []
    resolution = args.resolution.split("x")
    width = resolution[0]
    height = resolution[1]
    with open(os.path.join(args.set_dir, "Setting.json"), 'r') as f:
        setting = json.load(f)
        obj = setting['Object']
        u_getitem(obj, objs)
        size = setting['Size']
        u_getitem(size, sizes)
        speed = setting['Speed']
        u_getitem(speed, speeds)
        move = setting['Movement']
        u_getitem(move, moves)
        back = setting['Background']
        u_getitem(back, backs)
    f.close()
    print("Finish reading setting")

    # Step2: Generate basic file for the groups
    with open(os.path.join(args.out_dir, "Groupinfo.md"), 'w') as f:
        f.write("#### Attribute\n")
        f.write("|Attribute|Possible Values|Dynamic/Static|\n"
                "|---|---|---|\n"
                "|Object|cube, sphere, cylinder, icosphere, cone, torus|S|\n"
                "|Color|RGB[ [0,0,0] , [255,255,255] ]|S|\n"
                "|Size|[0.2,0.7]]|S|\n"
                "|Speed|[0.1,0.5]]|D|\n"
                "|Movement|controls, control_1, control_2, control_3, control_4, control_5|D|\n"
                "|Background|base_scene, green, blue, red, purple, yellow|S|\n"
                )
        f.write("&nbsp;\n")
        f.write("##### Resolution: " + args.resolution+"\n")
        f.write("##### Group Number: " + str(args.num)+"\n")
    f.close()

    # Step 3: start generate group data
    print("Starting generating")
    for i in range(0, args.num):
        print("Group" + str(i) + ":")
        p_gen(i, objs, sizes, speeds, moves, backs, args.out_dir, args.cle_dir, args.keep_temp, width, height)


if __name__ == '__main__':
    user = os.path.join("/home", getpass.getuser())
    de_vle = os.path.join("/home", user, "clevr-dataset-gen/image_generation")
    parser = argparse.ArgumentParser(description='plvlvideo')
    parser.add_argument('--out_dir', type=str, default=os.path.join("/home", user, "GSL_Clevr_Video"),
                        help="Output dir for group data")
    parser.add_argument('--cle_dir', type=str, default=de_vle, help="root dir for render_images.py")
    parser.add_argument('--set_dir', type=str, default=de_vle, help="setting file location")
    parser.add_argument('--num', type=int, default=1, help="number of groups")
    parser.add_argument('--keep_temp', type=bool, default=False, help="Keep the temp frames or not")
    parser.add_argument('--resolution', type=str, default="512x512", help="resultion of the generate video")
    args = parser.parse_args()
    main(args)
