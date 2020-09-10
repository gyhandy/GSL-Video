from blender_for_gsl import *
import os
import argparse
import shutil
import getpass
import json
import random
import sys

# render(
#            path="/home/shana/te.avi",
#            obj='cube',
#            color=(0.8,0.2,0.15,1.0),
#            bg=(0.25,0.9,0.8,1.0),
#            size=2,
#            speed=1,
#            move=gen_moves()[0],
#            rot=[1,math.radians(45)]
#     )
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

def u_getitem(js, box):
    """
    :param js: json file
    :param box: list name use to store value
    :return:
    """
    for key, value in js.items():
        box.append(value)


def u_diffite(set, item):
    """

    :param set:
    :param item:
    :return:
    """
    if item == set[0]:
        return False
    return True


def u_getun(set, ori):
    """
    :param set:  Group value
    :param ori: Possible values
    :return:
    """
    temp = random.choice(ori)
    while True:
        # print(temp)
        if u_diffite(set, temp):
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

    # Step 3: Generate doc
    with open(doc, 'w') as f:
        f.write("Group" + str(num) + "\n")
        f.write(format("Name", '<10') + "\t")
        f.write(format("Object", '<10') + "\t")
        f.write(format("Color", '<40') + "\t")
        f.write(format("Size", '<20') + "\t")
        f.write(format("Speed", '<20') + "\t")
        f.write(format("Movement", '<10') + "\t")
        f.write(format("Background", '<10') + "\t")
        f.write(format("Rotation", '<10') + "\t")
        f.write(format("Common", '<10') + "\n")
        for i in range(0, 8):
            f.write(format(name[i], '<10') + "\t")
            f.write(format(obj[i], '<10') + "\t")
            f.write(format(col[i], '<40') + "\t")
            f.write(format(str(size[i]), '<20') + "\t")
            f.write(format(str(speed[i]), '<20') + "\t")
            f.write(format(move[i], '<10') + "\t")
            f.write(format(back[i], '<10') + "\t")
            f.write(format(rotation[i], '<10') + "\t")
            f.write(format(common[i], '<10') + "\n")
    print("Finish generating doc")

    # Step 4: Render the videos
    for i in range(0, 8):
        render(
            path=os.path.join(video_dir, name[i]),
            object=obj[i],
            color=col[i],
            background=back[i],
            sizes=size[i],
            speeds=speed[i],
            movement=move[i],
            rot=rotation[i],
            x=resolution[0],
            y=resolution[1]
        )


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

    f.close()
    print("Finish reading setting")

    # Step 2: Generate Readme for groups data
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
                "|Rotation|base_scene, green, blue, red, purple, yellow|D|\n"
                )
        f.write("&nbsp;\n")
        f.write("##### Resolution: " + args.resolution + "\n")
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
