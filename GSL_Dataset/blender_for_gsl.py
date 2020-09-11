"""
Put this file in the /path/to/blender/your-version/scripts/addons to work.
"""

import os
import math
from random import randint, choice

import bpy
from bpy import context as C
from bpy import data as D
from bpy import ops as OPS


def use_gpu():
    cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
    cycles_prefs.compute_device_type = 'CUDA'
    bpy.context.scene.cycles.device = 'GPU'
    for devices in bpy.context.preferences.addons['cycles'].preferences.get_devices():
        for d in devices:
            d.use = True
            if d.type == 'CPU':
                d.use = False
            print("Device '{}' type {} : {}".format(d.name, d.type, d.use))


def clear_scene():
    """ Clear all objects and related data. """
    for camera in D.cameras:
        D.cameras.remove(camera)
    for light in D.lights:
        D.lights.remove(light)
    for action in D.actions:
        D.actions.remove(action)
    for mesh in D.meshes:
        D.meshes.remove(mesh)
    for material in D.materials:
        D.materials.remove(material)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def add_light(name='sun', type='SUN', size=1, loc=(0, 0, 0), rot=(0, 0, 0), energy=4):
    """ Add lighting. """
    OPS.object.light_add(type=type,
                         radius=size,
                         align='WORLD',
                         location=loc)
    C.object.name = name
    D.objects[name].data.energy = energy
    D.objects[name].rotation_euler = rot


def add_camera(name='camera', loc=(0, 0, 0), rot=(0, 0, 0)):
    """ Add a camera. """
    OPS.object.camera_add(enter_editmode=False,
                          align='VIEW',
                          location=loc,
                          rotation=rot)
    C.object.name = name
    C.scene.camera = C.object


def load_materials(mat_path=None):
    """
    Load all available materials.
    Hardcoded here, but will be saved in files.
    """
    #   materials = json.load(mat_path)
    materials = {
        'Glossy BSDF': {
            'type': 'ShaderNodeBsdfGlossy',
            'color': (0.257, 0.905, 0.779, 1.000),
        },
        'Toon BSDF': {
            'type': 'ShaderNodeBsdfToon',
            'color': (0.232, 0.142, 0.800, 1.000),
        },
        'Glass BSDF': {
            'type': 'ShaderNodeBsdfGlass',
            'color': (0.985, 0.108, 0.559, 1.000),
        },
    }
    for name in materials.keys():
        if name not in D.materials.keys():
            new_mat = D.materials.new(name=name)
            new_mat.use_nodes = True
            nodes = new_mat.node_tree.nodes
            mat_out = nodes.get('Material Output')
            mat_in = nodes.new(type=materials[name]['type'])
            mat_in.inputs[0].default_value = materials[name]['color']
            links = new_mat.node_tree.links
            new_link = links.new(mat_in.outputs[0], mat_out.inputs[0])


def gen_moves(space=(10, 10, 10), center=(0, 0, 6), n=3, points=5, size=1):
    """ Generate movement points in a certain space. """
    moves = []
    for i in range(n):
        temp = []
        for j in range(points):
            temp.append((randint(center[0] - space[0] / 2, center[0] + space[0] / 2),
                         randint(center[1] - space[1] / 2, center[1] + space[1] / 2),
                         #                         randint(center[2]-space[2]/2, center[2]+space[2]/2)
                         size * 2))
        moves.append(temp)
    return moves


def add_material(obj=None, name=None):
    """ Add an existing material to an object. """
    if name in D.materials.keys():
        obj.data.materials.clear()
        obj.data.materials.append(D.materials[name])
    else:
        print("Wrong material name. ")


def set_color(obj=None, color=None):
    """ Set color for an object. """
    mat = obj.material_slots[0].material
    mat_name = mat.name
    mat.node_tree.nodes[mat_name].inputs[0].default_value = color


def add_plane(name='plane', size=200, color=(), loc=(0, 0, 0), rot=(0, 0, 0)):
    """ Add a plane. """
    OPS.mesh.primitive_plane_add(size=size,
                                 calc_uvs=True,
                                 enter_editmode=False,
                                 align='WORLD',
                                 location=loc,
                                 rotation=rot)
    C.object.name = name
    add_material(C.object, 'Glossy BSDF')
    set_color(C.object, color)


def add_cube(name='cube', size=2, color=(), loc=(0, 0, 0)):
    """ Add a cube. """
    OPS.mesh.primitive_cube_add(size=size,
                                calc_uvs=True,
                                enter_editmode=False,
                                align='WORLD',
                                location=loc,
                                rotation=(0, 0, 0))
    C.object.name = name
    add_material(C.object, 'Toon BSDF')
    set_color(C.object, color)


def add_uv_sphere(name='sphere', size=2, color=(), loc=(2, 2, 2)):
    """ Add a sphere. """
    OPS.mesh.primitive_uv_sphere_add(segments=64,
                                     ring_count=32,
                                     radius=size,
                                     calc_uvs=True,
                                     enter_editmode=False,
                                     align='WORLD',
                                     location=loc,
                                     rotation=(0, 0, 0))
    OPS.object.modifier_add(type='SUBSURF')
    C.object.modifiers["Subdivision"].levels = 3
    C.object.modifiers["Subdivision"].render_levels = 3
    OPS.object.shade_smooth()
    C.object.name = name
    add_material(C.object, 'Toon BSDF')
    set_color(C.object, color)


def add_cylinder(name='cylinder', size=1, color=(), loc=(1, 1, 1)):
    """ Add a cylinder. """
    OPS.mesh.primitive_cylinder_add(vertices=64,
                                    radius=size,
                                    depth=size * 2,
                                    end_fill_type='NGON',
                                    calc_uvs=True,
                                    enter_editmode=False,
                                    align='WORLD',
                                    location=loc,
                                    rotation=(0, 0, 0))
    C.object.name = name
    add_material(C.object, 'Toon BSDF')
    set_color(C.object, color)


def add_torus(name='torus', size=1, color=(), loc=(0, 0, 0), rot=(0, 0, 0)):
    """ Add a torus. """
    bpy.ops.mesh.primitive_torus_add(align='WORLD',
                                     location=loc,
                                     rotation=rot,
                                     major_radius=size,
                                     minor_radius=size * 0.25,
                                     abso_major_rad=1.25,
                                     abso_minor_rad=0.75)
    OPS.object.modifier_add(type='SUBSURF')
    C.object.modifiers['Subdivision'].render_levels = 3
    C.object.modifiers['Subdivision'].levels = 3
    OPS.object.shade_smooth()
    C.object.name = name
    add_material(C.object, 'Toon BSDF')
    set_color(C.object, color)


def add_cone(name='torus', size=1, color=(), loc=(0, 0, 0)):
    """ Add a cone. """
    bpy.ops.mesh.primitive_cone_add(radius1=size,
                                    radius2=0,
                                    depth=size * 2,
                                    enter_editmode=False,
                                    align='WORLD',
                                    location=loc)
    OPS.object.shade_smooth()
    C.object.name = name
    add_material(C.object, 'Toon BSDF')
    set_color(C.object, color)


def add_movement(obj=None, speed=1, move=[], rot=[]):
    """ Add dynamics and link to key frames. """
    frame = 0
    for pos in move:
        C.scene.frame_set(frame)
        obj.location = pos
        obj.rotation_euler[rot[0]] += rot[1]
        obj.keyframe_insert(data_path='location', index=-1)
        obj.keyframe_insert(data_path='rotation_euler', index=-1)
        frame += 30 // speed


def prepare_scene(color=None, length=None, speed=None, fps=30, rx=512, ry=512):
    """ Prepare scene. """
    clear_scene()
    load_materials()

    C.scene.render.engine = 'CYCLES'
    C.scene.cycles.device = 'GPU'
    C.scene.render.resolution_x = rx
    C.scene.render.resolution_y = ry

    C.scene.cycles.samples = 256
    C.scene.render.tile_x = 512
    C.scene.render.tile_y = 512
    C.view_layer.cycles.use_denoising = True

    C.scene.render.fps = fps
    C.scene.render.image_settings.file_format = 'FFMPEG'
    #    C.scene.render.ffmpeg.format = 'MPEG4'  # for .mp4
    C.scene.render.ffmpeg.format = 'AVI'  # for .avi
    C.scene.frame_end = fps * length // speed

    #    add_light('sun', 'SUN', 2, loc=(0,-50,50),
    #              rot=(math.radians(45),0,0), energy=0.5)
    add_light('area', 'AREA', 100, loc=(0, 0, 50), energy=30000)
    add_camera('camera', loc=(50, -50, 50),
               rot=(math.radians(60), 0, math.radians(45)))
    add_plane('plane1', 100, color, loc=(0, 0, 0), rot=(0, 0, 0))
    add_plane('plane2', 100, color, loc=(0, 50, 0), rot=(math.radians(90), 0, 0))
    add_plane('plane3', 100, color, loc=(50, 0, 0), rot=(0, math.radians(90), 0))
    add_plane('plane4', 100, color, loc=(-50, 0, 0), rot=(0, math.radians(90), 0))


def render(path=None,
           object='',
           color=(),
           background=(),
           sizes=1,
           speeds=1,
           movement=[],
           rot=[0, math.radians(45)],
           x=512,
           y=512):
    """ Render a video. 
    Params:
        path: str, path to the rendered video, 
        obj: str, object name, 
        color: str, object color, 
        bg: str, background color, 
        size: int, object size (uniformed for different objects), 
        speed: int, object move speed, 
        move: list of 3d position tuples, 
        rot: tuple of rotation axis and degree.
    Returns: None
    """
    prepare_scene(color=background, length=len(movement) - 1, speed=speeds, fps=30, rx=x, ry=y)


    if object == 'cube':
        add_cube('cube', size=sizes, loc=movement[0], color=color)
        add_movement(obj=D.objects['cube'], speed=speeds, move=movement, rot=rot)
    elif object == 'sphere':
        add_uv_sphere('sphere', size=sizes, loc=movement[0], color=color)
        add_movement(obj=D.objects['sphere'], speed=speeds, move=movement, rot=rot)
    elif object == 'cylinder':
        add_cylinder('cylinder', size=sizes, loc=movement[0], color=color)
        add_movement(obj=D.objects['cylinder'], speed=speeds, move=movement, rot=rot)
    elif object == 'torus':
        add_torus('torus', size=sizes, loc=movement[0], color=color)
        add_movement(obj=D.objects['torus'], speed=speeds, move=movement, rot=rot)
    elif object == 'cone':
        add_cone('cone', size=sizes, loc=movement[0], color=color)
        add_movement(obj=D.objects['cone'], speed=speeds, move=movement, rot=rot)

    if path:  # render and save video
        C.scene.render.filepath = path
        OPS.render.render(animation=True)
