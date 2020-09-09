from blender_for_gsl import *

# Render video
render(
#    path="F:\\MachineLearning\\GSL\\dataset\\test_cube2_bg0_random.avi", 
    obj_name='torus', 
    color='Toon BSDF', 
    bg='Glossy BSDF', 
    move=gen_moves()[0]
)

#def load_textures():
#    pass


#def add_material(obj=None, name=None):
#    """ Add an existing material to an object. """
#    if name in D.materials.keys():
#        obj.data.materials.clear()
#        obj.data.materials.append(D.materials[name])
#    else:
#        print("Wrong material name. ")
#        

#def add_texture(obj, texture):
#    if texture in D.textures.keys():
#        obj.data.textures.clear()
#        obj.data.textures.append(D.materials[name])
#    else:
#        print("Wrong material name. ")