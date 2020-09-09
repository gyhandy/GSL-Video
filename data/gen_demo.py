from blender_for_gsl import *

# Render video
render(
#    path="F:\\MachineLearning\\GSL\\dataset\\test_cube2_bg0_random.avi", 
    obj_name='cube', 
    color='Toon BSDF', 
    bg_color='Glossy BSDF', 
    move=gen_moves()[0]
)