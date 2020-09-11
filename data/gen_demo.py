from blender_for_gsl import *

render(
       path='/home/shana/te.avi', 
       obj='cube', 
       color=(0.8,0.2,0.15,1.0), 
       bg=(0.25,0.9,0.8,1.0), 
       size=2, 
       speed=1, 
       move=gen_moves()[0], 
       rot=[1,math.radians(45)]
)
