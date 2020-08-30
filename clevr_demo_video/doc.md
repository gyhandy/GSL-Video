#### Basic info

1. Ubuntu18.04
2. *Italic* indicates shell commands

#### Install Blender 2.78c

1. Download : https://download.blender.org/release/Blender2.78/

2. Extract at $BLENDER
3. *gedit ~/.bashrc*
4. add "export PATH=$BLENDER:$PATH" 
5. *source ~./bashrc*



#### Install CLEVR IMAGE_GENERATION

1. *git clone https://github.com/facebookresearch/clevr-dataset-gen at $CLEVR*
2. *cd $CLEVR*
3. *echo $PWD/image_generation >> $BLENDER/$VERSION/python/lib/$PYTHONVERSION/site-packages/clevr.pth*


#### Test Installation
1. *cd $CLEVR/image_generation*
2. *blender --background --python render_images.py -- --num_images 1 --use_gpu 1*  
// If you use cuda higher than 8.0, it may takes more time on the first run


#### User Guide for generating GSL Videos 
1. Replace the original render_images.py 
2. Copy the content of background folder into $CLEVR/image_generation/data
3. Copy the content of movement folder into $CLEVR/image_generation/data
4. Copy the content of object folder into $CLEVR/image_generation/data/shapes
5. Run generate.py with the following parameters:
	'--out_dir' :  		Output location for group datas
	'--cle_dir'  :  	   Location of render_images.py
	'--set_dir'  :  	   Location of setting.json
	'--num'	   :	     Number of groups that you want to generate
	'--keep_temp': 	     Keep the images used to generate video or not, default False
	

###### Todo

- [] Color difference calculation

- [] Speed Design

- [x] Format of description file

- [] Randomization of movements

- [] More testing

- [] Object visible debug