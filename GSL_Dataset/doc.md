#### Basic info

1. Ubuntu18.04
2. *Italic* indicates shell commands

#### Install Blender 2.90

1. Download : https://www.blender.org/download/

2. Extract at $BLENDER
3. *gedit ~/.bashrc*
4. add "export PATH=$BLENDER:$PATH" 
5. *source ~./bashrc*



#### Install GSL_Video dataset

1. git clone https://github.com/gyhandy/GSL-Video at $GSL
2. *cd $GSL*
3. *echo $PWD/GSL_Dataset >> $BLENDER/$VERSION/python/lib/$PYTHONVERSION/site-packages/gsl.pth*


#### Test Installation
1. *cd $GSL/GSL_Dataset*

2. *blender --background --python generate.py*

  


#### User Guide for generating GSL Videos 
5. Run generate.py with the following parameters:
	'--out_dir'  :       Output location for group datas  
	'--set_dir'  :        Location of setting.json  
	'--num'	  :	     Number of groups that you want to generate  
	'--gpu'	   :  		Use GPU or not, default True  

###### Todo

- Attribute Design

