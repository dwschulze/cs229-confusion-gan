"""
Testing Script
Author: Arshmeet
"""

import os 
from options.test_options import TestOptions 
from data import create_dataset 
from models import create_model
from util import html, util
from util.visualizer import save_images
from PIL import Image

# Get the testing options, set up the model accordingly
opt = TestOptions().parse() 
testing_images = create_dataset(opt)
model = create_model(opt)
model.setup(opt)

# create output directory and image results 
html_output = html.HTML(opt.results_dir, "Test Results") 
os.makedirs(os.path.join(opt.results_dir, "images"), exist_ok = True)

# put model in evaluation mode 
model.eval()

for i, test_dict in enumerate(testing_images): 
    # set up model
    model.set_input(test_dict)
    model.test()
    # get the dictionary of generated images and their paths 
    generated_images = model.get_current_visuals() 
    generated_images_paths = model.get_image_paths() # since batch = 1, should only have 1
    # save test output images to the html file
    save_images(html_output, 
                generated_images, 
                generated_images_paths,
                aspect_ratio = opt.aspect_ratio, 
                width = opt.display_winsize
               )
    # save test output to folder
    for img_type, img_tensor in generated_images.items(): 
        img = util.tensor2im(img_tensor) 
        filename = os.path.basename(generated_images_paths[0])
        output_path = os.path.join(opt.results_dir, "images", f"{filename}_{img_type}.png")
        Image.fromarray(img).save(output_path)
        
html_output.save()