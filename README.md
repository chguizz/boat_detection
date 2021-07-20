# Boat detection

This project focuses on detecting boats with different points of view, sizes, shapes, types, background elements and illumination conditions.

## Basic Usage
In the first modality the application produces in output the image with green bounding boxes for all regions with boats. Above each box a score is present and represent the probability of that region to be a boat. As command line argument, give only the image path:

<code> image_path </code>

In the second modality in addition of what happens for the first one, in output is also produces the comaprison with the ground truth (Note: the assumption is that the ground truth is stored with same format as in [dataset]( https://drive.google.com/file/d/1XkVfXNjq_KMANKUBSlbpPrlMNe9cMhKk/view?usp=sharing), for example **boat:264;371;342;362;**). As command line arguments, give the image path and ground truth path (**.txt file**):

<code> image_path ground_truth_path </code>