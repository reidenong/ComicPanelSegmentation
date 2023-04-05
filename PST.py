# From PanelSegmentationTool_(DEV).ipynb

"""## Dependencies and Helper Functions"""

import os
import shutil
import numpy as np
import math
import json
import imageio
import cv2
from tqdm import tqdm
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import canny
from skimage.morphology import dilation
from skimage.color import rgb2gray
from skimage.measure import label
from skimage.color import label2rgb
from skimage.transform import probabilistic_hough_line
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.backends.backend_pdf

# CLI Interface
# =====================================================================
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--filepath", default="Data/", help="directories where images are received as input, default is set as 'Data/'.")
parser.add_argument("--outputfilepath", default="Output/", help="directories where images are saved(or overwrites) as output, default is set as 'Output/'.")
parser.add_argument("--bboxjson", action="store_true", help="Choose not to save images, best used with --pdf to reduce unecessary processing")
parser.add_argument("--nosaveimage", action="store_false", help="Choose not to save images, best used with --pdf to reduce unecessary processing")
parser.add_argument("--pdf", action="store_true", help="Choose to generate a pdf displaying program decision making. Useful for troubleshooting.")
args = parser.parse_args()

"""### **Helper functions**"""

def protractor(line):       # in degrees
    p0, p1 = line
    x0,y0 = p0
    x1,y1 = p1
    if(x1-x0 == 0):     # to account for vertical lines
        angle_rad = np.arctan(np.inf)
    else:
        angle_rad = np.arctan((y1-y0)/(x1-x0))
    angle = angle_rad * 180 / math.pi
    return(angle)

# helper function to determine if 2 lines r inline
# ps: only works for angle_deviation = 0, ie. both lines are guaranteed parallel to xy axes
def in_line(line1, line2, orientation):  # orientation is 1 for horizontal, 0 for vertical
    # extracting individual points from line1, line2
    (p1r, p1c),(p2r, p2c) = line1
    (par, pac),(pbr, pbc) = line2

    if orientation: # horizontal
        if (p1r == par) and (p2r == pbr):
            return True
        else:
            return False
    else:    # vertical
        if (p1c == pac) and (p2c == pbc):
            return True
        else: 
            return False

def merge_line(line1, line2, orientation, distance):
    # extracting individual points from line1, line2 into sorted lists, row list and col list
    row_list = [0,0,0,0]
    col_list = [0,0,0,0]
    p1, p2 = line1
    (row_list[0], col_list[0]),(row_list[1], col_list[1]) = line1
    (row_list[2], col_list[2]),(row_list[3], col_list[3]) = line2
    row_list.sort()
    col_list.sort()
    
    # since both lines are inline and disconnected, row_list once sorted, indices 0,3 are the outer points whereas 1,2 are the inner points
    if orientation: # horizontal, so merge by column
        if (col_list[2] - col_list[1] <= distance):
            #print("merged horizontally")
            return ((row_list[0],col_list[0]),(row_list[3],col_list[3]))
        else:
            return line1
    else:       # vertical, merge by row
        if (row_list[2] - row_list[1] <= distance):
            #print("merged vertically")
            return ((row_list[0],col_list[0]),(row_list[3],col_list[3]))
        else:
            return line1

# length of an orthogonal line
def line_len(line): 
    (a,b),(c,d) = line
    return(max((abs(a-c)),(abs(b-d))))

# sliding window to take truncated mean of all close parallel lines
def parallel_merge(lst, merge_threshold):
    lst = list(set(lst))
    lst.sort()
    for i in range(len(lst)-1):
        if(abs(lst[i]-lst[i+1]) <= merge_threshold):
            lst[i] = lst[i+1] = int(np.mean([lst[i],lst[i+1]]))
    lst = list(set(lst))
    lst.sort()
    return lst

def new_parallel_merge(lst, merge_threshold):
    lst = list(set(lst))
    lst.sort()
    curr_xvalue = 0
    for i in range(len(lst)):
        if(abs(lst[i]-curr_xvalue) <= merge_threshold):
            lst[i] = curr_xvalue
        else:
            curr_xvalue = lst[i]
    lst = list(set(lst))
    lst.sort()
    return lst

line1 = ((811,1162),(811,761))
line2 = ((811,731),(811,463))
#print(in_line(line1, line2, 1))
#merge_line(line1, line2, 1, 300)
#line_len(line1)

def verticalcuts(lb, ub, vlist):
    lst = []
    for line in vlist:
        (x,a),(y,b) = line
        if a <= lb and b >= ub:
            lst += [x]
    lst = list(dict.fromkeys(lst)) # remove duplicates
    if lst == []:
        return False
    else: 
        return lst

"""## Image Segmentation for N images"""

# Static global variables
images = []                     # list of images
segment_lines = []              # list of segment lines per image
cutting_lines = []              # list of lines which image is cut by
bbox_dict = {}
filepath = args.filepath     # filepath containing dataset
output_filepath = args.outputfilepath  # filepath where cut images will be output
filenames = os.listdir(filepath)
N = len(filenames)

pdf_images = args.pdf
show_images = True
save_images = args.nosaveimage
save_bboxjson = args.bboxjson

# Static parameters
hough_threshold = 70            # in ???, empirically determined
hough_line_length = 60          # in ???            ||
hough_line_gap = 1              # in ???            ||
angle_deviation = 0             # acceptable deviation of line from orthogonal in degrees
distance = 999999999            # distance between 2 collinear lines before they r allowed to merge (in pixels)

# Dynamic parameters
width_border_factor = 0.1                   # fraction of the image that is acceptable as a width border, corresponds to width_border 60,70,200,100
height_border_factor = 0.09                 # fraction of the image that is acceptable as a height border, corresponds to height_border
hor_line_length_factor = 0.4                # fraction of the image where horizontal line is not accepted if its too short, corresponds to line_length_pm
ver_line_length_factor = 0.22               # fraction of the image where vertical line is not accepted if its too short, corresponds to ver_line_length
parallel_merge_dst_factor = 0.1             # fraction of the image that is acceptable for 2 parallel lines to merge, corresponds to parallel_merge_dst

# Matplotlib plotting tools
if show_images:
    fig, axes = plt.subplots(N, 4,figsize=(15, int(2.8*N)))    # 
    ax = axes.ravel()
    ax[0].set_title('Input image')
    ax[1].set_title('Probabilistic Hough Transformation')
    ax[2].set_title('Segment lines')
    ax[3].set_title('Cutting lines')

# Output Directory Preprocessing
if save_images:
    if os.path.exists(output_filepath):         # creates new directory for output filepath every time
        shutil.rmtree(output_filepath)
    os.makedirs(output_filepath)

# Main
for image_i in tqdm(range(N)):

    # Receiving Image input
    image_filepath = filepath + filenames[image_i]
    im = cv2.imread(image_filepath)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    images += [im]
    if show_images:
        axes[image_i, 0].imshow(images[image_i], cmap=cm.gray)

    # Initializing dynamic variables
    im_height, im_width, rgb_constant = images[image_i].shape
    width_border = int(width_border_factor * im_width)                                       # acceptable space from the border before a segment line is considered a cutting line 
    height_border = int(height_border_factor * im_height)                                    #                                ||
    hor_line_length = int(hor_line_length_factor *im_width)                                  # minimum length of line, post merge
    ver_line_length = int(ver_line_length_factor * im_height)
    parallel_merge_dst = int(parallel_merge_dst_factor * np.mean([im_width, im_height]))     # distance between 2 parallel cutting lines before they can merge
    # Alt static values for dynamic variables                            
    parallel_merge_dst = 50

    # Canny Edge Detection, followed by Probabilistic Hough Transformation to identify straight lines
    grayscale = rgb2gray(im)
    edges = canny(grayscale)
    lines = probabilistic_hough_line(edges, threshold=hough_threshold, line_length=hough_line_length, line_gap=hough_line_gap)
    if show_images:
        axes[image_i, 1].imshow(edges * 0)
    image_i_lines = []

    # Line filtering based on angle (orthogonals)
    lines = probabilistic_hough_line(edges, threshold=70, line_length=60,line_gap=1)    #line length?
    if show_images:
        axes[image_i, 1].imshow(edges * 0)
        axes[image_i, 1].imshow(images[image_i], cmap=cm.gray, alpha=0.25)
    for line in lines:
        line_angle = protractor(line)
        if( (abs(line_angle - 90) <= angle_deviation) or (abs(line_angle - 0) <= angle_deviation)):
            p0, p1 = line
            if show_images:
                axes[image_i, 1].plot((p0[0], p1[0]), (p0[1], p1[1]))
            image_i_lines += [line]
    if show_images:
        axes[image_i, 1].set_xlim((0, im.shape[1]))
        axes[image_i, 1].set_ylim((im.shape[0], 0))
    segment_lines += [image_i_lines]

    # Line Processing to convert Segment Lines into Cutting Lines
    #===========================================================================

    # Deriving vertical cutting lines 
    for segment_line in segment_lines[image_i]:              
        # iterate through first line to look at all cases
        p0, p1 = segment_line
        for segment_line2 in segment_lines[image_i]:
            if(in_line(segment_line, segment_line2, 1)):
                new_line = merge_line(segment_line, segment_line2, 1, distance)
                if (new_line != segment_line): #
                    if segment_line in segment_lines[image_i]:
                        segment_lines[image_i].remove(segment_line)
                    if segment_line2 in segment_lines[image_i]:
                        segment_lines[image_i].remove(segment_line2)
                    segment_lines[image_i].append(new_line)
                    #print("new line replaced")

    # Deriving horizontal cutting lines
    for segment_line in segment_lines[image_i]:              
        # iterate through first line to look at all cases
        p0, p1 = segment_line
        for segment_line2 in segment_lines[image_i]:
            if(in_line(segment_line, segment_line2, 0)):
                new_line = merge_line(segment_line, segment_line2, 0, distance)
                if (new_line != segment_line): #
                    if segment_line in segment_lines[image_i]:
                        segment_lines[image_i].remove(segment_line)
                    if segment_line2 in segment_lines[image_i]:
                        segment_lines[image_i].remove(segment_line2)
                    segment_lines[image_i].append(new_line)
                    #print("new line replaced")

    # Length post merge filtering, filtering cutting lines
    image_i_cutting_lines = []
    if show_images:
        axes[image_i,2].imshow(edges * 0)
        axes[image_i,2].imshow(images[image_i], cmap=cm.gray, alpha=0.25)
    for line in segment_lines[image_i]:
        if (protractor(line)<45 and line_len(line) >= hor_line_length) or (protractor(line)>45 and line_len(line) >= ver_line_length):
            #print(line_len(line))           # see all lines which are not cut
            p0, p1 = line
            if show_images:
                axes[image_i, 2].plot((p0[0], p1[0]), (p0[1], p1[1]))
            image_i_cutting_lines += [line]
    if show_images:
        axes[image_i, 2].set_xlim((0, im.shape[1]))
        axes[image_i, 2].set_ylim((im.shape[0], 0))
    cutting_lines += [image_i_cutting_lines]

    # Temporary containers for cutting line values
    horizontal_c_pos = []           # list of integers which correspond to the pixel row to extrapolate line
    horizontal_c_lines = []         # list of horizontal cutting lines
    vertical_c_pos = []             # list of tuples (column pos, lower bound, upper bound) of each vertical cutting line
    vertical_c_lines = []           # list of vertical cutting lines

    # Obtaining horizontals cutting lines positional row
    for cutting_line in cutting_lines[image_i]:
        if (protractor(cutting_line) < 45):           # not a hard 90 to account for the angle deviation function
            (p0c,p0r),(p1c,p1r) = cutting_line
            if(p0r > height_border) and (p0r < im_height - height_border):  # to remove all border lines for smoother end result image
                horizontal_c_pos += [p0r]
    
    # Extrapolating horizontal cutting lines to img width
    horizontal_c_pos = new_parallel_merge(horizontal_c_pos, parallel_merge_dst)
    for hpos in horizontal_c_pos:
        horizontal_c_lines += [((0,hpos),(im_width-1,hpos))]

    # Determining vertical cutting lines positional column
    for cutting_line in cutting_lines[image_i]:
        if (protractor(cutting_line) > 45):             # not a hard 90 to account for the angle deviation function
            (p0c,p0r),(p1c,p1r) = cutting_line
            if(p0c > width_border) and (p0c < im_width - width_border):  # to remove all border lines for smoother end result image
                lower_b, upper_b = min(p0r, p1r), max(p0r, p1r)
                vertical_c_pos += [(p0c,lower_b,upper_b)]

    # Changing Column-oriented upper and lower bounds for vertical lines
    horizontal_c_pos.insert(0,0)                #adding upper and lower bounds of horizontal lines as the edges of the border for vertical lines
    horizontal_c_pos.append(im_height-1)
    horizontal_c_pos.append(im_height-1)
    for vpos_itr in range(len(vertical_c_pos)):
        for itr in range(len(horizontal_c_pos)-1):
            if (vertical_c_pos[vpos_itr][1] >= horizontal_c_pos[itr]) and (vertical_c_pos[vpos_itr][1] < horizontal_c_pos[itr+1]):
                vertical_c_pos[vpos_itr] = (vertical_c_pos[vpos_itr][0], horizontal_c_pos[itr],vertical_c_pos[vpos_itr][2])
            if (vertical_c_pos[vpos_itr][2] <= horizontal_c_pos[itr+1]) and (vertical_c_pos[vpos_itr][2] > horizontal_c_pos[itr]):
                vertical_c_pos[vpos_itr] = (vertical_c_pos[vpos_itr][0], vertical_c_pos[vpos_itr][1], horizontal_c_pos[itr+1])

   # Extrapolating and merging vertical lines
    vertical_c_pos.sort(key=lambda a: a[0])
    if vertical_c_pos != []:
        c_pos_range = [vertical_c_pos[0][0]]
        #print(c_pos_range, np.mean(c_pos_range))
        for i in range(len(vertical_c_pos)):
            if(abs(vertical_c_pos[i][0] - np.mean(c_pos_range)) <= parallel_merge_dst):
                #c_pos_range += [vertical_c_pos[i][0]]
                new_c_pos = int(np.mean(c_pos_range))
                vertical_c_pos[i] = (new_c_pos, vertical_c_pos[i][1], vertical_c_pos[i][2])
            else:
                c_pos_range = [vertical_c_pos[i][0]]
    for vpos in vertical_c_pos:
        vertical_c_lines += [((vpos[0],vpos[1]),(vpos[0],vpos[2]))]

    # Displaying cutting lines
    if show_images:
        axes[image_i,3].imshow(edges * 0)
        axes[image_i, 3].imshow(images[image_i], cmap=cm.gray, alpha=0.6)
    for line in horizontal_c_lines:
        p0, p1 = line
        if show_images:
            axes[image_i, 3].plot((p0[0], p1[0]), (p0[1], p1[1]), color="lime")
    for line in vertical_c_lines:
        p0, p1 = line
        if show_images:
            axes[image_i, 3].plot((p0[0], p1[0]), (p0[1], p1[1]), color="lime")
    if show_images:
        axes[image_i, 3].set_xlim((0, im.shape[1]))
        axes[image_i, 3].set_ylim((im.shape[0], 0))


    # Cutting Images into output
    #===========================================================================
    output_dir = output_filepath  + filenames[image_i][:-4]
    
    if save_bboxjson:
        order_ctr = 0
        # Cutting Horizontal Sections
        for i in range(len(horizontal_c_pos)-2):       # CHECK FOR HEAVY BUGS
            img_h_section = im[horizontal_c_pos[i]:horizontal_c_pos[i+1],:]
            try:
                if verticalcuts(horizontal_c_pos[i], horizontal_c_pos[i+1], vertical_c_lines) != False:
                    cutting_points = verticalcuts(horizontal_c_pos[i], horizontal_c_pos[i+1], vertical_c_lines)
                    cutting_points.insert(0, 0)
                    cutting_points += [im_height-1]
                    #print(cutting_points, horizontal_c_pos[i], horizontal_c_pos[i+1])
                    for j in range(len(cutting_points)-1):
                        # print("hhvv", horizontal_c_pos[i],horizontal_c_pos[i+1],cutting_points[j],cutting_points[j+1])
                        img_v_section = im[horizontal_c_pos[i]:horizontal_c_pos[i+1],cutting_points[j]:cutting_points[j+1]]
                        bbox_dict[filenames[image_i][:-4] + "_" +  str(order_ctr)] = [horizontal_c_pos[i], horizontal_c_pos[i+1],cutting_points[j], cutting_points[j+1]]
                        #print("saved", filenames[image_i][:-4] + "_" + str(order_ctr) )
                        #cv2.imwrite(output_dir + "/" + str(order_ctr)+".jpg", cv2.cvtColor(img_v_section, cv2.COLOR_RGB2BGR))
                        order_ctr += 1
                else:
                    plt.imshow(img_h_section) # ???
                    #print("saved", filenames[image_i][:-4] + "_" + str(order_ctr) )
                    bbox_dict[filenames[image_i][:-4] + "_" + str(order_ctr)] = [horizontal_c_pos[i], horizontal_c_pos[i+1],1, im_width-1]
                    #cv2.imwrite(output_dir + "/" + str(order_ctr)+".jpg", cv2.cvtColor(img_h_section, cv2.COLOR_RGB2BGR))
                    order_ctr += 1
            except ValueError:
                pass
    
    if save_images:
        os.makedirs(output_dir)
        order_ctr = 0
        # Cutting Horizontal Sections
        for i in range(len(horizontal_c_pos)-2):       # CHECK FOR HEAVY BUGS
            img_h_section = im[horizontal_c_pos[i]:horizontal_c_pos[i+1],:]
            try:
                if verticalcuts(horizontal_c_pos[i], horizontal_c_pos[i+1], vertical_c_lines) != False:
                    cutting_points = verticalcuts(horizontal_c_pos[i], horizontal_c_pos[i+1], vertical_c_lines)
                    cutting_points.insert(0, 0)
                    cutting_points += [im_height-1]
                    #print(cutting_points, horizontal_c_pos[i], horizontal_c_pos[i+1])
                    for j in range(len(cutting_points)-1):
                        # print("hhvv", horizontal_c_pos[i],horizontal_c_pos[i+1],cutting_points[j],cutting_points[j+1])
                        img_v_section = im[horizontal_c_pos[i]:horizontal_c_pos[i+1],cutting_points[j]:cutting_points[j+1]]
                        cv2.imwrite(output_dir + "/" + str(order_ctr)+".jpg", cv2.cvtColor(img_v_section, cv2.COLOR_RGB2BGR))
                        order_ctr += 1
                else:
                    plt.imshow(img_h_section) # ???
                    cv2.imwrite(output_dir + "/" + str(order_ctr)+".jpg", cv2.cvtColor(img_h_section, cv2.COLOR_RGB2BGR))
                    order_ctr += 1
            except ValueError:
                pass
        
        

# Outprocessing
# Show plots

if save_bboxjson:
    print("Saving BBox json")
    with open("BBox.json", "w") as fp:
        json.dump(bbox_dict, fp)

if show_images:
    for a in ax:
        a.set_axis_off()
    plt.show()

if pdf_images:
    fig.savefig('results.pdf', format='pdf', dpi=600)
