#%%
import os
from scipy import ndimage 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from findpeaks import findpeaks
from torchvision.ops import nms
from PIL import Image
from NMS_implementations import *
#%%
def region_of_interest(image,points):
    '''
    defines region of interest of an image by creating a mask and multiplying it by the original image
    '''
    mask=np.zeros_like(image)
    cv2.fillPoly(mask, points, 255) 
    masked_image = cv2.bitwise_and(image, mask) 
    save_and_show_image(masked_image,"roiEdges.png")

    return masked_image
def hough(edges_image,angle_step_size=1):
    '''
    🐕 🐕‍🦺 transform
    '''
    angles_count=int(1+180/angle_step_size)#number of angles, depends on step size of the image.
    angles=np.linspace(start=0,stop=np.pi, endpoint=True, num=angles_count)

    edge_points=np.array(np.where(edges_image==255)).T
    𝜌_axis_length = calc_length(*edges_image.shape)
    H=np.zeros((2*𝜌_axis_length,angles_count,))

    𝜌s = calc_𝜌s(angles, edge_points)
    𝜌s_as_indexes=np.round(𝜌s).astype(int)+𝜌_axis_length
    Fill_accumolator(H,𝜌s_as_indexes,angles_count)
    save_and_show_image(H, "H_space.png")
    return H


def calc_𝜌s(angles, edge_points):
    sines=np.sin(angles)
    cosines=np.cos(angles)
    sinesmul=np.array([np.multiply(i, sines) for i in edge_points[:, 1]])
    cosmul=np.array([np.multiply(i, cosines) for i in edge_points[:, 0]])
    𝜌s=np.add(sinesmul,  cosmul)
    return 𝜌s

def calc_length(xlen, ylen):
    y_axis_length=int(np.ceil(np.sqrt(xlen*xlen+ylen*ylen)))
    return y_axis_length
#%%
def Fill_accumolator(H,𝜌s,degrees_count):
    angles=np.array(list(range(0, degrees_count))*len(𝜌s))
    𝜌s_all_in_one=np.hstack(𝜌s)
    psudoAngels=angles+1#angles shifted by 1 .. this is not nessesry but we do it anyway
    psudoAngels[np.where(psudoAngels==degrees_count)]=0
    psudoAngels2=angles-1
    psudoAngels2[np.where(psudoAngels==-1)]=degrees_count-1

    np.add.at(H,(-𝜌s_all_in_one,angles ),1)

    np.add.at(H,(-𝜌s_all_in_one+1,angles ),1)
    np.add.at(H,(-𝜌s_all_in_one-1,angles ),1)

    np.add.at(H,(-𝜌s_all_in_one,psudoAngels),1)
    np.add.at(H,(-𝜌s_all_in_one,psudoAngels2),1)

    np.add.at(H,(-𝜌s_all_in_one+1,psudoAngels),1)
    np.add.at(H,(-𝜌s_all_in_one-1,psudoAngels),1)

    np.add.at(H,(-𝜌s_all_in_one+1,psudoAngels2),1)
    np.add.at(H,(-𝜌s_all_in_one-1,psudoAngels2),1)

def correct_possible_nhood_overflow(idx_x, idx_y, nhood_size):
    start_x = idx_x - int(nhood_size/2)
    end_x = (idx_x + int(nhood_size/2)) + 1
    start_y = idx_y - int(nhood_size/2)
    end_y = (idx_y + int(nhood_size/2)) + 1

    if start_x < 0: 
        start_x = 0

    if end_x > H.shape[1]:
        end_x = H.shape[1]
    
    if start_y < 0:
        start_y = 0

    if end_y > H.shape[0]: 
        end_y = H.shape[0]
    
    return start_x, end_x, start_y, end_y

def is_not_close(indicies, idx, threshold):
    for i in range(len(indicies)):
        dist = np.linalg.norm(np.array(indicies[i]) - np.array(idx))
        if dist <= threshold:
            return False
    return True

def NMS(H,th=500,box_size=(11,11), max_iou=0.1):
    xdist=box_size[1]//2
    ydist=box_size[0]//2
    h=np.zeros_like(H)
    AboveThreshHoldIndexes=np.where(H>=th)
    h[AboveThreshHoldIndexes]=H[AboveThreshHoldIndexes]
    #peak detection in 2d, there are many algorithms, we pick the mask

    fp = findpeaks(method='mask')
    peaks=fp.fit(h)["Xdetect"]
    peaks=np.where(peaks==True)

    #generates the boxes of the 
    boxes=torch.tensor([[x-xdist,y-ydist, x+xdist, y+ydist] for x,y in zip(*peaks)],dtype=torch.float32)
    scores = torch.tensor( H[peaks] , dtype=torch.float32)

    qp=nms(boxes, scores,max_iou)
    return np.array(peaks)[:,qp]

def NMS_without_libs2_0(H, nhood_size=11, value_th=500, spatial_th=80):
    indicies = [] # a list of tuples, each tuple is the index of the peak. Will be turned into a 2d-array 
    h=np.zeros_like(H)
    AboveThreshHoldIndexes=np.where(H>=value_th)
    h[AboveThreshHoldIndexes]=H[AboveThreshHoldIndexes]

    # a while loop that is bound to end due to the supression process
    # this is instead of choosing number of peaks by hand like NMS_without_libs1
    while not np.allclose(h, 0):
        # get index of a peak, if it were in a 1d-array ex. idx = 100
        idx = np.argmax(h) 
        # turn the index into a 2d-index ex. if idx = 100 and h.shape = (100, 100) then h_idx = (1, 0)
        # ex if idx = 250 and h.shape = (100, 100) then h_idx = (2, 50)
        h_idx = np.unravel_index(idx, h.shape) 
        idx_y, idx_x = h_idx

        # check if h_idx is not close to all the points in indicies, if true then append to indicies
        # if false then it might correspond to a line that is close to another line in the final image
        # in this version, if false I supress it and all the elements in its nhood
        # in NMS_without_libs2_1 I only supress it but not its nhood
        if is_not_close(indicies, h_idx, threshold=spatial_th): 
            indicies.append(h_idx)

        # a function that corrects the nhood indices just so we don't get a negative index of an index out of bounds
        start_x, end_x, start_y, end_y = correct_possible_nhood_overflow(idx_x, idx_y, nhood_size)

        # supression of the "peak" and its neighbourhood
        for x in range(start_x, end_x):
            for y in range(start_y, end_y):
                h[y, x] = 0
    
    return np.array(indicies).T
#%%
def create_Line(image, line_parameters,angle_step):
    #recorrecting rho
    𝜌_shift=calc_length(*image.shape)
    rho,theta=line_parameters
    rho=2*𝜌_shift-rho
    rho-=𝜌_shift
    #this is going to need debugging 
    theta=theta*np.pi/(180/angle_step)
    y1=rho*np.cos(theta)
    x1=rho*np.sin(theta)
    x2=int(x1-2000*np.cos(theta))
    y2=int(y1+2000*np.sin(theta))
    x3=int(x1+2000*np.cos(theta))
    y3=int(y1-2000*np.sin(theta))
    cv2.line(img=image, pt1=(x3,y3), pt2=(x2,y2),color= (255, 255, 0),thickness= 10)

def display_lines(image, lines_coordinates):
    print(lines_coordinates)
    line_image = np.zeros_like(image)
    if lines_coordinates is not None:
        for x1, y1, x2, y2 in lines_coordinates:
            # y2%=image.shape[0]
            cv2.line(line_image, (x1, y1), (x2, y2),color= (255, 255, 255),thickness= 10)
    return line_image


def apply_median_filter(img,filter_size=9):
    img =ndimage.median_filter(img, size=filter_size)
    save_and_show_image( img,"after_median.png")
    return img
#%%
def save_image_to_file(img,file_name):
    Image.fromarray(img).convert('RGB').save("./outputImages/"+file_name)

def initialize_environment():
    if not os.path.exists("outputImages"):
        os.mkdir("outputImages")
    plt.gray()

def save_and_show_image( img,fileName):
    plt.imshow(img)
    save_image_to_file(img,fileName)
#%%
def extract_edges(img):
    edges=cv2.Canny(img,50,150)
    save_and_show_image(edges,"edges.png")
    return edges
def get_roi_boundries(img):
    image_height=img.shape[0]
    return np.array([[(0,image_height),(1010,image_height),(600,310),(0,400)]])
def red_image(image_name):
    img =cv2.imread(image_name,0)
    return img
#%%
def draw_lines_on_empty_image(img, ROI_polygon_vertices, angle_step, best_points):
    lines_image=np.zeros_like(img)
    
    for i in zip(*best_points):#zip(best_points[0], bestpoints[1])
        create_Line(lines_image, i,angle_step)
    lines_image=region_of_interest(lines_image,ROI_polygon_vertices)
    return lines_image


def draw_lines_on_original_image(original_image_name,lines_image,line_color=[26, 255, 0]):

    original_image_rgb=cv2.cvtColor(cv2.imread(filename=original_image_name), cv2.COLOR_BGR2RGB)
    
    lines_image=cv2.cvtColor(lines_image, cv2.COLOR_BGR2RGB)
    
    lines_image[np.where((lines_image==[255,255,255]).all(axis=2))] = line_color
    combo_image = cv2.addWeighted(original_image_rgb, 0.8, lines_image, 1, 1)
    save_and_show_image(combo_image,"final_result.png") 
#%%
if(__name__=="__main__"):
    
    initialize_environment()
    original_imge_name='testHough.png'
    original_imge = red_image(image_name=original_imge_name)

    save_and_show_image( original_imge,fileName='greyScale.png')
    img=apply_median_filter(original_imge,filter_size=9)
    save_and_show_image(img, fileName="afterMedian.png")

    edges=extract_edges(img)


    ROI_polygon_vertices=get_roi_boundries(img)
    edges=region_of_interest(edges,ROI_polygon_vertices) 

    #🐶🐶🐶
    angle_step_size=0.1
    H=hough(edges,angle_step_size)

    #   getting peaks and then non-max suppression using findpeaks and torch library
    best_points=NMS(H, th=550,box_size=(81, 81), max_iou=0.1)
    best_points=NMS(H, th=550,box_size=(81, 81), max_iou=0.1)
    lines_image = draw_lines_on_empty_image(original_imge, ROI_polygon_vertices, angle_step_size, best_points) 
    
    draw_lines_on_original_image(original_imge_name,lines_image)

# %%
