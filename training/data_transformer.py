import tensorflow as tf
import cv2
import copy
import random
from math import exp,sqrt
from point_operations import Point,addPoints,addScalar,mulScalar,subtractPoint
from pycocotools.coco import COCO
import os
import numpy as np
import json

class AugmentSelection(object):
    flip = None
    degree = None
    crop = None
    scale = None

    def __init__(self,flip,degree,crop,scale):
        self.flip = flip
        self.degree = degree
        self.crop = crop
        self.scale = scale

class DataTransformer(object):
    # TransformParameter
    param = None
    # Number of parts in annotation
    np_ann = 0
    # Number of parts
    num_parts = 0
    is_table_set_ = False

    def __init__(self,transforParam):
        self.param = transforParam
        self.np_ann = self.param.num_parts_in_annot
        self.num_parts = self.param.num_parts  

    def transform(self,data): # data[0] = img, data[1] = joint_all, data[2] = mask_miss, data[3] = mask_all
        aug = AugmentSelection(False,0.0,(),0)
        # coco = COCO(annotation_file=anno_path)
        # filename = filename.decode("utf-8")
        # img,meta,mask_miss,mask_all = self.create_data_info(coco,filename,img_dir)
        # *** might have to decode
        
        #convert strings back to np arrays
        img = np.fromstring(data[0],dtype=np.uint8) # not working properly, giving us incorrect array
        mask_miss = np.fromstring(data[2], dtype=np.uint8) 
        mask_all = np.fromstring(data[3], dtype=np.uint8) 
        
        # meta = self.format_meta_data(meta)
        meta = self.format_meta_data(data[1])

        if(self.param.transform_body_joint):
            meta = self.TransformMetaJoints(meta)
        
        # Start transformation
        img_aug = np.zeros((self.param.crop_size_y,self.param.crop_size_x,3))
        
        mask_miss_aug = None

        aug.scale,img_temp,mask_miss = self.AugmentationScale(img,mask_miss,meta)
        aug.degree,img_temp2,mask_miss = self.AugmentationRotate(img_temp,mask_miss,meta)
        aug.crop,img_temp3,mask_miss_aug = self.AugmentationCropped(img_temp2,mask_miss,meta)
        aug.flip,img_aug,mask_miss_aug = self.AugmentationFlip(img_temp3,mask_miss_aug,meta)

        mask_miss_aug = cv2.resize(mask_miss_aug,(0,0),fx=1.0/self.param.stride,fy=1.0/self.param.stride,interpolation=cv2.INTER_CUBIC)

        offset = img_aug.shape[0] * img_aug.shape[1]
        rezX = img_aug.shape[1]
        rezY = img_aug.shape[0]
        grid_x = rezX / self.param.stride
        grid_y = rezY / self.param.stride
        channel_offset = grid_y * grid_x

        # label size is image size/ stride
        transformed_label = [0.0]*((self.param.crop_size_x / self.param.stride) * (self.param.crop_size_y / self.param.stride) * self.num_parts)
        for g_y in range(grid_y):
            for g_x in range(grid_x):
                for i in range(self.num_parts+1):
                    mask = float(mask_miss_aug[g_y,g_x]) / 255
                    transformed_label[i*channel_offset + g_y*grid_x + g_x] = mask
        
        self.GenerateLabelMap(transformed_label,img_aug,meta)
        
        t_label = np.copy(transformed_label)
        weights = np.reshape(t_label, shape = [grid_y * self.num_parts, grid_x])
        vec = np.reshape(np.copy(transformed_label + start_label_data), shape = [grid_y * self.num_parts, grid_x])
        label = np.multiply(vec, weights)
        mask = np.reshape(t_label, shape = [grid_y, grid_x])
        
        return data_img, mask, label 
    
    def TransformMetaJoints(self,meta=None):
        meta["joint_self"] = self.TransformJoints(meta["joint_self"]) # joint_self,joint_others => (17,3)
        assert meta["joint_self"].shape == (56,3)
        for j in meta["joint_others"]:
            j = self.TransformJoints(j)
            assert j.shape == (56,3)
        return meta
    def TransformJoints(self,j=None):
        # Coco dataset
        jo = np.copy(j)
        if(self.num_parts == 56):
            # joint is a connection between 2 body parts
            from_body_part = [1,6,7,9,11,6,8,10,13,15,17,12,14,16,3,2,5,4]
            to_body_part = [1,7,7,9,11,6,8,10,13,15,17,12,14,16,3,2,5,4]
            
            jo.resize((56,3))
            for i in range(18):
                jo[i,0] = j[from_body_part[i]-1,0] + j[to_body_part[i]-1,0] * 0.5 
                jo[i,1] = j[from_body_part[i]-1,1] + j[to_body_part[i]-1,1] * 0.5
                if(j[from_body_part[i]-1,2] == 2 or j[to_body_part[i]-1,2] == 2):
                    jo[i,2] = 2
                elif(j[from_body_part[i]-1,2] == 3 or j[to_body_part[i]-1,2] == 3):
                    jo[i,2] = 3
                else:
                    jo[i,2] = 1 if(j[from_body_part[i]-1,2] != 0 and j[to_body_part[i]-1,2] != 0) else 0
        return jo

    def AugmentationScale(self,img_src,mask_miss,meta):
        print("hi")
        dice = random.random()

        if(dice > self.param.scale_prob):
            img_temp = np.copy(img_src)
            scale_multiplier = 1
        else:
            print("hi")
            dice2 = random.random()
            scale_multiplier = (self.param.scale_max - self.param.scale_min) * dice2 + self.param.scale_min
        scale_abs = self.param.target_dist/meta["scale_provided"]
        scale = scale_abs * scale_multiplier

        img_temp = cv2.resize(img_src,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
        mask_miss = cv2.resize(mask_miss,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)


        meta["objpos"] = [i * scale for i in meta["objpos"]]
        for i in range(self.num_parts):
            (meta["joint_self"])[i,0] *= scale
            (meta["joint_self"])[i,1] *= scale
        for p in range(meta["num_other_people"]):
            meta["objpos_other"][p] = [x * scale for x in meta["objpos_other"][p]]
            for i in range(self.num_parts):
                meta["joint_others"][p][i,0] *= scale
                meta["joint_others"][p][i,1] *= scale

        return scale_multiplier,img_temp,mask_miss
    
    def AugmentationRotate(self,img_src,mask_miss, meta):
        if(self.param.aug_way == "rand"):
            dice = random.random()
            degree = (dice - 0.5) * 2 * self.param.max_rotate_degree
        elif(self.param.aug_way == "table"):
            degree = aug_degs_[meta["write_number"]][meta["epoch"] % self.param.num_total_augs] # assuming augmentation table set in ReadMetaData
        else:
            degree = 0
        
        center = (img_src.shape[1]/2.0,img_src.shape[0]/2.0) # columns,rows
        R = cv2.getRotationMatrix2D(center,degree, 1.0)
        img_dst = cv2.warpAffine(src=img_src,M=R,flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=(128,128,128)) 
        mask_miss = cv2.warpAffine(src=mask_miss,M=R,flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=(255)) # borderValue 0 for MPI/255 for COCO

        meta["objpos"][0], meta["objpos"][1] = self.RotatePoint((meta["objpos"])[0],(meta["objpos"])[1],R)
        for i in range(self.num_parts):
            meta["joint_self"][i,0],meta["joint_self"][i,1] = self.RotatePoint(meta["joint_self"][i,0],meta["joint_self"][i,1],R)
        for p in range(meta["num_other_people"]):
            meta["objpos_other"][p][0], meta["objpos_other"][p][1] = self.RotatePoint(meta["objpos_other"][p][0],meta["objpos_other"][p][1],R)
            for i in range(self.num_parts):
                meta["joint_others"][p][i,0], meta["joint_others"][p][i,1] = self.RotatePoint(meta["joint_others"][p][i,0],meta["joint_others"][p][i,1],R)
        return degree,img_dst,mask_miss

    def AugmentationCropped(self,img_src,mask_miss,meta):
        dice_x = random.random()
        dice_y = random.random()
        
        x_offset = (dice_x - 0.5) * 2 * self.param.center_perterb_max
        y_offset = (dice_y - 0.5) * 2 * self.param.center_perterb_max
        
        center = [meta["objpos"][0] + x_offset,meta["objpos"][1] + y_offset]
    
        offset_left = -(center[0] - (self.param.crop_size_x/2))
        offset_up = -(center[1] - (self.param.crop_size_y/2))

        img_dst = np.zeros((self.param.crop_size_y, self.param.crop_size_x, 3)) + (128,128,128)
        mask_miss_aug = np.zeros((param.crop_size_y, self.param.crop_size_x)) + (255)
        
        for i in range(self.param.crop_size_y):
            for j in range(self.param.crop_size_x):
                coord_x_on_img = center[0] - self.param.crop_size_x/2 + j
                coord_y_on_img = center[1] - self.param.crop_size_y/2 + i
                if(self.OnPlane(coord_x_on_img,coord_y_on_img,img_src.shape)):
                    img_dst[i][j] = img_src[coord_y_on_img][coord_x_on_img]
                    mask_miss_aug[i][j] = mask_miss[coord_y_on_img][coord_x_on_img]
        
        offset = [offset_left,offset_up]
        meta["objpos"][0] += offset_left
        meta["objpos"][1] += offset_up

        for i in range(self.num_parts):
            meta["joint_self"][i,0] += offset_left
            meta["joint_self"][i,1] += offset_up

        for p in range(meta["num_other_people"]):
            meta["objpos_other"][p][0] += offset_left
            meta["objpos_other"][p][1] += offset_up
            for i in range(self.num_parts):
                meta["joint_others"][p][i,0] += offset_left
                meta["joint_others"][p][i,1] += offset_up
        
        return [x_offset,y_offset],img_dst,mask_miss_aug

    def AugmentationFlip(self,img_src,mask_miss_aug,meta):
        if(self.param.aug_way == "rand"):
            dice = random.random()
            doflip = (dice <= self.param.flip_prob)
        elif(self.param.aug_way == "table"):
            doflip = aug_flips_[meta.write_number][meta.epoch % self.param.num_total_augs] == 1
        else:
            doflip = False
        
        if(doflip):
           img_aug = cv2.flip(img_src,1)
           w = img_src.shape[1]
           mask_miss_aug = cv2.flip(mask_miss_aug,1)
           meta.objpos.x = w - 1 - meta.objpos.x
           
           for i in range(self.num_parts):
               if(meta.joint_self.joints[i] is not None):
                   (meta.joint_self.joints[i]).x = w - 1 - (meta.joint_self.joints[i]).x
            
           if(self.param.transform_body_joint):
               self.SwapLeftRight(meta.joint_self)
        
           for p in range(meta.num_other_people):
               meta.objpos_other[p].x = w - 1 - meta.objpos_other[p].x
                
               for i in range(self.num_parts):
                   if(meta.joint_others[p].joints[i] is  not None):
                       meta.joint_others[p].joints[i].x = w - 1 - meta.joint_others[p].joints[i].x
                
               if(self.param.transform_body_joint):
                   self.SwapLeftRight(meta.joint_others[p])
        else:
            img_aug = np.copy(img_src)
        
        return doflip,img_aug,mask_miss_aug
    
    def AugmentationFlip(self,img_src,mask_miss_aug,meta):
        if(self.param.aug_way == "rand"):
            dice = random.random()
            doflip = (dice <= self.param.flip_prob)
        elif(self.param.aug_way == "table"):
            doflip = aug_flips_[meta["write_number"]][meta["epoch"] % self.param.num_total_augs] == 1
        else:
            doflip = False
        
        if(doflip):
           img_aug = cv2.flip(img_src,1)
           w = img_src.shape[1]
           mask_miss_aug = cv2.flip(mask_miss_aug,1)
           meta["objpos"][0] = w - 1 - meta["objpos"][0]
           
           for i in range(self.num_parts):
                meta["joint_self"][i,0] = w - 1 - meta["joint_self"][i,0]
            
           if(self.param.transform_body_joint):
               self.SwapLeftRight(meta["joint_self"])
        
           for p in range(meta["num_other_people"]):
               meta["objpos_other"][p][0] = w - 1 - meta["objpos_other"][p][0]
                
               for i in range(self.num_parts):
                   meta["joint_others"][p][i,0] = w - 1 - meta["joint_others"][p][i,0]
                
               if(self.param.transform_body_joint):
                   self.SwapLeftRight(meta["joint_others"][p])
        else:
            img_aug = np.copy(img_src)
        
        return doflip,img_aug,mask_miss_aug
    
    def RotatePoint(self,x=None,y=None,R=None):
        # Come back and check that shapes are correct
        point = np.asarray([p.x,p.y,1.0])
        point.reshape((3,1))
        
        new_point = R * point
        return new_point[0][0],new_point[1][0]
    
    def OnPlane(self,x=None,y=None,img_shape=None):
        if (x < 0 or y < 0):
            return False
        if (x >= img_shape[1] or y >= img_shape[0]):
            return False
        return True

    def SwapLeftRight(self,j=None):
        if(self.num_parts == 56):
            right = [3,4,5,9,10,11,15,17]
            left = [6,7,8,12,13,14,16,18]
            for i in range(8):
                ri = right[i] - 1
                li = left[i] - 1
                temp = j[ri]
                j[ri] = j[li]
                j[li] = temp
                temp_v = j[ri,2]
                j[ri,2] = j[li,2]
                j[li,2] = temp_v
    
    def GenerateLabelMap(self,transformed_label,img_aug,meta):

        rezX = img_aug.shape[1]
        rezY = img_aug.shape[0]
        stride = self.param.stride
        grid_x = rezX / stride
        grid_y = rezY / stride
        channelOffset = grid_y * grid_x

        for g_y in range(grid_y):
            for g_x in range(grid_x):
                for i in range(self.num_parts+1,2*(self.num_parts+1)):
                    transformed_label[i*channelOffset + g_y*grid_x + g_x] = 0.0

        # Creating heatmap
        if(self.num_parts == 56):
            for i in range(18):
                center = meta["joint_self"][i]
                if(meta["joint_self"][i][2] <= 1):
                    self.PutGaussianMaps(transformed_label + (i+self.num_parts+39)*channelOffset, center, stride,
                grid_x, grid_y, self.param.sigma)
                
                for j in range(meta["num_other_people"]):
                    center = meta["joint_others"][j][i] 
                    if(meta["joint_others"][j][2] <= 1):
                    #center = meta.joint_others[j].joints[i]
                    #if(meta.joint_others[j].is_visible[i] <= 1):
                        self.PutGaussianMaps(transformed_label + (i+self.num_parts+39)*channelOffset, center, stride,
                                        grid_x, grid_y, self.param.sigma)

        # Creating PAF
        mid_1 = [2, 9,  10, 2,  12, 13, 2, 3, 4, 3,  2, 6, 7, 6,  2, 1,  1,  15, 16]
        mid_2 = [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]
        thre = 1
        
        # Add vector maps for all limbs 
        for i in range(19):
            count = np.zeros((grid_y,grid_x))
            jo = meta["joint_self"]
            if(jo[mid_1[i]-1][2] <= 1 and jo[mid_2[i]-1][2] <= 1):
                self.PutVecMaps(transformed_label + (self.num_parts+ 1+ 2*i)*channelOffset, transformed_label + (self.num_parts+ 2+ 2*i)*channelOffset,
            count, jo[mid_1[i]-1], jo[mid_2[i]-1], stride, grid_x, grid_y, self.param.sigma, thre)

            for j in range(meta["num_other_people"]):
                jo2 = meta["joint_others"][j]
                if(jo2[mid_1[i]-1][2] <= 1 and jo2[mid_2[i]-1][2] <= 1):
                    self.PutVecMaps(transformed_label + (self.num_parts+ 1+ 2*i)*channelOffset, transformed_label + (self.num_parts+ 2+ 2*i)*channelOffset,
                count, jo2[mid_1[i]-1], jo2[mid_2[i]-1], stride, grid_x, grid_y, self.param.sigma, thre)   

        # Put background channel **** no idea what this is doing **** 
        for g_y in range(grid_y):
            for g_x in range(grid_x):
                maximum = 0
                # second background channel
                for i in range(self.num_parts+39,self.num_parts+57):
                    maximum = maximum if maximum > transformed_label[i*channelOffset + g_y*grid_x + g_x] else transformed_label[i*channelOffset + g_y*grid_x + g_x]
                transformed_label[(2*self.num_parts+1)*channelOffset + g_y*grid_x + g_x] = max(1.0-maximum, 0.0)

    def PutGaussianMaps(self,entry,center,stride,grid_x,grid_y,sigma):
        start = stride/2.0 - 0.5 # 0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...
        for i in range(grid_y):
            for j in range(grid_x):
                x = start + g_x * stride
                y = start + g_y * stride
                d2 = (x-center.x)*(x-center.x) + (y-center.y)*(y-center.y)
                exponent = d2 / 2.0 / sigma / sigma
                
                if(exponent > 4.6052): # ln(100) = -ln(1%)
                    continue
                
                entry[g_y*grid_x + g_x] += exp(-exponent)
                
                if (entry[g_y*grid_x + g_x] > 1):
                    entry[g_y*grid_x + g_x] = 1

    def PutVecMaps(self,entryX,entryY,count,centerA,centerB,stride,grid_x,grid_y,sigma,thre):         
        centerB = np.multiply(centerB,0.125)
        centerA = np.mulitply(centerA,0.125)

        bc = centerB-centerA
        min_x = max(int(round(min(centerA[0],centerB[0]) - thre)),0)
        max_x = min(int(round(max(centerA[0], centerB[0])+thre)),grid_x)
  
        min_y = max(int(round(min(centerA[1], centerB[1])-thre)),0)
        max_y = min(int(round(max(centerA[1], centerB[1])+thre)), grid_y)

        norm_bc = sqrt((bc[0] * bc[0]) + (bc[1] * bc[1]))
        
        # skip if body parts overlap
        if(norm_bc < 1e-8):
            return

        bc[0] = bc[0]/norm_bc
        bc[1] = bc[1]/norm_bc

        for g_x in range(min_y,max_y):
            for g_y in range(min_x,max_x):
                ba = [g_x - centerA[0],g_y - centerA[1]]
                dist = abs((ba[0] * bc[1]) - (ba[1] * bc[0]))
                if(dist <= thre):
                   cnt = count[g_y][g_x] 
                   if(cnt == 0):
                       entryX[g_y*grid_x + g_x] = bc[0]
                       entryY[g_y*grid_x + g_x] = bc[1]
                   else:
                       # averaging when limbs of multiple persons overlap
                       entryX[g_y*grid_x + g_x] = (entryX[g_y*grid_x + g_x]*cnt + bc[0]) / (cnt + 1)
                       entryY[g_y*grid_x + g_x] = (entryY[g_y*grid_x + g_x]*cnt + bc[1]) / (cnt + 1)
                       count[g_y][g_x] = cnt + 1

    def format_meta_data(self,meta):
        # joint_self and joint_others back to np arrays
        meta = json.loads(meta)
        meta["joint_self"] = np.array(meta["joint_self"])
        meta["joint_others"] = np.array(meta["joint_others"])
                                          
        for i in range(self.np_ann):
            joint = meta["joint_self"]
            if(joint[i,2] == 2):
                joint[i,2] = 3
            else:
                joint[i,2] = 0 if joint[i,2] == 0 else 1
                if(joint[i,0] < 0 or joint[i,1] < 0 or joint[i,0] >= meta["img_width"] or joint[i,1] >= meta["img_height"]):
                    joint[i,2] = 2
        
        for i in range(meta["numOtherPeople"]):
            for j in range(self.np_ann):
                joint = (meta["joint_others"])[i]
                if(joint[j,2] == 2):
                    joint[j,2] = 3
                else:
                    joint[j,2] = 0 if joint[j,2] == 0 else 1
                    if(joint[j,0] < 0 or joint[j,1] < 0 or joint[j,0] >= meta["img_width"] or joint[j,1] >= meta["img_height"]):
                        joint[j,2] = 2
        return meta 
    

