import tensorflow as tf
import cv2
import copy
import random
from math import exp,sqrt
from point_operations import Point,addPoints,addScalar,mulScalar,subtractPoint
from pycocotools.coco import COCO

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
        
class Joints(object):
    joints = [] # Point2f -> Point
    is_visible = [] # vector<float>

    def __init__():
        # TODO(): add initializaiton

class MetaData(object):
    dataset = None # string
    img_size = None # size ?
    is_validation = None # boolean
    num_other_people = None # int
    people_index = None # int
    annolist_index = None
    write_number = None # int
    total_write_number = None # int
    epoch = None # int
    objpos = None # Point ?
    scale_self = None # float 
    joint_self = None # Joint ?

    objpos_other = None # vector<Point2f>
    scale_other = None # vector<float>
    joint_others = None # vector<Joints>

    def __init__():
        # TODO(): add initializaiton

class DataTransformer(object):
    # TransformParameter
    param = None
    # Number of parts in annotation
    np_ann = 0
    # Number of parts
    num_parts = 0
    is_table_set_ = False

    def __init__(self,transforParam):
        param = transforParam
        self.np_ann = param.num_parts_in_annot
        self.num_parts = param.num_parts   

    def transform(filename=None,anno_path=None,img_dir=None):
        aug = AugmentSelection(False,0.0,(),0)
        coco = COCO(anno_path)

        img,meta,mask_miss,mask_all = create_data_info(coco,filename,img_dir)

        # Perform CLAHE
        if(param.do_clahe):
            # *** Currently false all the time, look into later
            # Code snippet
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            # cl1 = clahe.apply(img)
        
        # Convert to grayscale
        if(param.gray == 1):
            # Not sure why this is done in C++ server
            # cv::cvtColor(img, img, CV_BGR2GRAY);
            # cv::cvtColor(img, img, CV_GRAY2BGR);

        if(param.transform_body_joint):
            TransformMetaJoints(meta)
        
        # Start transformation
        img_aug = np.zeros(param.crop_size_y,param.crop_size_x,3)
        mask_miss_aug = None

        aug.scale,img_temp,mask_miss = AugmentationScale(img,mask_miss,meta)
        aug.degree,img_temp2,mask_miss = AugmentationRotate(img_temp,mask_miss,meta)
        aug.crop,img_temp3,mask_miss_aug = AugmentationCropped(img_temp2,mask_miss,meta)
        aug.flip,img_aug,mask_miss_aug = AugmentationFlip(img_temp3,mask_miss_aug,meta)

        mask_miss_aug = cv2.resize(mask_miss_aug,(0,0),fx=1.0/param.stride,fy=1.0/param.stride,interpolation=cv2.INTER_CUBIC)

        offset = img_aug.shape[0] * img_aug.shape[1]
        rezX = img_aug.shape[1]
        rezY = img_aug.shape[0]
        grid_x = rezX / param.stride
        grid_y = rezY / param.stride
        channel_offset = grid_y * grid_x

        # label size is image size/ stride
        transformed_label = [0.0]*((params.crop_size_x / param.stride) * (params.crop_size_y / param.stride) * num_parts)
        for g_y in range(grid_y):
            for g_x in range(grid_x):
                for i in range(num_parts+1):
                    mask = float(mask_miss_aug.at<uchar>(g_y, g_x)) / 255
                    transformed_label[i*channel_offset + g_y*grid_x + g_x] = mask
        
        GenerateLabelMap(transformed_label,img_aug,meta)
        data_img,mask_img,label = format_data(img_aug,transformed_label,meta)
        return data_img, mask_img, label 
    
    def TransformMetaJoints(meta=None):
        TransformJoints(meta["joint_self"]) # joint_self,joint_others => (17,3)
        for j in meta["joint_others"]:
            TransformJoints(j)

    def TransformJoints(j=None):
        # Coco dataset
        jo = np.copy(j)
        if(num_parts == 56):
            # joint is a connection between 2 body parts
            from_body_part = [1,6,7,9,11,6,8,10,13,15,17,12,14,16,3,2,5,4]
            to_body_part = [1,7,7,9,11,6,8,10,13,15,17,12,14,16,3,2,5,4]
            # Not sure if this resize is necessary maybe take out later
            jo.joints += [None]*(56 - len(joints))
            jo.is_visible += [None]*(56 - len(joints))
            
            for i in range(18):
                jo.joints[i] = mulScalar(addPoints(j.joints[from_body_part[i]-1],j.joints[to_body_part[i]-1]),0.5)

                if(j.is_visible[from_body_part[i]-1] == 2 or j.is_visible[to_body_part[i]-1] == 2):
                    jo.is_visible[i] = 2
                elif(j.is_visible[from_body_part[i]-1] == 3 or j.is_visible[to_body_part[i]-1] == 3):
                    jo.is_visible[i] = 3
                else:
                    jo.is_visible[i] = 1 if(j.is_visible[from_body_part[i]-1] != 0 and j.is_visible[to_body_part[i]-1] != 0) else 0
        j = copy.deepcopy(jo)

    def AugmentationScale(img_src,mask_miss,meta):
        dice = random.random()
        if(dice > param.scale_prob):
            img_temp = np.copy(img_src) # *** will probably break check when testing ***
            scale_multiplier = 1
        else:
            dice2 = random.random()
            scale_multiplier = (param.scale_max - param.scale_min) * dice2 + param.scale_min
        
        scale_abs = param.target_dist/meta.scale_self
        scale = scale_abs * scale_multiplier
        
        img_temp = cv2.resize(img_src,(0,0),fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
        mask_miss = cv2.resize(mask_miss,(0,0),fx=scale,fy=scale,interpolation=cv.INTER_CUBIC)
        meta.objpos = mulScalar(meta.objpos,scale)

        for i in range(num_parts):
            if(meta.joint_self.joints[i] is not None):
                meta.joint_self.joints[i] = mulScalar(meta.joint_self.joints[i],scale)
        for p in range(meta.num_other_people)
            meta.objpos_other[p] = mulScalar(meta.objpos_other[p],scale)
            for i in range(num_parts):
                if(meta.joint_others[p].joints[i] i not None):
                    meta.joint_others[p].joints[i] = mulScalar(meta.joint_others[p].joints[i],scale)
        return scale_multiplier,img_temp,mask_miss

    def AugmentationRotate(img_src,mask_miss, meta):
        if(param.aug_way == "rand"):
            dice = random.random()
            degree = (dice - 0.5) * 2 * param.max_rotate_degree
        elif(param.aug_way == "table"):
            degree = aug_degs_[meta.write_number][meta.epoch % param.num_total_augs] # assuming augmentation table set in ReadMetaData
        else:
            degree = 0
        
        center = (img_src.shape[1]/2.0,img_src.shape[0]/2.0) # columns,rows
        R = cv2.getRotationMatrix2D(center,degree, 1.0)
        img_dst = cv2.warpAffine(src=img_src,M=R,flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=(128,128,128)) 
        mask_miss = cv2.warpAffine(src=mask_miss,M=R,flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=(255)) # borderValue 0 for MPI/255 for COCO

        RotatePoint(meta.objpos,R)
        for i in range(num_parts):
            if(meta.joint_self.joints[i] is not None):
                RotatePoint(meta.joint_self.joints[i], R)
        for p in range(meta.num_other_people):
            RotatePoint(meta.objpos_other[p],R)
            for i in range(num_parts):
                if(meta.joint_others[p].joints[i] is not None):
                    RotatePoint(meta.joint_others[p].joints[i],R)
        return degree,img_dst,mask_miss
    
    def AugmentationCropped(img_src,mask_miss,meta):
        dice_x = random.random()
        dice_y = random.random()
        
        x_offset = (dice_x - 0.5) * 2 * param.center_perterb_max
        y_offset = (dice_y - 0.5) * 2 * param.center_perterb_max
        
        center = addPoints(meta.objpos,Point(x_offset,y_offset))
    
        offset_left = -(center.x - (param.crop_size_x/2))
        offset_up = -(center.y - (param.crop_size_y/2))

        img_dst = np.zeros((param.crop_size_y, param.crop_size_x, 3)) + (128,128,128)
        mask_miss_aug = np.zeros((param.crop_size_y, param.crop_size_x)) + (255)
        
        for i in range(param.crop_size_y):
            for j in range(param.crop_size_x):
                coord_x_on_img = center.x - param.crop_size_x/2 + j
                coord_y_on_img = center.y - param.crop_size_y/2 + i
                if(OnPlane(Point(coord_x_on_img, coord_y_on_img),img_src.shape)):
                    img_dst[i][j] = img_src[coord_y_on_img][coord_x_on_img]
                    mask_miss_aug[i][j] = mask_miss[coord_y_on_img][coord_x_on_img]
        
        offset = Point(offset_left,offset_up)
        meta.objpos = addPoints(meta.objpos,offset)
        for i in range(num_parts):
            if(meta.joint_self.joints[i] is not None):
                meta.joint_self.joints[i] = addPoints(meta.joint_self.joints[i],offset)
        for p in range(meta.num_other_people):
            meta.objpos_other[p] = addPoints(meta.objpos_other[p],offset)
            for i in range(num_parts):
                if(meta.joint_others[p].joints[i] is not None):
                    meta.joint_others[p].joints[i] = addPoints(meta.joint_others[p].joints[i],offset)
        
        return Point(x_offset,y_offset),img_dst,mask_miss_aug

    def AugmentationFlip(img_src,mask_miss_aug,meta):
        if(param.aug_way == "rand"):
            dice = random.random()
            doflip = (dice <= param.flip_prob)
        elif(param.aug_way == "table"):
            doflip = aug_flips_[meta.write_number][meta.epoch % param.num_total_augs] == 1
        else:
            doflip = False
        
        if(doflip):
           img_aug = cv2.flip(img_src,1)
           w = img_src.shape[1]
           mask_miss_aug = cv2.flip(mask_miss_aug,1)
           meta.objpos.x = w - 1 - meta.objpos.x
           
           for i in range(num_parts):
               if(meta.joint_self.joints[i] is not None):
                   (meta.joint_self.joints[i]).x = w - 1 - (meta.joint_self.joints[i]).x
            
            if(param.transform_body_joint):
                SwapLeftRight(meta.joint_self)
        
            for p in range(meta.num_other_people):
                meta.objpos_other[p].x = w - 1 - meta.objpos_other[p].x
                
                for i in range(num_parts):
                    if(meta.joint_others[p].joints[i] is  not None):
                        meta.joint_others[p].joints[i].x = w - 1 - meta.joint_others[p].joints[i].x
                
                if(param.transform_body_joint):
                    SwapLeftRight(meta.joint_others[p])
        else:
            img_aug = np.copy(img_src)
        
        return doflip,img_aug,mask_miss_aug
    
    def RotatePoint(p=None,R=None):
        # Come back and check that shapes are correct
        point = np.asarray([p.x,p.y,1.0])
        point.reshape((3,1))
        
        new_point = R * point
        p.x = new_point[0][0]
        p.y = new_point[1][0]
    
    def OnPlane(p=None,img_shape=None):
        if (p.x < 0 or p.y < 0):
            return False
        if (p.x >= img_shape[1] or p.y >= img_shape[0]):
            return False
        return True

    def SwapLeftRight(j=None):
        if(num_parts == 56):
            right = [3,4,5,9,10,11,15,17]
            left = [6,7,8,12,13,14,16,18]
            for i in range(8):
                ri = right[i] - 1
                li = left[i] - 1
                temp = j.joints[ri]
                j.joints[ri] = j.joints[li]
                j.joints[li] = temp
                temp_v = j.is_visible[ri]
                j.is_visible[ri] = j.is_visible[li]
                j.is_visible[li] = temp_v
    
    def GenerateLabelMap(transformed_label,img_aug,meta):

        rezX = img_aug.shape[1]
        rezY = img_aug.shape[0]
        stride = param.stride
        grid_x = rezX / stride
        grid_y = rezY / stride
        channelOffset = grid_y * grid_x

        for g_y in range(grid_y):
            for g_x in range(grid_x):
                for i in range(num_parts+1,2*(num_parts+1)):
                    transformed_label[i*channelOffset + g_y*grid_x + g_x] = 0.0

        # Creating heatmap
        if(num_parts == 56):
            for i in range(18):
                center = meta.joint_self.joints[i]
                if(meta.joint_self.is_visible[i] <= 1):
                    PutGaussianMaps(transformed_label + (i+num_parts+39)*channelOffset, center, param_.stride,
                grid_x, grid_y, param_.sigma)
                
                for j in range(meta.num_other_people):
                    center = meta.joint_others[j].joints[i]
                    if(meta.joint_others[j].is_visible[i] <= 1):
                            PutGaussianMaps(transformed_label + (i+num_parts+39)*channelOffset, center, param_.stride,
                    grid_x, grid_y, param_.sigma)

        # Creating PAF
        mid_1 = [2, 9,  10, 2,  12, 13, 2, 3, 4, 3,  2, 6, 7, 6,  2, 1,  1,  15, 16]
        mid_2 = [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]
        thre = 1
        
        # Add vector maps for all limbs 
        for i in range(19):
            count = np.zeros((grid_y,grid_x))
            jo = meta.joint_self
            if(jo.is_visible[mid_1[i]-1] <= 1 and jo.is_visible[mid_2[i]-1] <= 1):
                PutVecMaps(transformed_label + (num_parts+ 1+ 2*i)*channelOffset, transformed_label + (num_parts+ 2+ 2*i)*channelOffset,
            count, jo.joints[mid_1[i]-1], jo.joints[mid_2[i]-1], param_.stride, grid_x, grid_y, param_.sigma, thre)

            for j in range(meta.num_other_people)
                jo2 = meta.joint_others[j]
                if(jo2.is_visible[mid_1[i]-1] <= 1 and jo2.is_visible[mid_2[i]-1] <= 1):
                    PutVecMaps(transformed_label + (num_parts+ 1+ 2*i)*channelOffset, transformed_label + (num_parts+ 2+ 2*i)*channelOffset,
                count, jo2.joints[mid_1[i]-1], jo2.joints[mid_2[i]-1], param_.stride, grid_x, grid_y, param_.sigma, thre)   

        # Put background channel **** no idea what this is doing **** 
        for g_y in range(grid_y):
            for g_x in range(grid_x):
                maximum = 0
                # second background channel
                for i in range(num_parts+39,num_parts+57):
                    maximum = maximum if maximum > transformed_label[i*channelOffset + g_y*grid_x + g_x] else transformed_label[i*channelOffset + g_y*grid_x + g_x]
                transformed_label[(2*num_parts+1)*channelOffset + g_y*grid_x + g_x] = max(1.0-maximum, 0.0)
        

    def PutGaussianMaps(entry,center,stride,grid_x,grid_y,sigma):
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

    def PutVecMaps(entryX,entryY,count,centerA,centerB,stride,grid_x,grid_y,sigma,thre):         
        centerB = mulScalar(centerB,0.125)
        centerA = mulScalar(centerA,0.125)

        bc = subtractPoint(centerB,centerA)
        min_x = max(int(round(min(centerA.x,centerB.x) - thre)),0)
        max_x = min(int(round(max(centerA.x, centerB.x)+thre)),grid_x)
  
        min_y = max(int(round(min(centerA.y, centerB.y)-thre)),0)
        max_y = min(int(round(max(centerA.y, centerB.y)+thre)), grid_y)

        norm_bc = sqrt((bc.x * bc.x) + (bc.y * bc.y))
        
        # skip if body parts overlap
        if(norm_bc < 1e-8):
            return

        bc.x = bc.x/norm_bc
        bc.y = bc.y/norm_bc

        for g_x in range(min_y,max_y):
            for g_y in range(min_x,max_x):
                ba = Point(g_x - centerA.x,g_y - centerA.y)
                dist = abs((ba.x * bc.y) - (ba.y * bc.x))
                if(dist <= thre):
                   cnt = count[g_y][g_x] 
                   if(cnt == 0):
                       entryX[g_y*grid_x + g_x] = bc.x
                       entryY[g_y*grid_x + g_x] = bc.y
                   else:
                       # averaging when limbs of multiple persons overlap
                       entryX[g_y*grid_x + g_x] = (entryX[g_y*grid_x + g_x]*cnt + bc.x) / (cnt + 1)
                       entryY[g_y*grid_x + g_x] = (entryY[g_y*grid_x + g_x]*cnt + bc.y) / (cnt + 1)
                       count[g_y][g_x] = cnt + 1
    
    def create_data_batch(img_aug,transformed_label,meta):
        # Prepare batch


    def create_data_info(coco,filename,img_dir):
        img_id = filename[:len(filename) - 4]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        img_anns = coco.loadAnns(ann_ids)

        numPeople = len(img_anns)
        image = coco.imgs[img_id]
        h, w = image['height'], image['width']
        dataset_type = "COCO"

        print("Image ID ", img_id)

        persons = []
        prev_center = []
        joint_all = {}

        for p in range(numPeople):

            # skip this person if parts number is too low or if
            # segmentation area is too small
            if img_anns[p]["num_keypoints"] < 5 or img_anns[p]["area"] < 32 * 32:
                continue

            anno = img_anns[p]["keypoints"]

            pers = {}

            person_center = [img_anns[p]["bbox"][0] + img_anns[p]["bbox"][2] / 2,
                                img_anns[p]["bbox"][1] + img_anns[p]["bbox"][3] / 2]

            # skip this person if the distance to exiting person is too small
            flag = 0
            for pc in prev_center:
                a = np.expand_dims(pc[:2], axis=0)
                b = np.expand_dims(person_center, axis=0)
                dist = cdist(a, b)[0]
                if dist < pc[2]*0.3:
                    flag = 1
                    continue

            if flag == 1:
                continue

            pers["objpos"] = person_center
            pers["bbox"] = img_anns[p]["bbox"]
            pers["segment_area"] = img_anns[p]["area"]
            pers["num_keypoints"] = img_anns[p]["num_keypoints"]

            pers["joint"] = np.zeros((17, 3))
            for part in range(17):
                pers["joint"][part, 0] = anno[part * 3]
                pers["joint"][part, 1] = anno[part * 3 + 1]

                if anno[part * 3 + 2] == 2:
                    pers["joint"][part, 2] = 1
                elif anno[part * 3 + 2] == 1:
                    pers["joint"][part, 2] = 0
                else:
                    pers["joint"][part, 2] = 2

            pers["scale_provided"] = img_anns[p]["bbox"][3] / 368

            persons.append(pers)
            prev_center.append(np.append(person_center, max(img_anns[p]["bbox"][2], img_anns[p]["bbox"][3])))


        if len(persons) > 0:

            joint_all["dataset"] = dataset_type

            joint_all["img_width"] = w
            joint_all["img_height"] = h
            joint_all["image_id"] = img_id
            joint_all["annolist_index"] = i

            # set image path
            joint_all["img_path"] = os.path.join(img_dir, '%012d.jpg' % img_id)

            # set the main person
            joint_all["objpos"] = persons[0]["objpos"]
            joint_all["bbox"] = persons[0]["bbox"]
            joint_all["segment_area"] = persons[0]["segment_area"]
            joint_all["num_keypoints"] = persons[0]["num_keypoints"]
            joint_all["joint_self"] = persons[0]["joint"]
            joint_all["scale_provided"] = persons[0]["scale_provided"]

            # set other persons
            joint_all["joint_others"] = []
            joint_all["scale_provided_other"] = []
            joint_all["objpos_other"] = []
            joint_all["bbox_other"] = []
            joint_all["segment_area_other"] = []
            joint_all["num_keypoints_other"] = []

            for ot in range(1, len(persons)):
                joint_all["joint_others"].append(persons[ot]["joint"])
                joint_all["scale_provided_other"].append(persons[ot]["scale_provided"])
                joint_all["objpos_other"].append(persons[ot]["objpos"])
                joint_all["bbox_other"].append(persons[ot]["bbox"])
                joint_all["segment_area_other"].append(persons[ot]["segment_area"])
                joint_all["num_keypoints_other"].append(persons[ot]["num_keypoints"])

            joint_all["people_index"] = 0
            lenOthers = len(persons) - 1

            joint_all["numOtherPeople"] = lenOthers

        img = cv2.imread(joint_all["img_path"])
        mask_all,mask_miss = create_masks(img_anns,img.shape)

        height = img.shape[0]
        width = img.shape[1]

        if (width < 64):
            img = cv2.copyMakeBorder(img, 0, 0, 0, 64 - width, cv2.BORDER_CONSTANT,
                                     value=(128, 128, 128))
            print('saving padded image!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            cv2.imwrite('padded_img.jpg', img)
            width = 64
        
        return img, joint_all, mask_miss[...,None], mask_all[...,None] if "COCO" in joint_all['dataset'] else None

    def create_masks(img_anns,img_shape):
        h, w, c = img_shape

        mask_all = np.zeros((h, w), dtype=np.uint8)
        mask_miss = np.zeros((h, w), dtype=np.uint8)
        flag = 0
        for p in img_anns:
            seg = p["segmentation"]

            if p["iscrowd"] == 1:
                mask_crowd = coco.annToMask(p)
                temp = np.bitwise_and(mask_all, mask_crowd)
                mask_crowd = mask_crowd - temp
                flag += 1
                continue
            else:
                mask = coco.annToMask(p)

            mask_all = np.bitwise_or(mask, mask_all)

            if p["num_keypoints"] <= 0:
                mask_miss = np.bitwise_or(mask, mask_miss)

        if flag<1:
            mask_miss = np.logical_not(mask_miss)
        elif flag == 1:
            mask_miss = np.logical_not(np.bitwise_or(mask_miss, mask_crowd))
            mask_all = np.bitwise_or(mask_all, mask_crowd)
        else:
            raise Exception("crowd segments > 1")

        return mask_all * 255, mask_miss * 255
    