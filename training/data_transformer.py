import tensorflow as tf
import cv2
import copy
import random
from pointOperations import Point,addPoints,addScalar,mulScalar
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
    np = 0
    is_table_set_ = False

    def __init__(self,transforParam):
        param = transforParam
        self.np_ann = param.num_parts_in_annot
        self.np = param.num_parts
    
    # Might not need random seed
    def initRand():
        needs_rand = (param.mirror || param.crop_size)
    
    def transform(filename=None,anno_path=None):
        AugmentSelection aug = AugmentSelection(False,0.0,(),0)
        coco = COCO(anno_path)

        # Read image
        image_decoded = cv2.imread(filename.decode())
        # BGR -> RGB
        image_decoded = cv2.cvtColor(image_decoded,cv2.COLOR_BGR2RGB)
        
        # create miss mask
        miss_mask = create_miss_mask()

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
        # TODO(someone): implement 
        meta = ReadMetaData() # TODO(Mike/Charles): implement function
        if(param.transform_body_joint):
            TransformMetaJoints(meta)
        
        # Start transformation
        img_aug = np.zeros(param.crop_size_y,param.crop_size_x,3)
        mask_miss_aug = None
        img_temp, img_temp2, img_temp3 = None # size determined by scale

        aug.scale = AugmentationScale(img, img_temp, mask_miss, meta)
        aug.degree = AugmentationRotate(img_temp, img_temp2, mask_miss, meta)
        aug.crop = AugmentationCropped(img_temp2, img_temp3, mask_miss, mask_miss_aug, meta)
        aug.flip = AugmentationFlip(img_temp3, img_aug, mask_miss_aug, meta)

        mask_miss_aug = cv2.resize(mask_miss_aug,(0,0),fx=1.0/param.stride,fy=1.0/param.stride,interpolation=cv2.INTER_CUBIC)

        # TODO(isaac) copy transformed img (img_aug) into transformed_data
        # TODO(someone) implement GenerateLabelMap 

        

    def create_miss_mask():
        # TODO(someone): implement function
    
    def TransformMetaJoints(meta=None):
        TransformJoints(meta.joint_self)
        for j in meta.joint_others:
            TransformJoints(j)

    def TransformJoints(j=None):
        # Coco dataset
        jo = copy.deepcopy(j)
        if(np == 56):
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

    def AugmentationScale(img_src,img_temp,mask_miss,meta):
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

        for i in range(np):
            if(meta.joint_self.joints[i] is not None):
                meta.joint_self.joints[i] = mulScalar(meta.joint_self.joints[i],scale)
        for p in range(meta.num_other_people)
            meta.objpos_other[p] = mulScalar(meta.objpos_other[p],scale)
            for i in range(np):
                if(meta.joint_others[p].joints[i] i not None):
                    meta.joint_others[p].joints[i] = mulScalar(meta.joint_others[p].joints[i],scale)
        return scale_multiplier

    def AugmentationRotate(img_src, img_dst, mask_miss, meta):
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
        for i in range(np):
            if(meta.joint_self.joints[i] is not None):
                RotatePoint(meta.joint_self.joints[i], R)
        for p in range(meta.num_other_people):
            RotatePoint(meta.objpos_other[p],R)
            for i in range(np):
                if(meta.joint_others[p].joints[i] is not None):
                    RotatePoint(meta.joint_others[p].joints[i],R)
        return degree
    
    def AugmentationCropped(img_src, img_dst, mask_miss, mask_miss_aug, meta):
        dice_x = random.random()
        dice_y = random.random()
        
        x_offset = (dice_x - 0.5) * 2 * param.center_perterb_max
        y_offset = (dice_y - 0.5) * 2 * param.center_perterb_max
        
        center = addPoints(meta.objpos,Point(x_offset,y_offset))
    
        offset_left = -(center.x - (param.crop_size_x/2))
        offset_up = -(center.y - (param.crop_size_y/2))

        img_dst = np.zeros((param.crop_size_y, param.crop_size_x, 3)) + (128,128,128)
        mask_miss_aug = np.zeros((param.crop_size_y, param.crop_size_x,1)) + (255)
        
        for i in range(param.crop_size_y):
            for j in range(param.crop_size_x):
                coord_x_on_img = center.x - param.crop_size_x/2 + j
                coord_y_on_img = center.y - param.crop_size_y/2 + i
                if(OnPlane(Point(coord_x_on_img, coord_y_on_img),img_src.shape)):
                    img_dst[i][j] = img_src[coord_y_on_img][coord_x_on_img]
                    mask_miss_aug[i][j] = mask_miss[coord_y_on_img][coord_x_on_img]
        
        offset = Point(offset_left,offset_up)
        meta.objpos = addPoints(meta.objpos,offset)
        for i in range(np):
            if(meta.joint_self.joints[i] is not None):
                meta.joint_self.joints[i] = addPoints(meta.joint_self.joints[i],offset)
        for p in range(meta.num_other_people):
            meta.objpos_other[p] = addPoints(meta.objpos_other[p],offset)
            for i in range(np):
                if(meta.joint_others[p].joints[i] is not None):
                    meta.joint_others[p].joints[i] = addPoints(meta.joint_others[p].joints[i],offset)
        
        return Point(x_offset,y_offset)

    def AugmentationFlip(img_src,img_aug,mask_miss,meta):
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
           mask_miss = cv2.flip(mask_miss,1)
           meta.objpos.x = w - 1 - meta.objpos.x
           
           for i in range(np):
               if(meta.joint_self.joints[i] is not None):
                   (meta.joint_self.joints[i]).x = w - 1 - (meta.joint_self.joints[i]).x
            
            if(param.transform_body_joint):
                SwapLeftRight(meta.joint_self)
        
            for p in range(meta.num_other_people):
                meta.objpos_other[p].x = w - 1 - meta.objpos_other[p].x
                
                for i in range(np):
                    if(meta.joint_others[p].joints[i] is  not None):
                        meta.joint_others[p].joints[i].x = w - 1 - meta.joint_others[p].joints[i].x
                
                if(param.transform_body_joint):
                    SwapLeftRight(meta.joint_others[p])
        else:
            img_aug = np.copy(img_src)
        
        return doflip
    
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
        if(np == 56):
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