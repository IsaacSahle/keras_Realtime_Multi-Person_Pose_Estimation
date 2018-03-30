import tensorflow as tf
import cv2
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
        AugmentSelection aug = AugmentSelection(False,0.0,?Size(),0)
        coco = COCO(anno_path)

        # Read image
        image_str = tf.read_file(filename)
        image_decoded = tf.image.decoded_image(image_str)
        
        # create miss mask
        miss_mask = create_miss_mask()

        # Perform CLAHE
        if(param.do_clahe):
            # *** Currently flase all the time, look into later
            # Code snippet
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            # cl1 = clahe.apply(img)
        
        # Convert to grayscale
        if(param.gray == 1):
            # Not sure why this is done in C++ server
            # cv::cvtColor(img, img, CV_BGR2GRAY);
            # cv::cvtColor(img, img, CV_GRAY2BGR);
        if(param.transform_body_joint):
            

        

    def create_miss_mask():
        
        

}