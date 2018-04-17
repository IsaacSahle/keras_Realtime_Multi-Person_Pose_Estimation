from data_transformer import DataTransformer
from pycocotools.coco import COCO
import os
import os.path


class TransformationParameter(object): 
    mirror = False
    crop_size = 0
    stride = 4
    flip_prob = 0.5
    max_rotate_degree = 5.0
    crop_size_x = 368
    crop_size_y = 368
    scale_prob = 0.5
    scale_min = 0.9
    scale_max = 1.1
    target_dist = 1.0
    center_perterb_max = 10.0
    sigma = 7.0
    clahe_tile_size = 8.0
    clahe_clip_limit = 4.0
    do_clahe = False
    num_parts = 14
    num_parts_in_annot = 16
    num_total_augs = 82
    aug_way = "rand"
    gray = 0
    transform_body_joint = True

def preprocess(train=None,data=None):
    params = TransformationParameter()
    params.stride = 8
    params.crop_size_x = 368
    params.crop_size_y = 368
    params.target_dist = 0.6
    params.scale_prob = 1
    params.scale_min = 0.5
    params.scale_max = 1.1
    params.max_rotate_degree = 40
    params.center_perterb_max = 40
    params.do_clahe = False
    params.num_parts_in_annot = 17
    params.num_parts = 56
    params.mirror = True

    dataTransformer = DataTransformer(params)
    np = 2*(params.num_parts+1)
    stride = params.stride
    grid_x = params.crop_size_x / stride
    grid_y = params.crop_size_y / stride
    channelOffset = grid_y * grid_x
    vec_channels = 38
    heat_channels = 19
    ch = vec_channels + heat_channels
    start_label_data = (params.num_parts+1) * channelOffset

    transformed_data = [] # size: params.crop_size_x * params.crop_size_y * 3
    transformed_label = [] # size: grid_x * grid_y * np

    # Transformation
    print("Transforming...")
    data_img,mask_img,label = dataTransformer.transform(data)

    return data_img, mask_img,label

def _parse_tr_data(data=None): # data[0] = img, data[1] = joint_all, data[2] = mask_miss, data[3] = mask_all
    print("data")
    print(str(data[0]))
    print(data[1])
    print(data[2])
    print(data[3])

    data[0] = data[0].decode("utf-8")
    data[1] = data[1].decode("utf-8")
    data[2] = data[2].decode("utf-8")
    data[3] = data[3].decode("utf-8")
    
    data_img, mask_img, label = preprocess(True, data)

    # TODO(someone): Kwang, please not that since the result from preprocess is not
    # working, this portion of code has yet to be tested

    # Expected shapes
    # *** data_img -> (3,368,368) ***
    # *** mask_img -> (46,46) ***
    # *** label -> (57,46,46) ***
    # image
    data_img = np.transpose(data_img, (1, 2, 0))
    batches_x[sample_idx]=dta_img[np.newaxis, ...]

    # mask - the same for vec_weights, heat_weights
    vec_weights = np.repeat(mask_img[:,:,np.newaxis], self.vec_num, axis=2)
    heat_weights = np.repeat(mask_img[:,:,np.newaxis], self.heat_num, axis=2)

    batches_x1[sample_idx]=vec_weights[np.newaxis, ...]
    batches_x2[sample_idx]=heat_weights[np.newaxis, ...]

    # label
    vec_label = label[:self.split_point, :, :]
    vec_label = np.transpose(vec_label, (1, 2, 0))
    heat_label = label[self.split_point:, :, :]
    heat_label = np.transpose(heat_label, (1, 2, 0))

    batches_y1[sample_idx]=vec_label[np.newaxis, ...]
    batches_y2[sample_idx]=heat_label[np.newaxis, ...]

    sample_idx += 1

    if sample_idx == self.batch_size:
        sample_idx = 0

        batch_x = np.concatenate(batches_x)
        batch_x1 = np.concatenate(batches_x1)
        batch_x2 = np.concatenate(batches_x2)
        batch_y1 = np.concatenate(batches_y1)
        batch_y2 = np.concatenate(batches_y2)

        return [batch_x, batch_x1,  batch_x2], \
                [batch_y1, batch_y2,
                batch_y1, batch_y2,
                batch_y1, batch_y2,
                batch_y1, batch_y2,
                batch_y1, batch_y2,
                batch_y1, batch_y2]

def _parse_va_data(filename=None):
    return 0
