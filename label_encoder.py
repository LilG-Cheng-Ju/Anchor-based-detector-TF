import numpy as np
import tensorflow as tf
import math
from tqdm import tqdm

def get_feat_shape(image_size,step):
    feat_shape = []
    for i in step:
        feat_shape.append([int(image_size[0]/i),int(image_size[1]/i)])
    return tuple(feat_shape)

def get_bbox_size(image_size,m,Smin,Smax):
    ref , size = [],[]
    for i in range(m):
        sk = Smin + ((Smax - Smin)/(m-1))*i
        ref.append(round(sk*image_size[0],2))
    for j in range(len(ref)):       
        if j < len(ref)-1:
            skp = round((ref[j]*ref[j+1])**0.5,2)
            size.append([ref[j],skp])
        else:
            size.append([ref[j],round(ref[j]*1.1,2)])
    return tuple(size)

def anchor_one_layer(img_shape,  # 原始影象shape
                         feat_shape,  # 特徵圖shape
                         sizes,  # 預設box大小
                         ratios,  # 預設box長寬比
                         step,  # 特徵圖對應原圖縮小的比例
                         offset=0.5,
                         dtype=np.float32):

    # 計算BBox中心坐標
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0] #除以原影像尺寸取得相對座標
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)  # 預設框的個數
    h = np.zeros((num_anchors, ), dtype=dtype)  # 初始化高
    w = np.zeros((num_anchors, ), dtype=dtype)  # 初始化寬
    # Add first anchor boxes with ratio=1.
    h[0] = sizes[0] / img_shape[0]  # 新增長寬比為1的預設框
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]  # 新增一組特殊的預設框，長寬比為1，大小為sqrt（s（i） + s（i+1））
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):  # 新增不同比例的DBox（ratios中不含1）
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
    return y, x, h, w

def generate_anchors(img_shape, 
                  layers_shape, 
                  anchor_sizes, 
                  anchor_ratios, 
                  anchor_steps, 
                  offset=0.5, 
                  dtype=np.float32):

    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors

def label_encoder_one_layer(decoded_label,
                            anchors_layer,
                            image_shape,
                            dtype = tf.float32):

    yref, xref, href, wref = anchors_layer.copy()
    ymin = yref - href / 2.  # 轉換到預設框的左上角座標以及右下角座標
    xmin = xref - wref / 2.
    ymax = yref + href / 2.
    xmax = xref + wref / 2.
    vol_anchors = (xmax - xmin) * (ymax - ymin)
    shape = (yref.shape[0], yref.shape[1], href.size)

    dtype = tf.float32
    feat_labels = tf.zeros(shape, dtype=tf.int64)  # 存放預設框匹配的GTbox標籤
    feat_scores = tf.zeros(shape, dtype=dtype)  # 存放預設框與匹配的GTbox的IOU（交併比）

    feat_ymin = tf.zeros(shape, dtype=dtype)  # 存放預設框匹配到的GTbox的座標資訊
    feat_xmin = tf.zeros(shape, dtype=dtype)
    feat_ymax = tf.ones(shape, dtype=dtype)
    feat_xmax = tf.ones(shape, dtype=dtype)
    
    def anchor_GT_IoU(bbox):  # 計算IoU

        int_ymin = tf.maximum(ymin, bbox[1])
        int_xmin = tf.maximum(xmin, bbox[0])
        int_ymax = tf.minimum(ymax, bbox[3])
        int_xmax = tf.minimum(xmax, bbox[2])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        union_vol = vol_anchors - inter_vol \
            + (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        IOU = tf.divide(inter_vol, union_vol)
        return IOU

    #從每個annotation中抓出需要的box及label
    for label,box in zip(decoded_label[3],decoded_label[4]):

        IoU = anchor_GT_IoU(box/image_shape[0])   #計算每個位置與當前GT框的IoU
        mask = tf.greater(IoU, feat_scores)  #每個位置只記錄最大IoU的Box
        imask = tf.cast(mask, tf.int64) #0，1整數  
        fmask = tf.cast(mask, dtype)  #0，1浮點數

        feat_labels = imask * label + (1 - imask) * feat_labels  # 當imask為1時更新類別
        feat_scores = tf.where(mask, IoU, feat_scores)  # 當mask為true時更新IoU
        feat_ymin = fmask * box[1]/image_shape[0] + (1 - fmask) * feat_ymin  # 當fmask為1.0時更新配對坐標
        feat_xmin = fmask * box[0]/image_shape[0] + (1 - fmask) * feat_xmin
        feat_ymax = fmask * box[3]/image_shape[0] + (1 - fmask) * feat_ymax
        feat_xmax = fmask * box[2]/image_shape[0] + (1 - fmask) * feat_xmax

    feat_cx = (feat_xmax + feat_xmin) / 2.
    feat_cy = (feat_ymax + feat_ymin) / 2.
    feat_w = feat_xmax - feat_xmin
    feat_h = feat_ymax - feat_ymin
    #paper中的公式
    target_dx = ((feat_cx - xref)/wref)/0.1 #聽說最後除這個可以加快收斂
    target_dy = ((feat_cy - yref)/href)/0.1
    target_dw = tf.math.log(feat_w/wref)/0.2
    target_dh = tf.math.log(feat_h/href)/0.2
    feat_localizations = tf.stack([target_dx, target_dy, target_dw, target_dh], axis=-1)
    
    return feat_labels, feat_localizations, feat_scores

def Label_encoder_one_file(label,
                  image_shape, 
                  layers_shape, 
                  anchor_sizes, 
                  anchor_ratios, 
                  anchor_steps, 
                  offset=0.5, 
                  dtype=np.float32):
    target_labels = []  #有配對到GTbox的Dbox類別
    target_localizations = []  #有配對到GTbox的Dbox坐標
    target_scores = []  #記錄各點IoU
    anchors = generate_anchors(image_shape, 
                               layers_shape, 
                               anchor_sizes, 
                               anchor_ratios, 
                               anchor_steps, 
                               offset=0.5, 
                               dtype=np.float32)
    for i, anchors_layer in enumerate(anchors):  # 遍歷每個預測層的預設框
                t_labels, t_loc, t_scores = label_encoder_one_layer(label, 
                                                                    anchors_layer,
                                                                    image_shape)
                target_labels.append(t_labels)  # 匹配到的ground truth box對應標籤
                target_localizations.append(t_loc)  # 預設框與匹配到的ground truth box的座標差異
                target_scores.append(t_scores)  # 預設框與匹配到的ground truth box的IOU（交併比）
    return target_labels, target_localizations, target_scores

def Label_encoder(decode_labels,
                  image_shape, 
                  layers_shape, 
                  anchor_sizes, 
                  anchor_ratios, 
                  anchor_steps, 
                  offset=0.5, 
                  dtype=np.float32):
    
    target_labels = []  #有配對到GTbox的Dbox類別
    target_localizations = []  #有配對到GTbox的Dbox坐標
    target_scores = []  #記錄各點IoU
    for label in decode_labels:
        t_labels, t_loc, t_scores = Label_encoder_one_file(label,
                                                           image_shape, 
                                                           layers_shape, 
                                                           anchor_sizes, 
                                                           anchor_ratios, 
                                                           anchor_steps, 
                                                           offset=0.5, 
                                                           dtype=np.float32)
        target_labels.append(t_labels)  # 匹配到的ground truth box對應標籤
        target_localizations.append(t_loc)  # 預設框與匹配到的ground truth box的座標差異
        target_scores.append(t_scores)
    return target_labels, target_localizations, target_scores


def label_encoder_one_layer_v2(decoded_label,
                            anchors_layer,
                            image_shape,
                            dtype = tf.float32):
    
    yref, xref, href, wref = anchors_layer
    ymin = yref - href / 2.  # 轉換到預設框的左上角座標以及右下角座標
    xmin = xref - wref / 2.
    ymax = yref + href / 2.
    xmax = xref + wref / 2.
    vol_anchors = (xmax - xmin) * (ymax - ymin)
    shape = (yref.shape[0], yref.shape[1], href.size)

    dtype = tf.float32
    feat_labels = tf.zeros(shape, dtype=tf.int64)  # 存放預設框匹配的GTbox標籤
    feat_scores = tf.zeros(shape, dtype=dtype)  # 存放預設框與匹配的GTbox的IOU（交併比）

    feat_ymin = tf.zeros(shape, dtype=dtype)  # 存放預設框匹配到的GTbox的座標資訊
    feat_xmin = tf.zeros(shape, dtype=dtype)
    feat_ymax = tf.ones(shape, dtype=dtype)
    feat_xmax = tf.ones(shape, dtype=dtype)
    
    def anchor_GT_IoU(bbox):  # 計算IoU

        int_ymin = tf.maximum(ymin, bbox[1])
        int_xmin = tf.maximum(xmin, bbox[0])
        int_ymax = tf.minimum(ymax, bbox[3])
        int_xmax = tf.minimum(xmax, bbox[2])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        union_vol = vol_anchors - inter_vol \
            + (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        IOU = tf.divide(inter_vol, union_vol)
        return IOU

    #從每個annotation中抓出需要的box及label
    for label,box in zip(decoded_label[3],decoded_label[4]):

        IoU = anchor_GT_IoU(box/image_shape[0])   #計算每個位置與當前GT框的IoU
        mask = tf.greater(IoU, feat_scores)  #每個位置只記錄最大IoU的Box
        imask = tf.cast(mask, tf.int64) #0，1整數  
        fmask = tf.cast(mask, dtype)  #0，1浮點數

        feat_labels = imask * label + (1 - imask) * feat_labels  # 當imask為1時更新類別
        feat_scores = tf.where(mask, IoU, feat_scores)  # 當mask為true時更新IoU
        feat_ymin = fmask * box[1]/image_shape[0] + (1 - fmask) * feat_ymin  # 當fmask為1.0時更新配對坐標
        feat_xmin = fmask * box[0]/image_shape[0] + (1 - fmask) * feat_xmin
        feat_ymax = fmask * box[3]/image_shape[0] + (1 - fmask) * feat_ymax
        feat_xmax = fmask * box[2]/image_shape[0] + (1 - fmask) * feat_xmax

    feat_cx = (feat_xmax + feat_xmin) / 2.
    feat_cy = (feat_ymax + feat_ymin) / 2.
    feat_w = feat_xmax - feat_xmin
    feat_h = feat_ymax - feat_ymin
    #paper中的公式
    target_dx = ((feat_cx - xref)/wref)/0.1 #聽說最後除這個可以加快收斂
    target_dy = ((feat_cy - yref)/href)/0.1
    target_dw = tf.math.log(feat_w/wref)/0.2
    target_dh = tf.math.log(feat_h/href)/0.2
    feat_localizations = tf.stack([target_dx, target_dy, target_dw, target_dh], axis=-1)
    
    feat_labels = tf.reshape(feat_labels,(-1,1))
    feat_localizations = tf.reshape(feat_localizations,(-1,4))
    feat_scores = tf.reshape(feat_scores,(-1,1))

    
    feat_labels = tf.expand_dims(feat_labels, axis=0)
    feat_localizations = tf.expand_dims(feat_localizations, axis=0)
    feat_scores = tf.expand_dims(feat_scores, axis=0)
    
    return feat_labels, feat_localizations, feat_scores

def Label_encoder_one_file_v2(label,
                  image_shape, 
                  layers_shape, 
                  anchor_sizes, 
                  anchor_ratios, 
                  anchor_steps,
                  anchors,
                  offset=0.5, 
                  dtype=np.float32):
    target_labels = []  #有配對到GTbox的Dbox類別
    target_localizations = []  #有配對到GTbox的Dbox坐標
    target_scores = []  #記錄各點IoU
#     anchors = generate_anchors(image_shape, 
#                                layers_shape, 
#                                anchor_sizes, 
#                                anchor_ratios, 
#                                anchor_steps, 
#                                offset=0.5, 
#                                dtype=np.float32)

    for i, anchors_layer in enumerate(anchors):  # 遍歷每個預測層的預設框

        t_labels, t_loc, t_scores = label_encoder_one_layer_v2(label, 
                                                               anchors_layer,
                                                               image_shape)
        target_labels.append(t_labels)  # 匹配到的ground truth box對應標籤
        target_localizations.append(t_loc)  # 預設框與匹配到的ground truth box的座標差異
        target_scores.append(t_scores)  # 預設框與匹配到的ground truth box的IOU
                
#     print(target_labels[0].shape,target_labels[1].shape)
    target_labels = tf.concat(target_labels,axis=1)
    target_localizations = tf.concat(target_localizations,axis=1)
    target_scores = tf.concat(target_scores,axis=1)
    
    target_labels = tf.cast(target_labels,tf.float32)
    
    output = tf.concat([target_labels,target_scores,target_localizations],axis = 2)
                
    return output

def Label_encoder_v2(decode_labels,
                  image_shape, 
                  layers_shape, 
                  anchor_sizes, 
                  anchor_ratios, 
                  anchor_steps,
                  anchors,
                  offset=0.5, 
                  dtype=np.float32):
    output = []
    for label in tqdm(decode_labels):
        label_tensor = Label_encoder_one_file_v2(label,
                                              image_shape, 
                                              layers_shape, 
                                              anchor_sizes, 
                                              anchor_ratios, 
                                              anchor_steps,
                                              anchors,
                                              offset=0.5, 
                                              dtype=np.float32)
        output.append(label_tensor)
    output = tf.concat(output,axis = 0)
    return output