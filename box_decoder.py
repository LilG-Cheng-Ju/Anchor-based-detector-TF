from label_encoder import generate_anchors
import numpy as np
import cv2

class Box_decoder:
    def __init__(self, image_shape, feat_shape, sizes, ratios, steps, score_threshold = 0.8, iou_threshold = 0.7):
        self.img_shape = image_shape
        self.feat_shape = feat_shape
        self.anchor_sizes = sizes
        self.anchor_ratios = ratios
        self.anchor_steps = steps
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

        anchors = generate_anchors(self.img_shape, 
                                   self.feat_shape, 
                                   self.anchor_sizes, 
                                   self.anchor_ratios, 
                                   self.anchor_steps)
        
        anchor_pred = []
        #把default-anchor變成一條線(同預測結果)
        for anchor_onelayer in anchors:
            
            yref, xref, href, wref = anchor_onelayer
            
            box_num = len(xref)*len(yref)
            scale_num = len(href)
            
            #np.repeat --> 可以把[111,222]變成[111,111,111,222,222,222]
            xref = np.repeat(np.reshape(xref, [-1, 1]),scale_num)
            yref = np.repeat(np.reshape(yref, [-1, 1]),scale_num)
            #np.tile --> 可以把[111,222]變成[111,222,111,222,111,222]
            href = np.tile(href,box_num)
            wref = np.tile(wref,box_num)

            output = np.stack([yref,xref,href,wref],axis = 1)
            anchor_pred.append(output)
        self.anchor_pred = np.concatenate(anchor_pred,axis = 0)
        
    def __call__(self, result):
        
        Box_loc = result.numpy()[0][:,-4:]
        Box_score = result.numpy()[0][:,:-4]
        
        Box_score = self.softmax(Box_score)
        Box_class = np.argmax(Box_score,axis=1)

        condition = np.logical_and(Box_class != 0, np.max(Box_score,axis=1) > self.score_threshold)
        #select為score > threshold的正樣本索引
        select = np.where(condition)
        #用索引把需要的Box選出來
        final_loc = Box_loc[select]
        final_class = Box_class[select]
        final_score = np.max(Box_score[select],axis = 1)
        final_anchor = self.anchor_pred[select]
        
        cx = final_loc[:, 0] * final_anchor[:,3] * 0.1 + final_anchor[:,1]
        cy = final_loc[:, 1] * final_anchor[:,2] * 0.1 + final_anchor[:,0]
        w = final_anchor[:,3] * np.exp(final_loc[:, 2] * 0.2)
        h = final_anchor[:,2] * np.exp(final_loc[:, 3] * 0.2)

        bboxes = np.zeros(shape = (final_loc.shape[0],6))
        bboxes[:, 0] = cy - h / 2.
        bboxes[:, 1] = cx - w / 2.
        bboxes[:, 2] = cy + h / 2.
        bboxes[:, 3] = cx + w / 2.
        bboxes[:, 4] = final_class
        bboxes[:, 5] = final_score
        
        bboxes = self.nms(bboxes,self.iou_threshold)
        #(ymin, xmin, ymax, xmax, class, score)
        return bboxes
    
    def label_box(self, result):
        Box_loc = result.numpy()[0][:,-4:]
        Box_score = result.numpy()[0][:,1]
        Box_class = result.numpy()[0][:,0]
        
        condition = np.logical_and(Box_class != 0, np.max(Box_score) > 0.3)
        select = np.where(condition)
        
        #用索引把需要的Box選出來
        final_loc = Box_loc[select]
        final_class = Box_class[select]
        final_score = Box_score[select]
        final_anchor = self.anchor_pred[select]
        
        cx = final_loc[:, 0] * final_anchor[:,3] * 0.1 + final_anchor[:,1]
        cy = final_loc[:, 1] * final_anchor[:,2] * 0.1 + final_anchor[:,0]
        w = final_anchor[:,3] * np.exp(final_loc[:, 2] * 0.2)
        h = final_anchor[:,2] * np.exp(final_loc[:, 3] * 0.2)

        bboxes = np.zeros(shape = (final_loc.shape[0],6))
        bboxes[:, 0] = cy - h / 2.
        bboxes[:, 1] = cx - w / 2.
        bboxes[:, 2] = cy + h / 2.
        bboxes[:, 3] = cx + w / 2.
        bboxes[:, 4] = final_class
        bboxes[:, 5] = final_score
        
#         bboxes = self.nms(bboxes,self.iou_threshold)
        #(ymin, xmin, ymax, xmax, class, score)
        return bboxes
        
    
    def softmax(self,x):
    
        Max = np.max(x,axis=1,keepdims=True) #returns max of each row and keeps same dims
        e_x = np.exp(x - Max) #subtracts each row with its max value
        Sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
        f_x = e_x / Sum 
        return f_x
    
    def nms(self,data_set, threshold):

        # If no bounding boxes, return empty list
        if len(data_set) == 0:
            return [], []

        data_set = np.array(data_set)
        start_x = data_set[:, 0]
        start_y = data_set[:, 1]
        end_x = data_set[:, 2]
        end_y = data_set[:, 3]
        score = data_set[:, 5]

        # Picked bounding boxes
        picked_boxes = []

        # Compute areas of bounding boxes
        areas = (end_x - start_x + 1) * (end_y - start_y + 1)

        # Sort by confidence score of bounding boxes
        order = np.argsort(score)

        # Iterate bounding boxes
        while order.size > 0:
            # The index of largest confidence score
            index = order[-1]

            # Pick the bounding box with largest confidence score
            picked_boxes.append(data_set[index])

            # Compute ordinates of intersection-over-union(IOU)
            x1 = np.maximum(start_x[index], start_x[order[:-1]])
            x2 = np.minimum(end_x[index], end_x[order[:-1]])
            y1 = np.maximum(start_y[index], start_y[order[:-1]])
            y2 = np.minimum(end_y[index], end_y[order[:-1]])

            # Compute areas of intersection-over-union
            w = np.maximum(0.0, x2 - x1 + 1)
            h = np.maximum(0.0, y2 - y1 + 1)
            intersection = w * h

            # Compute the ratio between intersection and union
            ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

            left = np.where(ratio < threshold)
            order = order[left]
            
        return picked_boxes

def boxVisualization(Boxes,image,image_size = (512,512)):
    img = image.copy()
    for box in Boxes:
        ymin = box[0]*image_size[0]
        xmin = box[1]*image_size[1]
        ymax = box[2]*image_size[0]
        xmax = box[3]*image_size[1]
        img = cv2.rectangle(img,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(230,255,255),2)
    return img