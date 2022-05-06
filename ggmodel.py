from tensorflow.keras.layers import Conv2D,BatchNormalization,MaxPooling2D,Input,Reshape,Concatenate,ReLU,ELU
from tensorflow.keras.models import Model
# from tensorflow.keras.activations import gelu


def detector(inputs,num_classes,size,ratio):

    x = inputs
    num_anchors = len(size) + len(ratio)

    # Location.
    num_loc_pred = num_anchors * 4
    loc_pred = Conv2D(num_loc_pred,(3,3),padding = 'same')(x)
#     loc_pred = tf.reshape(loc_pred,[-1]+list(loc_pred.shape)[1:-1]+[num_anchors, 4])
    loc_pred =  Reshape((-1, 4))(loc_pred)
    
    # Class prediction.
    num_cls_pred = num_anchors * (num_classes+1)
    cls_pred = Conv2D(num_cls_pred,(3,3),padding = 'same')(x)

#     cls_pred = tf.reshape(cls_pred,[-1]+list(cls_pred.shape)[1:-1]+[num_anchors, num_classes+1])
    cls_pred = Reshape((-1, num_classes+1))(cls_pred)
 

    
    return cls_pred, loc_pred

def GG_SSD(num_classes,anchor_sizes,anchor_ratios):
    end_points = {}
    inputs = Input(shape = (512,512,3))
    
    x = Conv2D(64,(3,3),padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = MaxPooling2D()(x)
    
    x = Conv2D(64,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = MaxPooling2D()(x)
    
    x = Conv2D(128,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = MaxPooling2D()(x) #8
    end_points['step-8'] = x
    
    x = Conv2D(128,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = MaxPooling2D()(x) #16
    end_points['step-16'] = x
    
    x = Conv2D(256,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = MaxPooling2D()(x) #32
    end_points['step-32'] = x
    
    x = Conv2D(256,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = MaxPooling2D()(x) #64
    end_points['step-64'] = x
    
    x = Conv2D(256,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = MaxPooling2D()(x) #128
    end_points['step-128'] = x
    
    logits = []
    localisations = []
    
    layer_predict = ['step-8','step-16','step-32','step-64','step-128']
    for i, layer in enumerate(layer_predict):
        p, l = detector(end_points[layer],
                        num_classes,
                        anchor_sizes[i],
                        anchor_ratios[i])
        logits.append(p)
        localisations.append(l)
        
    output_logits = Concatenate(axis = 1)(logits)
    output_localisations = Concatenate(axis = 1)(localisations)
    outputs = Concatenate(axis = 2)([output_logits, output_localisations])
#     print(outputs.shape)
    
    model = Model(inputs = inputs,outputs = outputs) 

    return model 