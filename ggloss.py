import tensorflow as tf
class SSD_Loss_v2:
    def __init__(self,num_class):
        self.num_class = num_class+1
        
    def compute_loss(self, y_true, y_pred):
        
        
        logits = y_pred[:,:,:self.num_class]
        gclasses = y_true[:,:,0]
        gscores = y_true[:,:,1]
        localisations = y_pred[:,:,self.num_class:]
        glocalisations = y_true[:,:,2:]
        
        batch_size = (logits[0].shape)[0]

        
        logits = tf.reshape(logits,(-1,self.num_class))
        localisations = tf.reshape(localisations,(-1,4))
        gclasses = tf.reshape(gclasses,(-1,))
        gscores = tf.reshape(gscores,(-1,1))
        glocalisations = tf.reshape(glocalisations,(-1,4))
        
        dtype = logits.dtype
        pmask = gscores > 0.25   # IoU是否大於0.5 #0.3好像會丟失目標
        fpmask = tf.cast(pmask, dtype)
        fpmask = tf.reshape(fpmask,(-1,))
        
        n_positives = tf.reduce_sum(fpmask)  # 正樣本數目
        no_classes = tf.cast(pmask, tf.int32)
        no_classes = tf.reshape(no_classes,(-1,))

        predictions = tf.keras.activations.softmax(logits)
        predictions = tf.reshape(predictions[:,0],(-1,1))

        nmask = tf.logical_and(tf.logical_not(pmask),  # IoU小於0.5並大於-0.5的負樣本
                               gscores > -0.8)
        fnmask = tf.cast(nmask, dtype)
        nvalues = tf.where(nmask,      # True時為背景概率，False時為1.0
                           predictions,   # 0 是 background
                           1. - fnmask)
        nvalues_flat = tf.reshape(nvalues, (-1,))

        # Number of negative entries to select.
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)  # 所有供選擇的負樣本數目
        n_neg = tf.cast(3. * n_positives, tf.int32) + batch_size #3為正負樣本的比例
        n_neg = tf.minimum(n_neg, max_neg_entries)  # 負樣本的個數

        val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg, sorted=False)  # 按順序排獲取前k個值，以及對應id
        max_hard_pred = -val[-1]  # 負樣本的背景概率閾值
        # Final negative mask.
        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)  # 交併比小於0.5並大於-0.5的負樣本，且概率小於max_hard_pred
        fnmask = tf.cast(nmask, dtype)
        fnmask = tf.reshape(fnmask,(-1,))
        
        gclasses = tf.cast(gclasses,tf.int32)
        ploss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                              labels=gclasses)

        ploss = tf.divide(tf.reduce_sum(ploss * fpmask), batch_size)  # fpmask是正樣本的mask，正1，負0
        
        
        #負樣本損失
        nloss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                              labels=no_classes)
        nloss = tf.divide(tf.reduce_sum(nloss * fnmask), batch_size)  # fnmask是負樣本的mask，負為1，正為0
        
        
        weights = tf.expand_dims(1.0 * fpmask, axis=-1)
        absx = tf.abs(localisations - glocalisations)
        minx = tf.minimum(absx, 1)
        r = 0.5 * ((absx - 1) * minx + absx)
        Lloss = tf.reduce_sum(r * weights,axis = -1)
        
        return ploss+nloss+Lloss