import mxnet as mx
import numpy as np
from mxnet.contrib import autograd
import gc

class FPNROIPoolingOperator(mx.operator.CustomOp):
    def __init__(self, feat_strides, pooled_height, pooled_width, output_dim):
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
        self.feat_strides = feat_strides
        self.output_dim = output_dim
        self.in_grad_hist_list = []
        self.num_strides = len(self.feat_strides)
        self.roi_pool = [None for _ in range(self.num_strides)]
        self.feat_idx = [None for _ in range(self.num_strides)]
    
    def forward(self, is_train, req, in_data, out_data, aux):
        rois = in_data[-1].asnumpy()
        w = rois[:,3] - rois[:,1] + 1
        h = rois[:,4] - rois[:,2] + 1
        feat_id = np.clip(np.floor(2 + np.log2(np.sqrt(w*h)/224)),0,len(self.feat_strides) - 1)
        pyramid_idx = []

        rois_p = [None for _ in range(self.num_strides)]
        for i in range(self.num_strides):
            self.feat_idx[i] = np.where(feat_id == i)[0]  #第i个尺度的roi的下标
            if len(self.feat_idx[i]) == 0:
                rois_p[i] = np.zeros((1,5))
                pyramid_idx.append(-1)
            else:
                rois_p[i] = rois[self.feat_id[i]]     #rois_p[i] 第i个尺度所有的roi
                pyramid_idx.append(self.feat_idx[i])  #pyramid[i]:第i个尺度的roi的下标
        rois_idx = np.argsort(np.hstack(pyramid_idx))[-rois.shape[0]:]
        if is_train:
            for i in range(self.num_strides):
                self.in_grad_hist_list.append(mx.nd.zeros_like(in_data[i]))

            autograd.mark_variables([in_data[i] for i in range(self.num_strides)],self.in_grad_hist_list)
            with autograd.train_section():
                for i in range(self.num_strides):
                    self.roi_pool[i] = mx.nd.ROIPooling(in_data[i],mx.nd.array(rois_p[i],in_data[i].context),(7,7),spatial_scale = 1.0 / self.feat_strides[i])

            roi_pool = mx.nd.concatenate(self.roi_pool, axis = 0)
        else:
            roi_pool = [None for _ in range(self.num_strides)]
            for i in range(self.num_strides):
            roi_pool[i] = mx.nd.ROIPooling(in_data[i], mx.nd.array(rois_p[i], in_data[i].context), (7, 7), spatial_scale=1.0 / self.feat_strides[i])

            roi_pool = mx.nd.concatenate(roi_pool, axis=0)
        roi_pool = mx.nd.take(roi_pool, mx.nd.array(rois_idx, roi_pool.context))
        self.assign(out_data[0],req[0],roi_pool)
    
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i],req[i],0)

        with autograd.train_section():
            for i in range(self.num_strides):
                if len(self.feat_idx[i] > 0):
                    autograd.compute_gradient([mx.nd.take(out_grad[0],mx.nd.array(self.feat_id[i],out_grad[0].context)) * self.roi_pool[i]])

        for i in range(0,self.num_strides):
            self.assign(in_grad[i], req[i], self.in_grad_hist_list[i])

        gc.collect()

@mx.operator.register('fpn_roi_pooling')
class FPNROIPoolingProp(mx.operator.CustomOpProp):
    def __init__(self, feat_strides='(4,8,16,32)',pooled_height = '7',pooled_width='7',output_dim='256'):
        super(FPNROIPoolingProp,self).__init__(need_top_grad = True)
        self.pooled_height = int(pooled_height)
        self.pooled_width = int(pooled_width)
        self.feat_strides = np.fromstring(feat_strides[1:-1],dtype = int, sep=',')
        self.output_dim = int(output_dim)
        self.num_strides = len(self.feat_strides)
    def list_arguments(self):
        args_list = []
        for i in range(self.num_strides):
            args_list.append('data_p{}'.format(2+i))

        args_list.append('rois')
        return args_list

    def list_outputs(self):
        return ['output']
    def infer_shape(self, in_shape):
        output_feat_shape = [in_shape[-1][0],in_shape[0][1],self.pooled_height,self.pooled_width]
        return in_shape,[output_feat_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return FPNROIPoolingOperator(self.feat_strides, self.pooled_height, self.pooled_width, self.output_dim, self.with_deformable)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return [out_grad[0]]
