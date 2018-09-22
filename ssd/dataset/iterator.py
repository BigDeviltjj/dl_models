import mxnet as mx
import numpy as np

class DetRecordIter(mx.io.DataIter):
    def __init__(self, path_imgrec, batch_size, data_shape, path_imglist="",
                 label_width=-1, label_pad_width=-1, label_pad_value=-1,
                 resize_mode='force',  mean_pixels=[123.68, 116.779, 103.939],
                 **kwargs):
        super(DetRecordIter, self).__init__()
        self.rec = mx.io.ImageDetRecordIter(
            path_imgrec = path_imgrec,
            path_imglist = path_imglist,
            label_width = label_width,
            label_pad_width = label_pad_width,
            label_pad_value = label_pad_value,
            batch_size = batch_size,
            data_shape = data_shape,
            mean_r = mean_pixels[0],
            mean_g = mean_pixels[1],
            mean_b = mean_pixels[2],
            resize_mode = resize_mode,
            **kwargs
        )
        self.provide_label = None
        self._get_batch()
        if not self.provide_label:
            raise RuntimeError('invalid imagedetrecordIter:'+path_imgrec)
        self.reset()
    
    @property
    def provide_data(self):
        return self.rec.provide_data
    
    def reset(self):
        self.rec.reset()
    
    def iter_next(self):
        return self._get_batch()


    def next(self):
        if self.iter_next():
            return self._batch
        else:
            raise StopIteration

    def _get_batch(self):
        self._batch = self.rec.next()
        if not self._batch:
            return False

        if self.provide_label is None:
            first_label = self._batch.label[0][0].asnumpy()
            self.batch_size = self._batch.label[0].shape[0]
            self.label_header_width = int(first_label[4])
            self.label_object_width = int(first_label[5])
            assert self.label_object_width >= 5,'obj width must >=5'
            self.label_start = 4 + self.label_header_width
            self.max_objects = (first_label.size - self.label_start) // self.label_object_width
            self.label_shape = (self.batch_size,self.max_objects,self.label_object_width)
            self.label_end = self.label_start + self.max_objects * self.label_object_width
            self.provide_label = [('label',self.label_shape)]

        label = self._batch.label[0].asnumpy()
        label = label[:,self.label_start:self.label_end].reshape(
            (self.batch_size, self.max_objects, self.label_object_width))
        self._batch.label = [mx.nd.array(label)]
        return True