import mxnet as mx
def conv_act_layer(from_layer, name, num_filter, kernel=(1,1), pad=(0,0), \
    stride=(1,1), act_type="relu", use_batchnorm=False):
    bias = mx.symbol.Variable(name="{}_conv_bias".format(name),   
        init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
    conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad, \
        stride=stride, num_filter=num_filter, name="{}_conv".format(name), bias=bias)
    if use_batchnorm:
        conv = mx.symbol.BatchNorm(data=conv, name="{}_bn".format(name))
    relu = mx.symbol.Activation(data=conv, act_type=act_type, \
        name="{}_{}".format(name, act_type))
    return relu
def multi_layer_feature(body, from_layers, num_filters, strides, pads,
        min_filter):

    internals = body.get_internals()
    layers = []

    for k, params in enumerate(zip(from_layers, num_filters, strides,pads)):
        from_layer, num_filter, s, p = params
        if from_layer.strip():
            layer = internals[from_layer.strip() + '_output']
            layers.append(layer)
        else:
            layer = layers[-1]
            num_1x1 = max(min_filter, num_filter//2)
            conv_1x1 = conv_act_layer(layer, 'multi_feat_%d_conv_1x1'%(k),
                                      num_1x1,kernel=(1,1),pad=(0,0),stride = (1,1),act_type = 'relu')
            conv_3x3 = conv_act_layer(layer, 'multi_feat_%d_conv_3x3'%(k),
                                      num_filter,kernel=(3,3),pad=(p,p),stride = (s,s),act_type = 'relu')
            layers.append(conv_3x3)
    return layers
            
def multibox_layer(from_layers, num_classes, sizes=[.2, .95],
                    ratios=[1], normalization=-1, num_channels=[],
                    clip=False, interm_layer=0, steps=[]):
    if not isinstance(ratios[0], list):
        # provided only one ratio list, broadcast to all from_layers
        ratios = [ratios] * len(from_layers)

    if len(sizes) == 2 and not isinstance(sizes[0], list):
        # provided size range, we need to compute the sizes for each layer
         assert sizes[0] > 0 and sizes[0] < 1
         assert sizes[1] > 0 and sizes[1] < 1 and sizes[1] > sizes[0]
         tmp = np.linspace(sizes[0], sizes[1], num=(len(from_layers)-1))
         min_sizes = [start_offset] + tmp.tolist()
         max_sizes = tmp.tolist() + [tmp[-1]+start_offset]
         sizes = zip(min_sizes, max_sizes)

    if not isinstance(normalization, list):
        normalization = [normalization] * len(from_layers)

    loc_pred_layers = []
    cls_pred_layers = []
    anchor_layers = []
    num_classes += 1
    for k,from_layer in enumerate(from_layers):
        from_name = from_layer.name
        if normalization[k] > 0:
            from_layer = mx.symbol.L2Normalization(data=from_layer,
                mode = 'channel',name='{}_norm'.format(from_name))
            scale = mx.symbol.Variable(name='{}_scale'.format(from_name),
                shape = (1, num_channels.pop(0),1,1),
                init = mx.init.Constant(normalization[k]),
                attr = {'__wd_mult__': '0.1'})
            from_layer = mx.sym.broadcast_mul(lhs = scale, rhs = from_layer)
        if interm_layer > 0:
            from_layer = mx.symbol.Convolution(data=from_layer, kernel=(3,3), \
                stride=(1,1), pad=(1,1), num_filter=interm_layer, \
                name="{}_inter_conv".format(from_name))
            from_layer = mx.symbol.Activation(data=from_layer, act_type="relu", \
                name="{}_inter_relu".format(from_name))
        
        size = sizes[k]
        size_str = '(' + ','.join([str(x) for x in size])+')'
        ratio = ratios[k]
        ratio_str = "(" + ",".join([str(x) for x in ratio]) + ")"
        num_anchors = len(size) -1 + len(ratio)
        num_loc_pred = num_anchors * 4
        bias = mx.symbol.Variable(name="{}_loc_pred_conv_bias".format(from_name),
            init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})

        loc_pred = mx.symbol.Convolution(data=from_layer, bias=bias, kernel=(3,3), \
            stride=(1,1), pad=(1,1), num_filter=num_loc_pred, \
            name="{}_loc_pred_conv".format(from_name))

        loc_pred = mx.symbol.transpose(loc_pred, axes=(0,2,3,1))
        loc_pred = mx.symbol.Flatten(data=loc_pred)
        loc_pred_layers.append(loc_pred)

        num_cls_pred = num_anchors * num_classes
        bias = mx.symbol.Variable(name="{}_cls_pred_conv_bias".format(from_name),
            init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
        cls_pred = mx.symbol.Convolution(data=from_layer, bias=bias, kernel=(3,3), \
            stride=(1,1), pad=(1,1), num_filter=num_cls_pred, \
            name="{}_cls_pred_conv".format(from_name))
        cls_pred = mx.symbol.transpose(cls_pred, axes=(0,2,3,1))
        cls_pred = mx.symbol.Flatten(data=cls_pred)
        cls_pred_layers.append(cls_pred)

        if steps:
            step = (steps[k], steps[k])
        else:
            step ='(-1.0, -1.0)'

        anchors = mx.contrib.sym.MultiBoxPrior(from_layer, sizes = size_str, 
            ratios = ratio_str, clip = clip,
            name = "{}_anchors".format(from_name),steps = step)

        anchors = mx.symbol.Flatten(data=anchors)
        anchor_layers.append(anchors)

    loc_preds = mx.symbol.Concat(*loc_pred_layers, num_args=len(loc_pred_layers), \
        dim=1, name="multibox_loc_pred")
    cls_preds = mx.symbol.Concat(*cls_pred_layers, num_args=len(cls_pred_layers), \
        dim=1)
    cls_preds = mx.symbol.Reshape(data=cls_preds, shape=(0, -1, num_classes))
    cls_preds = mx.symbol.transpose(cls_preds, axes=(0, 2, 1), name="multibox_cls_pred")
    anchor_boxes = mx.symbol.Concat(*anchor_layers, \
        num_args=len(anchor_layers), dim=1)
    anchor_boxes = mx.symbol.Reshape(data=anchor_boxes, shape=(0, -1, 4), name="multibox_anchors")
    return [loc_preds, cls_preds, anchor_boxes]