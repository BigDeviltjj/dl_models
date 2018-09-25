import numpy as np

def bbox_transform(ex_rois,gt_rois):
    ex_ctrx = (ex_rois[:,0] + ex_rois[:,2])/2
    ex_ctry = (ex_rois[:,1] + ex_rois[:,3])/2
    gt_ctrx = (gt_rois[:,0] + gt_rois[:,2])/2
    gt_ctry = (gt_rois[:,1] + gt_rois[:,3])/2
    ret = np.zeros_like(ex_rois)
    ret[:,0] = (gt_ctrx - ex_ctrx) / (ex_rois[:,2] - ex_rois[:,0] + 1)
    ret[:,1] = (gt_ctry - ex_ctry) / (ex_rois[:,3] - ex_rois[:,1] + 1)
    ret[:,2] = np.log((gt_rois[:,2] - gt_rois[:,0] + 1)/ (ex_rois[:,2] - ex_rois[:,0] + 1))
    ret[:,3] = np.log((gt_rois[:,3] - gt_rois[:,1] + 1)/ (ex_rois[:,3] - ex_rois[:,1] + 1))
    return ret
def bbox_pred(anchor, bbox):
    bbox = bbox.astype(np.float, copy = False)

    x = bbox[:,0] * (anchor[:,2] - anchor[:,0] + 1) + (anchor[:,2] + anchor[:,0]) / 2
    y = bbox[:,1] * (anchor[:,3] - anchor[:,1] + 1) + (anchor[:,3] + anchor[:,1]) / 2
    w = np.exp(bbox[:,2]) * (anchor[:,2] - anchor[:,0] + 1)
    h = np.exp(bbox[:,3]) * (anchor[:,3] - anchor[:,1] + 1)

    pred_bboxes = np.zeros_like(bbox)

    pred_bboxes[:,0] = x - 0.5 * (w-1)
    pred_bboxes[:,1] = y - 0.5 * (h-1)
    pred_bboxes[:,2] = x + 0.5 * (w-1)
    pred_bboxes[:,3] = y + 0.5 * (h-1)
    return pred_bboxes

def clip_boxes(boxes, im_shape):
    boxes[:,0] = np.maximum(np.minimum(boxes[:,0],im_shape[1] - 1), 0)
    boxes[:,1] = np.maximum(np.minimum(boxes[:,1],im_shape[0] - 1), 0)
    boxes[:,2] = np.maximum(np.minimum(boxes[:,2],im_shape[1] - 1), 0)
    boxes[:,3] = np.maximum(np.minimum(boxes[:,3],im_shape[0] - 1), 0)
    return boxes

def bbox_overlaps(bboxes,query_boxes):
    ret = np.zeros((bboxes.shape[0],query_boxes.shape[0]))
    for i,b in enumerate(bboxes):
        w = np.maximum(np.minimum(b[2],query_boxes[:,2]) - np.maximum(b[0],query_boxes[:,0]) + 1,0.)
        h = np.maximum(np.minimum(b[3],query_boxes[:,3]) - np.maximum(b[1],query_boxes[:,1]) + 1,0.)
        ret[i] = (w*h) / ((b[2]-b[0]+1)*(b[3]-b[1]+1)+(query_boxes[:,2]-query_boxes[:,0]+1)*(query_boxes[:,3]-query_boxes[:,1]+1)-(w*h))
    return ret