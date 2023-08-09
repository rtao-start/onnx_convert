import numpy as np

class DecodeBox():
    #-----------------------------------------------------------#
    #   13x13���������Ӧ��anchor��[116,90],[156,198],[373,326]
    #   26x26���������Ӧ��anchor��[30,61],[62,45],[59,119]
    #   52x52���������Ӧ��anchor��[10,13],[16,30],[33,23]
    #-----------------------------------------------------------#
    def __init__(self, anchors, num_classes, img_size):
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

    def detect(self, input):
                #-----------------------------------------------#
        #   �����inputһ�������������ǵ�shape�ֱ���
        #   batch_size, 255, 13, 13
        #   batch_size, 255, 26, 26
        #   batch_size, 255, 52, 52
        #-----------------------------------------------#
        batch_size = input.shape[0]
        input_height = input.shape[2]
        input_width = input.shape[3]

        #-----------------------------------------------#
        #   ����Ϊ416x416ʱ
        #   stride_h = stride_w = 32��16��8
        #-----------------------------------------------#
        stride_h = self.img_size[1] / input_height
        stride_w = self.img_size[0] / input_width
        #-------------------------------------------------#
        #   ��ʱ��õ�scaled_anchors��С��������������
        #-------------------------------------------------#
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors]
        #print('scaled_anchors:', scaled_anchors)

        #-----------------------------------------------#
        #   �����inputһ�������������ǵ�shape�ֱ���
        #   batch_size, 3, 13, 13, 85
        #   batch_size, 3, 26, 26, 85
        #   batch_size, 3, 52, 52, 85
        #-----------------------------------------------#

        prediction = input.reshape(batch_size, self.num_anchors,
                                self.bbox_attrs, input_height, input_width).transpose(0, 1, 3, 4, 2)

        # ����������λ�õĵ�������
        x = 1 / (1 + np.exp(-prediction[..., 0]))
        y = 1 / (1 + np.exp(-prediction[..., 1]))

        # �����Ŀ�ߵ�������
        w = prediction[..., 2]
        h = prediction[..., 3]

        # ������Ŷȣ��Ƿ�������
        conf = 1 / (1 + np.exp(-prediction[..., 4]))

        # �������Ŷ� 
        pred_cls = 1 / (1 + np.exp(-prediction[..., 5:]))

        #----------------------------------------------------------#
        #   ����������������ģ��������Ͻ� 
        #   batch_size,3,13,13
        #----------------------------------------------------------#
        # tensor.repeat(repeats, axis=None) repeats:ÿ��Ԫ���ظ��Ĵ����� axis:��Ҫ�ظ���ά��
        print('input_width:{}, input_height:{}, img_size:{}'.format(input_width, input_height, self.img_size))
        FloatArray = np.float32

        grid_x = np.linspace(0, input_width - 1, input_width)
        grid_x = np.tile(grid_x, (input_height, 1))
        grid_x = np.tile(grid_x, (batch_size * self.num_anchors, 1, 1)).reshape(x.shape).astype(FloatArray)

        grid_y = np.linspace(0, input_height - 1, input_height)
        grid_y = np.tile(grid_y, (input_width, 1)).T
        grid_y = np.tile(grid_y, (batch_size * self.num_anchors, 1, 1)).reshape(y.shape).astype(FloatArray)

        #----------------------------------------------------------#
        #   ���������ʽ���������Ŀ��
        #   batch_size,3,13,13
        #----------------------------------------------------------#
        anchor_w = np.array(scaled_anchors)[:, 0].repeat(batch_size).repeat(input_height * input_width).reshape(w.shape)
        anchor_h = np.array(scaled_anchors)[:, 1].repeat(batch_size).repeat(input_height * input_width).reshape(h.shape)

        #----------------------------------------------------------#
        #   ����Ԥ�������������е���
        #   ���ȵ������������ģ�����������������½�ƫ��
        #   �ٵ��������Ŀ�ߡ�
        #----------------------------------------------------------#
        pred_boxes = np.empty_like(prediction[..., :4])
        pred_boxes[..., 0] = x + grid_x
        pred_boxes[..., 1] = y + grid_y
        pred_boxes[..., 2] = np.exp(w) * anchor_w
        pred_boxes[..., 3] = np.exp(h) * anchor_h

        _scale = np.array([stride_w, stride_h, stride_w, stride_h], dtype=np.float32)
        output = np.concatenate((pred_boxes.reshape(batch_size, -1, 4) * _scale,
                                 conf.reshape(batch_size, -1, 1),
                                 pred_cls.reshape(batch_size, -1, self.num_classes)), axis=-1)

        return output

def nms(boxes, scores, iou_threshold):
    #�Զ���Ǽ������ƺ���ʵ��
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h

        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int32)

def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    #----------------------------------------------------------#
    #   ��Ԥ�����ĸ�ʽת�������Ͻ����½ǵĸ�ʽ��
    #   prediction  [batch_size, num_anchors, 85]
    #----------------------------------------------------------#
    box_corner = prediction.copy()
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None] * len(prediction)
    for image_i, image_pred in enumerate(prediction):
        #----------------------------------------------------------#
        #   ������Ԥ�ⲿ��ȡmax��
        #   class_conf  [batch_size, num_anchors, 1]    �������Ŷ�
        #   class_pred  [batch_size, num_anchors, 1]    ����
        #----------------------------------------------------------#
        class_scores = image_pred[:, 5:5 + num_classes]
        class_pred = np.argmax(class_scores, axis=1, keepdims=True)
        class_conf = np.max(class_scores, axis=1, keepdims=True)

        #----------------------------------------------------------#
        #   �������ŶȽ��е�һ��ɸѡ
        #----------------------------------------------------------#
        #��������
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

        #----------------------------------------------------------#
        #   �������ŶȽ���Ԥ������ɸѡ
        #----------------------------------------------------------#
        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]
        if not image_pred.size:
            continue

        #-------------------------------------------------------------------------#
        #   detections  [bbox_numbers, 7]
        #   7������Ϊ��x1, y1, x2, y2, obj_conf, class_conf, class_pred
        #-------------------------------------------------------------------------#
        detections = np.concatenate((image_pred[:, :5], class_conf.astype(np.float32),
                                     class_pred.astype(np.float32)), axis=1)

        #------------------------------------------#
        #   ���Ԥ�����а�������������
        #------------------------------------------#
        unique_labels = np.unique(detections[:, -1])

        for c in unique_labels:
            #------------------------------------------#
            #   ���ĳһ��÷�ɸѡ��ȫ����Ԥ����
            #------------------------------------------#
            detections_class = detections[detections[:, -1] == c]

            #------------------------------------------#
            #   ʹ�ùٷ��Դ��ķǼ������ƻ��ٶȸ���һЩ��
            #------------------------------------------#
            keep = nms(detections_class[:, :4], detections_class[:, 4] * detections_class[:, 5], nms_thres)
            max_detections = detections_class[keep]

            output[image_i] = max_detections if output[image_i] is None else np.concatenate(
                (output[image_i], max_detections))

    return output

def yolo_correct_boxes(top, left, bottom, right, input_shape, image_shape, added_gray=True):
    new_shape = image_shape * np.min(input_shape / image_shape)

    offset = (input_shape - new_shape) / (2.0 * input_shape)
    scale = input_shape / new_shape

    box_yx = np.concatenate(((top + bottom) / 2, (left + right) / 2), axis=-1) / input_shape
    box_hw = np.concatenate((bottom - top, right - left), axis=-1) / input_shape

    if added_gray:
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

    box_mins = box_yx - (box_hw / 2.0)
    box_maxes = box_yx + (box_hw / 2.0)
    boxes = np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ], axis=-1)

    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes
