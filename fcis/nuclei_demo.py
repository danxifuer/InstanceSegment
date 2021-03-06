import os
import sys
import pprint
from fcis.config.config import config, update_config
from lib.utils.image import resize, transform
import glob

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + '/../experiments/fcis/cfgs/nuclei_end2end.yaml')

sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))
import mxnet as mx
from fcis.core.tester import im_detect, Predictor
from fcis.symbols import *
from lib.bbox.bbox_transform import clip_boxes
from lib.utils.load_model import load_param
from lib.utils.show_masks import show_masks
from lib.utils.tictoc import tic, toc
from lib.nms.nms import py_nms_wrapper
from lib.mask.mask_transform import gpu_mask_voting, cpu_mask_voting
print("use mxnet at", mx.__file__)


def main():
    # get symbol
    ctx_id = [int(i) for i in config.gpus.split(',')]
    # ctx_id = [mx.cpu()]
    print(ctx_id)
    pprint.pprint(config)
    sym_instance = eval(config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=False)

    # set up class names
    num_classes = 2
    classes = ['nuclei']

    # load demo data
    # image_root = '/home/daiab/machine_disk/work/kaggle_nuclei/data/stage1_test_images'
    image_root = '/home/daiab/machine_disk/work/kaggle_nuclei/data/LikeVOC/img'
    image_names = glob.glob(os.path.join(image_root, '*.png'))
    data = []
    for im_name in image_names:
        im = cv2.imread(im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        target_size = config.SCALES[0][0]
        max_size = config.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        data.append({'data': im_tensor, 'im_info': im_info})

    # get predictor
    data_names = ['data', 'im_info']
    label_names = []
    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in range(len(data))]
    max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in range(len(data))]
    provide_label = [None for i in range(len(data))]
    arg_params, aux_params = load_param('/home/daiab/machine_disk/projects/FCIS/'
                                        'output/fcis/nuclei/nuclei_end2end/SDS_train/e2e', 2, process=True)
    data = data
    max_data_shape = max_data_shape
    provide_data = provide_data
    predictor = Predictor(sym, data_names, label_names,
                          context=[mx.gpu(ctx_id[0])],
                          max_data_shapes=max_data_shape,
                          provide_data=provide_data,
                          provide_label=provide_label,
                          arg_params=arg_params,
                          aux_params=aux_params)

    # warm up
    for i in range(2):
        data_batch = mx.io.DataBatch(data=[data[0]], label=[], pad=0, index=0,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[0])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in range(len(data_batch.data))]
        # print('-----------', data_batch)
        _, _, _, _ = im_detect(predictor, data_batch, data_names, scales, config)

    # test
    for idx, im_name in enumerate(image_names):
        data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in range(len(data_batch.data))]

        tic()
        scores, boxes, masks, data_dict = im_detect(predictor, data_batch, data_names, scales, config)
        im_shapes = [data_batch.data[i][0].shape[2:4] for i in range(len(data_batch.data))]

        if not config.TEST.USE_MASK_MERGE:
            all_boxes = [[] for _ in range(num_classes)]
            all_masks = [[] for _ in range(num_classes)]
            nms = py_nms_wrapper(config.TEST.NMS)
            for j in range(1, num_classes):
                indexes = np.where(scores[0][:, j] > 0.7)[0]
                cls_scores = scores[0][indexes, j, np.newaxis]
                cls_masks = masks[0][indexes, 1, :, :]
                try:
                    if config.CLASS_AGNOSTIC:
                        cls_boxes = boxes[0][indexes, :]
                    else:
                        raise Exception()
                except:
                    cls_boxes = boxes[0][indexes, j * 4:(j + 1) * 4]

                cls_dets = np.hstack((cls_boxes, cls_scores))
                keep = nms(cls_dets)
                all_boxes[j] = cls_dets[keep, :]
                all_masks[j] = cls_masks[keep, :]
            dets = [all_boxes[j] for j in range(1, num_classes)]
            masks = [all_masks[j] for j in range(1, num_classes)]
        else:
            masks = masks[0][:, 1:, :, :]
            im_height = np.round(im_shapes[0][0] / scales[0]).astype('int')
            im_width = np.round(im_shapes[0][1] / scales[0]).astype('int')
            print(im_height, im_width)
            boxes = clip_boxes(boxes[0], (im_height, im_width))
            result_masks, result_dets = gpu_mask_voting(masks, boxes, scores[0], num_classes,
                                                        100, im_width, im_height,
                                                        config.TEST.NMS, config.TEST.MASK_MERGE_THRESH,
                                                        config.BINARY_THRESH, ctx_id[0])
            dets = [result_dets[j] for j in range(1, num_classes)]
            masks = [result_masks[j][:, 0, :, :] for j in range(1, num_classes)]
            # print(dets)
            # print(masks)
        print('testing {} {:.4f}s'.format(im_name, toc()))
        # visualize
        for i in range(len(dets)):
            keep = np.where(dets[i][:, -1] > 0.7)
            dets[i] = dets[i][keep]
            masks[i] = masks[i][keep]
        im = cv2.imread(im_name)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        show_masks(im, dets, masks, classes, config)

    print('done')


if __name__ == '__main__':
    main()
