import os
import time

import cv2
import torch
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from nanodet.util.path import mkdir
from nanodet.util.visualization import overlay_bbox_cv

image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
video_ext = ["mp4", "mov", "avi", "mkv"]


class Predictor(object):
    def __init__(self, cfg, model_path, logger, device="cuda:0"):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert

            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        time1 = time.time()
        result_img = self.model.head.show_result(
            meta["raw_img"][0], dets, class_names, score_thres=score_thres, show=True
        )

        class_text = self.overlay_bbox_cv(dets, class_names, score_thres)

        print("viz time: {:.3f}s".format(time.time() - time1))
        return result_img, class_text

    def get_image_list(path):
        image_names = []
        for maindir, subdir, file_name_list in os.walk(path):
            for filename in file_name_list:
                apath = os.path.join(maindir, filename)
                ext = os.path.splitext(apath)[1]
                if ext in image_ext:
                    image_names.append(apath)
        return image_names

    def overlay_bbox_cv(self, dets, class_names, score_thresh):
        all_box = []
        for label in dets:
            for bbox in dets[label]:
                score = bbox[-1]
                if score > score_thresh:
                    x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                    all_box.append([label, x0, y0, x1, y1, score])
        all_box.sort(key=lambda v: v[5])
        for box in all_box:
            label, x0, y0, x1, y1, score = box
            return class_names[label]
