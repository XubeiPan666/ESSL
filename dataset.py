import os
from numpy.random import f, permutation, rand
from PIL import Image
import time
import torch
import random
import pickle
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
from torch.utils.data import DataLoader




class UVAD_VideoAnomalyDataset_C3D(Dataset):
    """Video Anomaly Dataset."""

    def __init__(self,
                 logger,
                 dataset_txt,
                 data_dir,
                 obj_dir,
                 dataset='shanghaitech',
                 detect_dir=None,
                 fliter_ratio=0.9,
                 frame_num=7,
                 static_threshold=0.1):

        assert os.path.exists(data_dir), "{} does not exist.".format(data_dir)
        assert dataset in ['shanghaitech', 'ped2', 'avenue'], 'wrong type of dataset.'

        self.logger = logger
        self.dataset = dataset
        self.data_dir = data_dir
        self.obj_dir = obj_dir
        self.fliter_ratio = fliter_ratio
        self.static_threshold = static_threshold

        with open(file=dataset_txt, mode='r') as f:
            file_list = f.readlines()
        file_list.sort()

        self.videos = 0

        self.frame_num = frame_num
        assert self.frame_num % 2 == 1, 'We prefer odd number of frames'
        self.half_frame_num = self.frame_num // 2

        self.videos_list = []

        if 'train' in dataset_txt:
            self.test_stage = False
        else:
            self.test_stage = True
        # else:
        #     raise ValueError("data dir: {} is error, not train or test.".format(data_dir))

        # if not self.test_stage and self.dataset == 'shanghaitech':
        #     self.sample_step = 5
        # else:
        #     self.sample_step = 1

        if detect_dir != None:
            with open(detect_dir, 'rb') as f:
                self.detect = pickle.load(f)
        else:
            self.detect = None

        self.objects_list = []
        self._load_data(file_list)

    # 获取对应视频视频帧列表
    def get_frame_num(self, video_file, dataset):
        if dataset == 'shanghaitech':
            if len(video_file) == 6:
                frame_path = os.path.join(self.data_dir, 'training', 'frames', video_file)
            elif len(video_file) == 7:
                frame_path = os.path.join(self.data_dir, 'testing', 'frames', video_file)
            else:
                raise ValueError("video {} error!".format(video_file))
        else:
            if 'train' in video_file:
                frame_path = os.path.join(self.data_dir, 'training', 'frames', video_file)
            elif len(video_file) == 2:
                frame_path = os.path.join(self.data_dir, 'testing', 'frames', video_file)
            else:
                raise ValueError("video {} error!".format(video_file))

        return os.listdir(frame_path)

    # 获取视频中视频帧训练的开始编号
    def get_start_ind(self, video_file, dataset):
        if dataset == 'shanghaitech':
            if len(video_file) == 6:
                return self.frame_num - 1
            elif len(video_file) == 7:
                return self.half_frame_num
            else:
                raise ValueError("video {} error!".format(video_file))
        else:
            if 'train' in video_file:
                return self.frame_num - 1
            elif len(video_file) == 2:
                return self.half_frame_num
            else:
                raise ValueError("video {} error!".format(video_file))

    # 采样步骤，上海数据集太大了，训练集每5帧采一下
    def get_sample_step(self, video_file, dataset):
        if dataset == 'shanghaitech':
            if len(video_file) == 6:
                return 5
            elif len(video_file) == 7:
                return 1
            else:
                raise ValueError("video {} error!".format(video_file))
        else:
            if 'train' in video_file or len(video_file) == 2:
                return 1
            else:
                raise ValueError("video {} error!".format(video_file))

    def _load_data(self, file_list):
        t0 = time.time()
        total_frames = 0
        contain = 0
        total_small_ = 0
        for video_file in file_list:
            video_file = video_file.replace('\n', '')
            if video_file not in self.videos_list:
                self.videos_list.append(video_file)

            l = self.get_frame_num(video_file, self.dataset)

            self.videos += 1
            length = len(l)
            total_frames += length
            start_ind = self.get_start_ind(video_file, self.dataset)
            sample_step = self.get_sample_step(video_file, self.dataset)
            for frame in range(start_ind, length - start_ind, sample_step):
                if self.detect is not None:
                    detect_result = self.detect[video_file][frame]
                    # 筛选置信度比较大的检测对象
                    detect_result = detect_result[detect_result[:, 4] > self.fliter_ratio, :]
                    object_num = detect_result.shape[0]
                else:
                    object_num = 1

                flag = detect_result[:, None, :4].repeat(object_num, 1) - detect_result[None, :, :4].repeat(object_num, 0)
                is_contain = np.all(np.concatenate((flag[:, :, :2] > 0, flag[:, :, 2:] < 0), -1), -1)
                is_contain = is_contain.any(-1)
                is_small = (detect_result[:, 2:4] - detect_result[:, 0:2]).max(-1) < 10
                width = detect_result[:, 2] - detect_result[:, 0]
                height = detect_result[:, 3] - detect_result[:, 1]
                # aspect_ratio = np.minimum(width / height, height / width)
                aspect_ratio = height / width
                for i in range(object_num):
                    if not is_contain[i]:
                        if not is_small[i]:
                            self.objects_list.append({"video_name": video_file, "frame": frame, "object": i,
                                                      "loc": detect_result[i, :4], "aspect_ratio": aspect_ratio[i]})
                        else:
                            total_small_ += 1
                    else:
                        contain += 1

        # print("Load {} videos {} frames, {} objects, excluding {} inside objects and {} small objects in {} s." \
        #       .format(self.videos, total_frames, len(self.objects_list), contain, total_small_, time.time() - t0))

        self.logger.info("Load {} videos {} frames, {} objects, excluding {} inside objects and {} small objects in {} s." \
              .format(self.videos, total_frames, len(self.objects_list), contain, total_small_, time.time() - t0))


    def __len__(self):
        return len(self.objects_list)

    def __video_list__(self):
        return self.videos_list

    def __getitem__(self, idx):
        # 是否进行时序变换
        # temproal_flag = random.randint(0, 1) == 1
        temproal_flag = idx % 2 == 0
        record = self.objects_list[idx]
        if self.test_stage:
            perm = np.arange(self.frame_num)
        else:
            if random.random() < 0.0001:
                perm = np.arange(self.frame_num)
            else:
                perm = np.random.permutation(self.frame_num)
        obj = self.get_object(record["video_name"], record["frame"], record["object"])

        if not temproal_flag and not self.test_stage:
            if random.random() < 0.0001:
                spatial_perm = np.arange(9)
            else:
                spatial_perm = np.random.permutation(9)
        else:
            spatial_perm = np.arange(9)
        obj = self.jigsaw(obj, border=2, patch_size=20, permuation=spatial_perm, dropout=False)
        obj = torch.from_numpy(obj)

        clip_id = str(record["frame"]) + '_' + str(record["object"])

        # NOT permute clips containing static contents
        if (obj[:, -1, :, :] - obj[:, 0, :, :]).abs().max() < self.static_threshold:
            perm = np.arange(self.frame_num)

        if temproal_flag:
            obj = obj[:, perm, :, :]
        obj = torch.clamp(obj, 0., 1.)

        ret = {"video": record["video_name"], "frame": record["frame"], "obj": obj, "label": perm,
               "trans_label": spatial_perm, "loc": record["loc"], "aspect_ratio": record["aspect_ratio"],
               "temporal": temproal_flag}
        return ret

    def get_object(self, video_name, frame, obj_id):
        if self.dataset == 'shanghaitech':
            if len(video_name) == 6:
                video_dir = os.path.join(self.obj_dir, self.dataset, 'training', video_name)
            else:
                video_dir = os.path.join(self.obj_dir, self.dataset, 'testing', video_name)
        else:
            if 'train' in video_name:
                video_dir = os.path.join(self.obj_dir, self.dataset, 'training', video_name)
            else:
                video_dir = os.path.join(self.obj_dir, self.dataset, 'testing', video_name)

        obj = np.load(os.path.join(video_dir, str(frame) + '_' + str(obj_id) + '.npy'))  # (3, 7, 64, 64)

        if not self.test_stage:
            if random.random() < 0.5:
                obj = obj[:, :, :, ::-1]
        return obj

    def split_image(self, clip, border=2, patch_size=20):
        """
        image: (C, T, H, W)
        """
        patch_list = []

        for i in range(3):
            for j in range(3):
                y_offset = border + patch_size * i
                x_offset = border + patch_size * j
                patch_list.append(clip[:, :, y_offset: y_offset + patch_size, x_offset: x_offset + patch_size])

        return patch_list

    def concat(self, patch_list, border=2, patch_size=20, permuation=np.arange(9), num=3, dropout=False):
        """
        batches: [(C, T, h1, w1)]
        """
        clip = np.zeros((3, self.frame_num, 64, 64), dtype=np.float32)
        drop_ind = random.randint(0, len(permuation) - 1)
        for p_ind, i in enumerate(permuation):
            if drop_ind == p_ind and dropout:
                continue
            y = i // num
            x = i % num
            y_offset = border + patch_size * y
            x_offset = border + patch_size * x

            clip[:, :, y_offset: y_offset + patch_size, x_offset: x_offset + patch_size] = patch_list[p_ind]
        return clip

    def jigsaw(self, clip, border=2, patch_size=20, permuation=None, dropout=False):
        patch_list = self.split_image(clip, border, patch_size)
        clip = self.concat(patch_list, border=border, patch_size=patch_size, permuation=permuation, num=3,
                           dropout=dropout)
        return clip


if __name__ == '__main__':
    dataset_txt = './UVAD_dataset/UVAD_shanghaitech/train_split.txt'
    data_dir = f'raw dataset root'
    obj_dir = f'your patches save_path'
    detect_dir = f'./UVAD_detect/shanghaitech/_detect.pkl'

    vad_dataset = UVAD_VideoAnomalyDataset_C3D(dataset_txt=dataset_txt,
                                               data_dir=data_dir,
                                               obj_dir=obj_dir,
                                               dataset='shanghaitech',
                                               detect_dir=detect_dir,
                                               fliter_ratio=0.8,
                                               frame_num=9,
                                               static_threshold=0.2)

    vad_dataloader = DataLoader(vad_dataset, batch_size=64, shuffle=True, num_workers=8
                                , pin_memory=True)

    for epoch in range(1):
        for it, data in enumerate(vad_dataloader):
            video, obj, temp_labels, spat_labels, t_flag = data['video'], data['obj'], data['label'], data[
                "trans_label"], data["temporal"]
        print('one epoth')
