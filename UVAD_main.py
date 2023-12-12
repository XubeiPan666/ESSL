import os
import argparse
import torch
import time
import pickle
import numpy as np

from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import UVAD_VideoAnomalyDataset_C3D
from models import model

from tqdm import tqdm
from aggregate import remake_video_output, evaluate_auc, remake_video_3d_output
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import random
import logging

torch.backends.cudnn.benchmark = False


def set_seed(seed=321):
    """set random seed"""
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_logger(path):
    logger_name = 'Main-logger'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    file_hander = logging.FileHandler(path, encoding='utf-8')
    fmt = '[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s'
    console_handler.setFormatter(logging.Formatter(fmt))
    file_hander.setFormatter(logging.Formatter(fmt))
    logger.addHandler((console_handler))
    logger.addHandler(file_hander)
    return logger


# Config
def get_configs():
    parser = argparse.ArgumentParser(description="VAD-Jigsaw config")
    parser.add_argument("--sk", type=int, default=0.9)
    parser.add_argument("--tk", type=int, default=0.7)
    parser.add_argument("--val_step", type=int, default=100)
    parser.add_argument("--print_interval", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gpu_id", type=str, default=0)
    parser.add_argument("--log_date", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--static_threshold", type=float, default=0.2)
    parser.add_argument("--sample_num", type=int, default=9)
    parser.add_argument("--filter_ratio", type=float, default=0.8)
    parser.add_argument("--checkpoint", type=str, default=None)
    # 注意修改
    parser.add_argument("--dataset", type=str, default="shanghaitech", choices=['shanghaitech', 'ped2', 'avenue'])
    args = parser.parse_args()

    args.device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
    if args.dataset in ['shanghaitech', 'avenue']:
        args.filter_ratio = 0.8
    elif args.dataset == 'ped2':
        args.filter_ratio = 0.5
    return args


def remake_loss(loss, sample_num, device):
    l_loss = len(loss)
    if l_loss % sample_num != 0:
        print(f'len(loss): {l_loss} error! in sample_num:{sample_num}')
        exit(0)

    object_num = l_loss // sample_num
    object_loss = [0] * object_num
    object_loss = torch.tensor(object_loss, dtype=torch.float).cuda(device)

    for i in range(object_num):
        for j in range(sample_num):
            object_loss[i] += loss[i * sample_num + j]
    return object_loss / sample_num


def STL(loss, device, sample_num, k):
    object_loss = remake_loss(loss, sample_num, device)
    k = np.ceil(len(object_loss) * k).astype('int32')
    kmin_loss, k_min_index = torch.topk(object_loss, k=k, largest=False)
    return torch.mean(kmin_loss)


def train(args):
    set_seed()
    if not args.log_date:
        running_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    else:
        running_date = args.log_date
    logger = get_logger(f'./log/{running_date}.log')
    # print("The running_data : {}".format(running_date))
    logger.info("The running_data : {}".format(running_date))
    for k, v in vars(args).items():
        # print("-------------{} : {}".format(k, v))
        logger.info("-------------{} : {}".format(k, v))

    # Load Data
    dataset_txt = f'./SVAD_dataset/SVAD_{args.dataset}/train_split.txt'
    data_dir = f'E:\datasets/{args.dataset}'
    obj_dir = f'F:\Jigsaw'
    detect_dir = f'./UVAD_detect/{args.dataset}_detect.pkl'

    vad_dataset = UVAD_VideoAnomalyDataset_C3D(logger,
                                               dataset_txt=dataset_txt,
                                               data_dir=data_dir,
                                               obj_dir=obj_dir,
                                               dataset=args.dataset,
                                               detect_dir=detect_dir,
                                               fliter_ratio=args.filter_ratio,
                                               frame_num=args.sample_num,
                                               static_threshold=args.static_threshold)

    vad_dataloader = DataLoader(vad_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
                                , pin_memory=True)

    net = model.RANet(time_length=args.sample_num, num_classes=[args.sample_num * args.sample_num, 81],
                      use_1x1conv=False)

    if args.checkpoint is not None:
        state = torch.load(args.checkpoint)
        print('load ' + args.checkpoint)
        net.load_state_dict(state, strict=True)
        net.cuda()
        smoothed_auc, smoothed_auc_avg, _, _, _ = val(args, logger, net)
        exit(0)

    net.cuda(args.device)
    net = net.train()

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(params=net.parameters(), lr=1e-4, weight_decay=0.0005)
    scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=5e-5)

    t0 = time.time()
    global_step = 0
    t = 0

    max_auc = -1
    s_max_auc = -1
    t_max_auc = -1
    timestamp_in_max = None
    s_timestamp_in_max = None
    t_timestamp_in_max = None

    for epoch in range(args.epochs):
        for it, data in enumerate(vad_dataloader):
            video, obj, temp_labels, spat_labels, t_flag = data['video'], data['obj'], data['label'], data[
                "trans_label"], data["temporal"]
            # n_temp = t_flag.sum().item()

            obj = obj.cuda(args.device, non_blocking=True)
            temp_labels = temp_labels[t_flag].long().view(-1).cuda(args.device)
            spat_labels = spat_labels[~t_flag].long().view(-1).cuda(args.device)

            temp_logits, spat_logits = net(obj)
            temp_logits = temp_logits[t_flag].view(-1, args.sample_num)
            spat_logits = spat_logits[~t_flag].view(-1, 9)

            temp_loss = criterion(temp_logits, temp_labels)
            spat_loss = criterion(spat_logits, spat_labels)
            # if epoch >= 5:
            #     temp_loss = STL(temp_loss, args.device, args.sample_num, args.tk)
            #     spat_loss = STL(spat_loss, args.device, 9, args.sk)

            temp_loss = torch.mean(temp_loss)
            spat_loss = torch.mean(spat_loss)





            loss = temp_loss + spat_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (global_step + 1) % args.print_interval == 0:
                logger.info("[{}:{}/{}]\tloss: {:.6f} t_loss: {:.6f} s_loss: {:.6f} \ttime: {:.6f}". \
                            format(epoch, it + 1, len(vad_dataloader), loss.item(), temp_loss.item(), spat_loss.item(),
                                   time.time() - t0))

                t0 = time.time()

            global_step += 1
            t += 1
            if global_step % args.val_step == 0 and epoch >= 0:
                smoothed_auc, smoothed_auc_avg, temp_timestamp, s_auc, t_auc = val(args, logger, net)

                if smoothed_auc > max_auc:
                    max_auc = smoothed_auc
                    timestamp_in_max = epoch
                    save_path = 'your pth save_path'
                    torch.save(net.state_dict(), save_path)

                if s_auc > s_max_auc:
                    s_max_auc = s_auc
                    s_timestamp_in_max = epoch
                    save_path = 'your pth save_path'
                    torch.save(net.state_dict(), save_path)

                if t_auc > t_max_auc:
                    t_max_auc = t_auc
                    t_timestamp_in_max = epoch
                    save_path = 'your pth save_path'
                    torch.save(net.state_dict(), save_path)

                logger.info('cur s-max: ' + str(s_max_auc) + ' in epoch ' + str(s_timestamp_in_max))
                logger.info('cur t-max: ' + str(t_max_auc) + ' in epoch ' + str(t_timestamp_in_max))
                logger.info('cur max: ' + str(max_auc) + ' in epoch ' + str(timestamp_in_max))
                logger.info('_________________________________________________________')

                net = net.train()
        scheduler.step()


def val(args, logger, net=None):
    if not args.log_date:
        running_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    else:
        running_date = args.log_date
    # print("The running_date : {}".format(running_date))
    logger.info("The running_date : {}".format(running_date))

    # Load Data
    dataset_txt = f'./SVAD_dataset/SVAD_{args.dataset}/test_split.txt'
    data_dir = f'E:\datasets/{args.dataset}'
    obj_dir = f'F:\Jigsaw'
    detect_dir = f'./UVAD_detect/{args.dataset}_detect.pkl'

    testing_dataset = UVAD_VideoAnomalyDataset_C3D(logger,
                                                   dataset_txt=dataset_txt,
                                                   data_dir=data_dir,
                                                   obj_dir=obj_dir,
                                                   dataset=args.dataset,
                                                   detect_dir=detect_dir,
                                                   fliter_ratio=args.filter_ratio,
                                                   frame_num=args.sample_num,
                                                   static_threshold=args.static_threshold)

    testing_data_loader = DataLoader(testing_dataset, batch_size=256, shuffle=False, num_workers=0, drop_last=False)

    net.eval()

    video_output = {}
    for data in tqdm(testing_data_loader):
        videos = data["video"]
        frames = data["frame"].tolist()
        obj = data["obj"].cuda(args.device)

        with torch.no_grad():
            temp_logits, spat_logits = net(obj)
            temp_logits = temp_logits.view(-1, args.sample_num, args.sample_num)
            spat_logits = spat_logits.view(-1, 9, 9)

        spat_probs = F.softmax(spat_logits, -1)
        diag = torch.diagonal(spat_probs, offset=0, dim1=-2, dim2=-1)
        scores = diag.sum(-1).cpu().numpy()
        # scores = diag.min(-1)[0].cpu().numpy()

        temp_probs = F.softmax(temp_logits, -1)
        diag2 = torch.diagonal(temp_probs, offset=0, dim1=-2, dim2=-1)
        scores2 = diag2.sum(-1).cpu().numpy()
        # scores2 = diag2.min(-1)[0].cpu().numpy()

        for video_, frame_, s_score_, t_score_ in zip(videos, frames, scores, scores2):
            if video_ not in video_output:
                video_output[video_] = {}
            if frame_ not in video_output[video_]:
                video_output[video_][frame_] = []
            video_output[video_][frame_].append([s_score_, t_score_])

    micro_auc, macro_auc, s_auc, t_auc = save_and_evaluate(logger, video_output=video_output, dataset_txt=dataset_txt,
                                                           dataset=args.dataset)
    return micro_auc, macro_auc, running_date, s_auc, t_auc


def save_and_evaluate(logger, video_output, dataset_txt, dataset='shanghaitech'):
    # pickle_path = './log/video_output_ori_{}.pkl'.format(running_date)
    # with open(pickle_path, 'wb') as write:
    #     pickle.dump(video_output, write, pickle.HIGHEST_PROTOCOL)

    if dataset == 'shanghaitech':
        video_output_spatial, video_output_temporal, video_output_complete = remake_video_output(
            video_output=video_output,
            dataset=dataset)
    else:
        video_output_spatial, video_output_temporal, video_output_complete = remake_video_3d_output(
            dataset_txt=dataset_txt,
            video_output=video_output,
            dataset=dataset)
    s_res, s_res_list = evaluate_auc(logger, video_output_spatial, dataset_txt, dataset=dataset)
    t_res, t_res_list = evaluate_auc(logger, video_output_temporal, dataset_txt, dataset=dataset)
    smoothed_res, smoothed_auc_list = evaluate_auc(logger, video_output_complete, dataset_txt, dataset=dataset)
    return smoothed_res.auc, np.mean(smoothed_auc_list), s_res.auc, t_res.auc


if __name__ == '__main__':
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    args = get_configs()
    train(args)
