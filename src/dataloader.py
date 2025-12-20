import torch
import os
import torchaudio

torchaudio.set_audio_backend("librosa")
import ffmpeg
import numpy as np
from PIL import Image
import torchaudio
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as T
import csv
import random
import io
from PIL import Image

from collections import Counter
import random
import torchvision.transforms.functional as F


# Video: frames, audio: fbank
class VideoAudioDataset_Pretraining(Dataset):
    def __init__(self, csv_file, conf, num_frames=16):
        self.num_frames = num_frames

        self.data = []
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过 CSV 文件的第一行(表头: video_name,target)
            for row in reader:
                self.data.append(row)
        print(f'According to the csv file, dataset has {len(self.data)} samples')

        self.num_samples = len(self.data)
        self.conf = conf
        self.melbins = self.conf.get('num_mel_bins')  # dictionary.get(key, default=None)

        self.norm_mean = self.conf.get('mean')
        self.norm_std = self.conf.get('std')

        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.conf.get('skip_norm') if self.conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print(
                'use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))

        self.target_length = self.conf.get('target_length')

        # train or eval
        self.mode = self.conf.get('mode')
        print('now in {:s} mode.'.format(self.mode))

        # by default, all models use 224*224, other resolutions are not tested
        self.im_res = self.conf.get('im_res', 224)
        print('now using {:d} * {:d} image input'.format(self.im_res, self.im_res))
        self.preprocess = T.Compose([
            T.Resize(size=(self.im_res, self.im_res)),
            T.ToTensor(),
        ])


    def extract_audio_from_video(self, video_file, output_audio_file, log_file="failed_audio_videos.txt"):
        """从.mp4文件中提取音频并保存为.wav格式"""
        try:
            # 使用 ffmpeg 从视频中提取音频并保存为 wav 格式
            ffmpeg.input(video_file).output(output_audio_file, ac=1, ar='16k').run(
                quiet=True)  # ac=1 表示单声道，ar='16k' 设置采样率
            print(f"Audio successfully extracted to {output_audio_file}.")
        except ffmpeg.Error as e:
            # 记录读取失败的视频
            with open(log_file, "a") as f:
                f.write(video_file + "\n")
            print(f"Error extracting audio: {e}")

    def _wav2fbank(self, filename):
        # 如果文件是 .mp4 格式，先提取音频
        if filename.endswith('.mp4'):
            temp_audio_file = filename.replace('.mp4', '.wav')
            if not os.path.exists(temp_audio_file):
                self.extract_audio_from_video(filename, temp_audio_file)
            else:
                # print(f"Audio file {temp_audio_file} already exists.")
                pass
            filename = temp_audio_file

        # 加载音频文件
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()

        try:
            # 尝试提取梅尔频率倒谱系数（MFCC）
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        except Exception as e:
            # 捕获具体异常并输出错误信息
            print(f"Error in loading audio or computing fbank: {e}")
            # 返回默认的 fbank 值以避免崩溃
            fbank = torch.zeros([512, 128]) + 0.01
            print('There was an error loading the fbank. Returning default tensor.')

        # 调整 fbank的大小到1024*128
        # (time_frames, 128)-->(1, time_frames, 128)-->(1, 128, time_frames)
        # -->(1, 128, 1024)-->(1, 1024, 128)-->(1024, 128)
        fbank = torch.nn.functional.interpolate(
            fbank.unsqueeze(0).transpose(1, 2), size=(self.target_length,),
            mode='linear', align_corners=False).transpose(1, 2).squeeze(0)

        return fbank

    def _get_frames(self, folder_path):
        """
        从指定的文件夹中按顺序读取 16 张 face_00.png ~ face_15.png 图像，并进行预处理。
        """
        frames = []
        for i in range(self.num_frames):  # self.num_frames = 16
            frame_path = os.path.join(folder_path, f"face_{i:02d}.png")
            try:
                frame = Image.open(frame_path).convert('RGB')  # 转为RGB确保通道一致
                frame = self.preprocess(frame)  # 使用已有的self.preprocess
            except Exception as e:
                print(f"Error loading frame {frame_path}: {e}")
                frame = torch.zeros(3, self.im_res, self.im_res)
            frames.append(frame)
        return torch.stack(frames)  # 输出形状: [16, 3, H, W]

    def __getitem__(self, index):
        video_name, face_crop_folder, label = self.data[index]  # eg. xx.mp4,0
 
        try:
            fbank = self._wav2fbank(video_name)
        except:
            fbank = torch.zeros([self.target_length, 128]) + 0.01
            print('there is an error in loading audio3')

        frames = self._get_frames(face_crop_folder)
        frames = frames.permute(1, 0, 2, 3)

        # normalize the input for both training and test
        if self.skip_norm == False:
            fbank = (fbank - self.norm_mean) / (self.norm_std)
        # skip normalization the input ONLY when you are trying to get the normalization stats.
        else:
            pass

        # not used in pre-training stage
        label = torch.tensor(int(label), dtype=torch.long)  

        return fbank, frames, label

    def __len__(self):
        return self.num_samples


# Video: frames, audio: fbank, audio_label, video_label, overall_label
class VideoAudioDataset_Finetuning(Dataset):
    def __init__(self, csv_file, conf, num_frames=16):
        self.num_frames = num_frames

        self.data = []
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过 CSV 文件的第一行(表头: video_name,target)
            for row in reader:
                self.data.append(row)
        print(f'According to the csv file, dataset has {len(self.data)} samples')

        self.labels = [int(row[-1]) for row in self.data]
        self.video_labels = [int(row[-2]) for row in self.data]
        self.audio_labels = [int(row[-3]) for row in self.data]

        # 0:ra+rv 1:ra+fv 2:fa+rv 3:fa+fv
        self.comb_labels = [a*2 + v for a, v in zip(self.audio_labels, self.video_labels)]

        self.num_samples = len(self.data)
        self.conf = conf
        self.melbins = self.conf.get('num_mel_bins')  # dictionary.get(key, default=None)

        self.visual_augment = self.conf.get('visual_augment')
        if self.visual_augment:
            print("using visual augmentation.")
        else:
            print("Not using visual augmentation.")

        self.audio_augment = self.conf.get('audio_augment')
        if self.audio_augment:
            print("using audio augmentation.")
        else:
            print("Not using audio augmentation.")

        self.freqm = self.conf.get('freqm', 0)
        self.timem = self.conf.get('timem', 0)

        # 将数据调整为均值为 0, 标准差为 1 的分布
        self.norm_mean = self.conf.get('mean')
        self.norm_std = self.conf.get('std')

        self.skip_norm = self.conf.get('skip_norm') if self.conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print(
                'use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))

        self.target_length = self.conf.get('target_length')

        # train or eval
        self.mode = self.conf.get('mode')
        print('now in {:s} mode.'.format(self.mode))

        # by default, all models use 224*224, other resolutions are not tested
        self.im_res = self.conf.get('im_res', 224)
        print('now using {:d} * {:d} image input'.format(self.im_res, self.im_res))



        self.visual_preprocess = Visual_Preprocess(
            im_res=self.im_res
        )

        self.real_indices = [i for i, item in enumerate(self.data) if item[-1] == '0']
        print("There are {:d} real samples in the dataset.".format(len(self.real_indices)))

    def extract_audio_from_video(self, video_file, output_audio_file):
        """从.mp4文件中提取音频并保存为.wav格式"""
        try:
            # 使用 ffmpeg 从视频中提取音频并保存为 wav 格式
            ffmpeg.input(video_file).output(output_audio_file, ac=1, ar='16k').run(
                quiet=False)  # ac=1 表示单声道，ar='16k' 设置采样率
            print(f"Audio successfully extracted to {output_audio_file}.")
        except ffmpeg.Error as e:
            print(f"Error extracting audio: {e}")

    def _wav2fbank(self, filename):
        # 如果文件是 .mp4 格式，先提取音频
        if filename.endswith('.mp4'):
            temp_audio_file = filename.replace('.mp4', '.wav')
            if not os.path.exists(temp_audio_file):
                self.extract_audio_from_video(filename, temp_audio_file)
            else:
                # print(f"Audio file {temp_audio_file} already exists.")
                pass
            filename = temp_audio_file

        # 加载音频文件
        try:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        except Exception as e:
            print(f"Error loading audio file {filename}: {e}")
            #waveform = torch.zeros(1, 16000)

        try:
            # 尝试提取梅尔频率倒谱系数（MFCC）
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        except Exception as e:
            # 捕获具体异常并输出错误信息
            print(f"Error in loading audio or computing fbank: {e}")
            # 返回默认的 fbank 值以避免崩溃
            fbank = torch.zeros([512, 128]) + 0.01
            print('There was an error loading the fbank. Returning default tensor.')

        # 调整 fbank的大小到1024*128
        # (time_frames, 128)-->(1, time_frames, 128)-->(1, 128, time_frames)
        # -->(1, 128, 1024)-->(1, 1024, 128)-->(1024, 128)
        fbank = torch.nn.functional.interpolate(
            fbank.unsqueeze(0).transpose(1, 2), size=(self.target_length,),
            mode='linear', align_corners=False).transpose(1, 2).squeeze(0)

        return fbank

    def _get_frames(self, folder_path):
        """
        从指定的文件夹中按顺序读取 16 张 face_00.png ~ face_15.png 图像，并进行预处理。
        """
        frames = []
        for i in range(self.num_frames):  # self.num_frames = 16
            frame_path = os.path.join(folder_path, f"face_{i:02d}.png")
            try:
                frame = Image.open(frame_path).convert('RGB')  # 转为RGB确保通道一致
            except Exception as e:
                print(f"Error loading frame {frame_path}: {e}")
                frame = torch.zeros(3, self.im_res, self.im_res)
            frames.append(frame)

        if self.mode == "eval":            
            frames = self.visual_preprocess(frames=frames, visual_augment=False)
        else:
            frames = self.visual_preprocess(frames=frames, visual_augment=self.visual_augment)
        
        return frames  # 输出形状: [16, 3, H, W]


    def __getitem__(self, index):
        video_name, face_crop_folder, audio_label, video_label, label = self.data[index]  # eg. xx.mp4,0

        # 在验证模型下不使用数据增强
        if self.mode == 'eval':
            try:
                fbank = self._wav2fbank(video_name)
            except Exception as e:
                fbank = torch.zeros([self.target_length, 128]) + 0.01
                print("!!!There is an error in loading audio:")
                print(f"Error in loading audio or computing fbank: {e}")

            frames = self._get_frames_eval(face_crop_folder)

        # train mode
        else:
            try:
                fbank = self._wav2fbank(video_name)
            except:
                fbank = torch.zeros([self.target_length, 128]) + 0.01
                print('there is an error in loading audio!')

            frames = self._get_frames(face_crop_folder)

            if self.audio_augment:
                freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
                timem = torchaudio.transforms.TimeMasking(self.timem)
                fbank = torch.transpose(fbank, 0, 1)
                fbank = fbank.unsqueeze(0)
                if self.freqm != 0:
                    fbank = freqm(fbank)
                if self.timem != 0:
                    fbank = timem(fbank)
                fbank = fbank.squeeze(0)
                fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if self.skip_norm == False:
            fbank = (fbank - self.norm_mean) / (self.norm_std)
        # skip normalization the input ONLY when you are trying to get the normalization stats.
        else:
            pass

        if self.audio_augment and random.random() < 0.5:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10

        # fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        # frames: (T, C, H, W) -> (C, T, H, W)
        frames = frames.permute(1, 0, 2, 3)

        # label = torch.tensor([int(label), 1-int(label)]).float()
        classify_loss = "BCE"
        if classify_loss == "BCE":
            label = torch.tensor(int(label), dtype=torch.float32)   # 确保label是浮点数
            audio_label = torch.tensor(int(audio_label), dtype=torch.float32) 
            video_label = torch.tensor(int(video_label), dtype=torch.float32)
        else: # CE loss
            label = torch.tensor(int(label), dtype=torch.long)  # 确保label是整数
            audio_label = torch.tensor(int(audio_label), dtype=torch.long)
            video_label = torch.tensor(int(video_label), dtype=torch.long)


        return fbank, frames, audio_label, video_label, label
    
    def __len__(self):
        return self.num_samples

    def labels_distribution(self):
        counts = Counter(self.comb_labels)
        print(counts)
        return counts

    def get_comb_weighted_sampler(self):
        """生成按4个组合类别加权的 WeightedRandomSampler"""
        counts = self.labels_distribution()
        num_samples = len(self.comb_labels)

        # 每个类别的权重 = 样本总数 / 类别样本数
        class_weights = {cls: num_samples / count for cls, count in counts.items()}
        #class_weights[0] *= 2

        # 为每个样本分配权重
        sample_weights = [class_weights[label] for label in self.comb_labels]

        # 创建 WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,
            replacement=True
        )
        return sampler


class Visual_Preprocess:
    def __init__(self, im_res):
        self.im_res = im_res

    def __call__(self, frames, visual_augment):
        """
        frames: List[PIL.Image], length = T
        return: Tensor [T, 3, H, W]
        """

        # ---------- 1. resize ----------
        frames = [F.resize(f, (self.im_res, self.im_res)) for f in frames]

        if not visual_augment:
            return torch.stack([F.to_tensor(f) for f in frames])

        # ---------- 2. 采样一次随机参数 ----------

        do_flip = random.random() < 0.5   # 你可以调成 0.2~0.3

        # RandomResizedCrop
        i, j, h, w = T.RandomResizedCrop.get_params(
            frames[0],
            scale=(0.8, 1.0),
            ratio=(0.75, 1.33),
        )

        # ColorJitter
        brightness = random.uniform(1 - 0.2, 1 + 0.2)
        contrast   = random.uniform(1 - 0.2, 1 + 0.2)
        saturation = random.uniform(1 - 0.2, 1 + 0.2)
        hue        = random.uniform(-0.1, 0.1)


        # Gaussian blur
        do_blur = random.random() < 0.2
        if do_blur:
            sigma = random.uniform(0.1, 2.0)

        # JPEG
        do_jpeg = random.random() < 0.3
        if do_jpeg:
            jpeg_quality = random.randint(30, 95)

        # Down-Up resize
        do_downup = random.random() < 0.3
        if do_downup:
            scale = random.uniform(0.5, 0.9)

        # ---------- 3. 应用到所有帧 ----------
        out = []
        for f in frames:
            if do_flip:
                f = F.hflip(f)

            f = F.resized_crop(f, i, j, h, w, (self.im_res, self.im_res))

            if do_downup:
                small = int(self.im_res * scale)
                f = F.resize(f, (small, small))
                f = F.resize(f, (self.im_res, self.im_res))

            if do_jpeg:
                f = self.jpeg_compress(f, jpeg_quality)

            if do_blur:
                f = F.gaussian_blur(f, kernel_size=13, sigma=sigma)

            f = F.adjust_brightness(f, brightness)
            f = F.adjust_contrast(f, contrast)
            f = F.adjust_saturation(f, saturation)
            f = F.adjust_hue(f, hue)


            out.append(F.to_tensor(f))

        return torch.stack(out)

    def jpeg_compress(self, img, quality):
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)