import torch
import numpy as np

# from .utils import *
import time
import cv2
import copy

import queue
from threading import Thread, Event
import multiprocessing as mp


from lipasr import LipASR
import asyncio
from av import AudioFrame, VideoFrame
from basereal import BaseReal
 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
 

def __mirror_index(size, index):
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1 

def inference(render_event, batch_size, face_list_cycle, audio_feat_queue, audio_out_queue, res_frame_queue, gol_model):
    
    length = len(face_list_cycle)
    index = 0
    print('started real video service...')

    while True:
        if render_event.is_set():
            mel_batch = []
            try:
                mel_batch = audio_feat_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
                
            is_all_silence=True
            audio_frames = []

            for _ in range(batch_size * 2):
                frame,type = audio_out_queue.get()
                audio_frames.append((frame,type))
                if type==0:
                    is_all_silence=False

            if is_all_silence:
                for i in range(batch_size):
                    res_frame_queue.put((None,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                    index = index + 1
            else:
                
                img_batch = []
                for i in range(batch_size):
                    idx = __mirror_index(length,index+i)
                    face = face_list_cycle[idx]
                    img_batch.append(face)
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, face.shape[0]//2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
                
                img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
                mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

                with torch.no_grad():
                    pred = gol_model(mel_batch, img_batch)
                pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.


                for i,res_frame in enumerate(pred):
                    
                    res_frame_queue.put((res_frame,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                    index = index + 1

                torch.cuda.empty_cache()
        else:
            time.sleep(1)
    
@torch.no_grad()
class LipReal(BaseReal):

    def __init__(self, opt, avatar_model_class):
        """
        初始化LipReal类的实例。

        参数:
        - opt: 一个包含配置选项的对象。

        属性:
        - W: 配置中的宽度。
        - H: 配置中的高度。
        - fps: 配置中的帧率。
        - avatar_id: 配置中的角色ID。
        - avatar_path: 角色的路径。
        - batch_size: 配置中的批量大小。
        - idx: 当前索引。
        - res_frame_queue: 结果帧队列。
        - asr: LipASR 实例。
        - render_event: 渲染事件。

        方法:
        - __loadavatar: 加载角色的头像。
        - asr.warm_up: 预热ASR模型。
        """
        super().__init__(opt)

        self.W = opt.W
        self.H = opt.H

        self.fps = opt.fps # 20 ms per frame

        #### musetalk
        self.avatar_id = opt.avatar_id
        self.avatar_path = f"./data/avatars/{self.avatar_id}"
        self.batch_size = opt.batch_size
        self.idx = 0
        self.res_frame_queue = mp.Queue(self.batch_size*2)

        # 传入一坨对象
        self.coord_list_cycle = avatar_model_class.gol_coord_list_cycle
        self.frame_list_cycle = avatar_model_class.gol_frame_list_cycle

        self.asr = LipASR(opt,self)
        self.asr.warm_up()

        self.render_event = mp.Event()
        mp.Process(target=inference, args=(self.render_event,self.batch_size,avatar_model_class.gol_face_list_cycle, self.asr.feat_queue,self.asr.output_queue,self.res_frame_queue, avatar_model_class.gol_model)).start()

    def process_frames(self, quit_event, loop=None, audio_track=None, video_track=None):
        """
        处理视频帧的方法。该方法从结果帧队列中获取处理后的视频帧，并根据音频是否静音来决定是否使用自定义视频或原始视频帧。
        然后，它将处理后的视频帧和音频帧放入相应的队列中。

        参数:
        - quit_event: 一个事件对象，用于通知线程何时退出。
        - loop: 一个事件循环对象，用于在异步环境中运行协程。
        - audio_track: 一个音频轨道对象，用于将音频帧放入队列中。
        - video_track: 一个视频轨道对象，用于将视频帧放入队列中。

        返回:
        - None

        异常:
        - queue.Empty: 如果结果帧队列为空，则会捕获此异常并继续循环。

        注意:
        - 该方法会在一个无限循环中运行，直到 quit_event 被设置。
        - 如果音频帧的类型不是静音，则会将处理后的视频帧放入视频轨道队列中，并将音频帧放入音频轨道队列中。
        - 如果音频帧的类型是静音，则会根据是否有自定义视频来决定使用哪个视频帧。
        """
        while not quit_event.is_set():
            try:
                res_frame, idx, audio_frames = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            # 全为静音数据，只需要取fullimg
            if audio_frames[0][1]!=0 and audio_frames[1][1]!=0: 
                self.speaking = False
                audiotype = audio_frames[0][1]

                if self.custom_index.get(audiotype) is not None: #有自定义视频
                    mirindex = self.mirror_index(len(self.custom_img_cycle[audiotype]),self.custom_index[audiotype])
                    combine_frame = self.custom_img_cycle[audiotype][mirindex]
                    self.custom_index[audiotype] += 1
                
                else:
                    combine_frame = self.frame_list_cycle[idx]

            else:
                self.speaking = True
                bbox = self.coord_list_cycle[idx]
                combine_frame = copy.deepcopy(self.frame_list_cycle[idx])
                 
                y1, y2, x1, x2 = bbox
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
                except:
                    continue
                combine_frame[y1:y2, x1:x2] = res_frame

            image = combine_frame
            new_frame = VideoFrame.from_ndarray(image, format="bgr24")
            asyncio.run_coroutine_threadsafe(video_track._queue.put(new_frame), loop)
            
            if self.recording:
                self.recordq_video.put(new_frame) 

            for audio_frame in audio_frames:
                frame, type = audio_frame
                frame = (frame * 32767).astype(np.int16)

                new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                new_frame.planes[0].update(frame.tobytes())
                new_frame.sample_rate=16000

                asyncio.run_coroutine_threadsafe(audio_track._queue.put(new_frame), loop)
                if self.recording:
                    self.recordq_audio.put(new_frame) 
            
            torch.cuda.empty_cache()



    # 开始渲染视频 把自己传入到 webrtc.py 中。然后启动线程 执行了这个render函数
    def render(self,quit_event, loop=None, audio_track=None, video_track=None):
        self.tts.render(quit_event)
        self.init_customindex()

        process_thread = Thread(target=self.process_frames, args=(quit_event,loop,audio_track,video_track))
        process_thread.start()

        # start infer process render
        self.render_event.set() 

        while not quit_event.is_set(): 

            self.asr.run_step()

            if video_track._queue.qsize() >= 5:
                time.sleep(0.04 * video_track._queue.qsize() * 0.8)

        self.render_event.clear() #end infer process render
