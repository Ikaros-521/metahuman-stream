import time
import numpy as np
import soundfile as sf
import resampy
import asyncio
import edge_tts

from typing import Iterator

import requests

import queue
from queue import Queue
from io import BytesIO
from threading import Thread, Event
from enum import Enum

class State(Enum):
    RUNNING=0
    PAUSE=1

class BaseTTS:
    def __init__(self, opt, parent):
        self.opt=opt
        self.parent = parent

        self.fps = opt.fps # 20 ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // self.fps # 320 samples per chunk (20ms * 16000 / 1000)
        self.input_stream = BytesIO()

        self.msgqueue = Queue()
        self.state = State.RUNNING

    def pause_talk(self):
        self.msgqueue.queue.clear()
        self.state = State.PAUSE

    def put_msg_txt(self,msg): 
        if len(msg)>0:
            self.msgqueue.put(msg)

    def render(self,quit_event):
        process_thread = Thread(target=self.process_tts, args=(quit_event,))
        process_thread.start()
    
    def process_tts(self,quit_event):        
        while not quit_event.is_set():
            try:
                msg = self.msgqueue.get(block=True, timeout=1)
                self.state=State.RUNNING
            except queue.Empty:
                continue
            self.txt_to_audio(msg)
        print('ttsreal thread stop')
    
    def txt_to_audio(self,msg):
        pass
    

###########################################################################################
class EdgeTTS(BaseTTS):
    def txt_to_audio(self,msg):
        voicename = "zh-CN-YunxiaNeural"
        text = msg
        t = time.time()
        asyncio.new_event_loop().run_until_complete(self.__main(voicename,text))
        print(f'-------edge tts time:{time.time()-t:.4f}s')
        if self.input_stream.getbuffer().nbytes<=0: #edgetts err
            print('edgetts err!!!!!')
            return
        
        self.input_stream.seek(0)
        stream = self.__create_bytes_stream(self.input_stream)
        streamlen = stream.shape[0]
        idx=0
        while streamlen >= self.chunk and self.state==State.RUNNING:
            self.parent.put_audio_frame(stream[idx:idx+self.chunk])
            streamlen -= self.chunk
            idx += self.chunk
        #if streamlen>0:  #skip last frame(not 20ms)
        #    self.queue.put(stream[idx:])
        self.input_stream.seek(0)
        self.input_stream.truncate() 

    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        print(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            print(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            print(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream
    
    async def __main(self,voicename: str, text: str):
        try:
            communicate = edge_tts.Communicate(text, voicename)

            #with open(OUTPUT_FILE, "wb") as file:
            first = True
            async for chunk in communicate.stream():
                if first:
                    first = False
                if chunk["type"] == "audio" and self.state==State.RUNNING:
                    #self.push_audio(chunk["data"])
                    self.input_stream.write(chunk["data"])
                    #file.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    pass
        except Exception as e:
            print(e)

###########################################################################################
class VoitsTTS(BaseTTS):
    def txt_to_audio(self,msg): 
        self.stream_tts(
            self.gpt_sovits(
                msg,
                self.opt.REF_FILE,  
                self.opt.REF_TEXT,
                "zh", #en args.language,
                self.opt.TTS_SERVER, #"http://127.0.0.1:5000", #args.server_url,
            )
        )

    def gpt_sovits(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        req={
            'text':text,
            'text_lang':language,
            'ref_audio_path':reffile,
            'prompt_text':reftext,
            'prompt_lang':language,
            'media_type':'raw',
            'streaming_mode':True
        }
        # req["text"] = text
        # req["text_language"] = language
        # req["character"] = character
        # req["emotion"] = emotion
        # #req["stream_chunk_size"] = stream_chunk_size  # you can reduce it to get faster response, but degrade quality
        # req["streaming_mode"] = True
        try:
            res = requests.post(
                f"{server_url}/tts",
                json=req,
                stream=True,
            )
            end = time.perf_counter()
            print(f"gpt_sovits Time to make POST: {end-start}s")

            if res.status_code != 200:
                print("Error:", res.text)
                return
                
            first = True
        
            for chunk in res.iter_content(chunk_size=12800): # 1280 32K*20ms*2
                if first:
                    end = time.perf_counter()
                    print(f"gpt_sovits Time to first chunk: {end-start}s")
                    first = False
                if chunk and self.state==State.RUNNING:
                    yield chunk
            #print("gpt_sovits response.elapsed:", res.elapsed)
        except Exception as e:
            print(e)

    def stream_tts(self,audio_stream):
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=32000, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk])
                    streamlen -= self.chunk
                    idx += self.chunk 

class GSVV2TTS(BaseTTS):
    def txt_to_audio(self, msg): 
        # 将输入的文本消息转换为音频
        self.stream_tts(
            self.gpt_sovits(
                msg,  # 待转换的文本
                self.opt.REF_FILE,  # 参考音频文件的路径
                self.opt.REF_TEXT,  # 参考文本，用于音频合成
                "zh",  # 语言参数，这里指定为中文
                self.opt.TTS_SERVER,  # TTS服务器的URL
            )
        )

    def gpt_sovits(self, text, reffile, reftext, language, server_url) -> Iterator[bytes]:
        # 使用gpt_sovits模型将文本转换为音频，返回音频数据流
        start = time.perf_counter()  # 记录开始时间
        req = {
            'text': text,  # 要转换的文本
            "text_lang": language,  # 文本的语言
            "ref_audio_path": reffile,  # 参考音频文件路径
            "aux_ref_audio_paths": [],  # 辅助参考音频路径列表
            "prompt_lang": "zh",  # 提示的语言，这里是中文
            "prompt_text": reftext,  # 提示文本
            "top_k": 5,  # 采样时考虑的最高概率的k个选项
            "top_p": 1.0,  # 采样时的核采样参数
            "temperature": 1.0,  # 采样的温度参数，影响生成的随机性
            "text_split_method": "cut0",  # 文本分割方法
            "batch_size": 1,  # 每次处理的批次大小
            "batch_threshold": 0.75,  # 批处理的阈值
            "split_bucket": True,  # 是否使用分割桶
            "speed_factor": 1.0,  # 音频播放的速度因子
            "fragment_interval": 0.3,  # 音频片段间隔
            "seed": -1,  # 随机种子
            "media_type": "wav",  # 媒体类型，指定为wav格式
            "streaming_mode": True,  # 是否使用流模式
            "parallel_infer": True,  # 是否并行推理
            "repetition_penalty": 1.35  # 重复惩罚系数
        }
        # 发送POST请求到TTS服务器
        res = requests.post(
            f"{server_url}/tts",
            json=req,  # 请求体为JSON格式
            stream=True,  # 以流方式接收响应
        )
        end = time.perf_counter()  # 记录结束时间
        print(f"gpt_sovits Time to make POST: {end - start}s")  # 打印请求所耗时间

        if res.status_code != 200:  # 检查请求是否成功
            print("Error:", res.text)  # 打印错误信息
            return
            
        first = True  # 标志变量，指示是否为第一次接收音频块
        for chunk in res.iter_content(chunk_size=32000):  # 每次读取32KB的音频数据
            if first:  # 如果是第一次接收
                end = time.perf_counter()  # 记录结束时间
                print(f"gpt_sovits v2 Time to first chunk: {end - start}s")  # 打印第一次接收所耗时间
                first = False  # 设置标志为False，表示已接收过第一次数据
            if chunk and self.state == State.RUNNING:  # 检查块是否有效且状态为运行中
                yield chunk  # 返回音频数据块

        print("gpt_sovits v2 response.elapsed:", res.elapsed)  # 打印响应所消耗的时间

    def stream_tts(self, audio_stream):
        # 将音频流转换为音频帧
        for chunk in audio_stream:  # 遍历音频流中的每一个块
            if chunk is not None and len(chunk) > 0:  # 检查块有效性
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767  # 将音频块转换为浮点型
                stream = resampy.resample(x=stream, sr_orig=32000, sr_new=self.sample_rate)  # 重采样为目标采样率
                streamlen = stream.shape[0]  # 获取音频流的长度
                idx = 0  # 初始化索引
                while streamlen >= self.chunk:  # 当剩余音频长度大于等于设置的块大小
                    self.parent.put_audio_frame(stream[idx:idx + self.chunk])  # 将音频帧发送到父对象
                    streamlen -= self.chunk  # 减去已处理的块大小
                    idx += self.chunk  # 更新索引


###########################################################################################
class CosyVoiceTTS(BaseTTS):
    def txt_to_audio(self,msg): 
        self.stream_tts(
            self.cosy_voice(
                msg,
                self.opt.REF_FILE,  
                self.opt.REF_TEXT,
                "zh", #en args.language,
                self.opt.TTS_SERVER, #"http://127.0.0.1:5000", #args.server_url,
            )
        )

    def cosy_voice(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        payload = {
            'tts_text': text,
            'prompt_text': reftext
        }
        try:
            files = [('prompt_wav', ('prompt_wav', open(reffile, 'rb'), 'application/octet-stream'))]
            res = requests.request("GET", f"{server_url}/inference_zero_shot", data=payload, files=files, stream=True)
            
            end = time.perf_counter()
            print(f"cosy_voice Time to make POST: {end-start}s")

            if res.status_code != 200:
                print("Error:", res.text)
                return
                
            first = True
        
            for chunk in res.iter_content(chunk_size=8820): # 882 22.05K*20ms*2
                if first:
                    end = time.perf_counter()
                    print(f"cosy_voice Time to first chunk: {end-start}s")
                    first = False
                if chunk and self.state==State.RUNNING:
                    yield chunk
        except Exception as e:
            print(e)

    def stream_tts(self,audio_stream):
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=22050, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk])
                    streamlen -= self.chunk
                    idx += self.chunk 

###########################################################################################
class XTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt,parent)
        self.speaker = self.get_speaker(opt.REF_FILE, opt.TTS_SERVER)

    def txt_to_audio(self,msg): 
        self.stream_tts(
            self.xtts(
                msg,
                self.speaker,
                "zh-cn", #en args.language,
                self.opt.TTS_SERVER, #"http://localhost:9000", #args.server_url,
                "20" #args.stream_chunk_size
            )
        )

    def get_speaker(self,ref_audio,server_url):
        files = {"wav_file": ("reference.wav", open(ref_audio, "rb"))}
        response = requests.post(f"{server_url}/clone_speaker", files=files)
        return response.json()

    def xtts(self,text, speaker, language, server_url, stream_chunk_size) -> Iterator[bytes]:
        start = time.perf_counter()
        speaker["text"] = text
        speaker["language"] = language
        speaker["stream_chunk_size"] = stream_chunk_size  # you can reduce it to get faster response, but degrade quality
        try:
            res = requests.post(
                f"{server_url}/tts_stream",
                json=speaker,
                stream=True,
            )
            end = time.perf_counter()
            print(f"xtts Time to make POST: {end-start}s")

            if res.status_code != 200:
                print("Error:", res.text)
                return

            first = True
        
            for chunk in res.iter_content(chunk_size=9600): #24K*20ms*2
                if first:
                    end = time.perf_counter()
                    print(f"xtts Time to first chunk: {end-start}s")
                    first = False
                if chunk:
                    yield chunk
        except Exception as e:
            print(e)
    
    def stream_tts(self,audio_stream):
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk])
                    streamlen -= self.chunk
                    idx += self.chunk 