"""
郭泽斌于2023.04.29参照app.py创建，用于连接github开源项目 Fay 数字人
"""

import time
import json
from tools import audio_pre_process, generate_video, audio_process
import os
import threading
import websocket
from pydub import AudioSegment
import cv2
import pygame
import hashlib

video_list = []

fay_ws = None
video_cache = {}

#增加MD5音频标记，避免重复生成视频
def hash_file_md5(filepath):
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        while True:
            data = f.read(65536)  # Read in 64kb chunks
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()

def connet_fay():
    global video_list
    global video_cache
    def on_message(ws, message):
        if "audio" in message:
            message_dict = json.loads(message)
            aud_dir = message_dict["Data"]["Value"]
            aud_dir = aud_dir.replace("\\", "/")
            print('message:', aud_dir, type(aud_dir))
            basedir = ""
            for i in aud_dir.split("/"):
                basedir = os.path.join(basedir,i)
            basedir = basedir.replace(":",":\\")   
            num = time.time()
            new_path = r'./inputs/test/source_audio/aud_%d.wav'%num  #新路径                
            old_path = basedir                
 
            convert_mp3_to_wav(old_path, new_path) 
            audio_hash = hash_file_md5(new_path)
            if audio_hash in video_cache:
                video_list.append({"video": video_cache[audio_hash], "audio": new_path})
                print("视频已存在，直接播放。")
            else:
                audio_path = 'inputs/test/source_audio/aud_%d.wav' % num
                audio_process(audio_path)
                video_path = 'inputs/test/source_video/001.mp4' 
                output_path = 'results/output_%d.mp4' % num
                generate_video(audio_path, video_path, output_path)
                video_list.append({"video" : output_path, "audio" : new_path})
                video_cache[audio_hash] = output_path

    def on_error(ws, error):
        print(f"Fay Error: {error}")
        fay_ws = None
        time.sleep(5)
        connect()

    def on_close(ws):
        print("Fay Connection closed")
        fay_ws = None

    def on_open(ws):
        print("Fay Connection opened")

    def connect():
        ws_url = "ws://127.0.0.1:10002"
        print("connect")
        fay_ws = websocket.WebSocketApp(ws_url,
                                    on_message=on_message,
                                    on_error=on_error,
                                    on_close=on_close)
        fay_ws.on_open = on_open
        fay_ws.run_forever()
    connect()

def convert_mp3_to_wav(input_file, output_file):
    audio = AudioSegment.from_mp3(input_file)
    audio.export(output_file, format='wav')


def play_video():
    global video_list
    video_path = None
    audio_path = None
    ret = None
    frame = None
    while True:
        if len(video_list) > 0:
            video_path = video_list[0].get("video")
            audio_path = video_list[0].get("audio")
            cap = cv2.VideoCapture(video_path)  # 打开视频文件
            video_list.pop(0)
        else:
            audio_path = None
            cap =  cv2.VideoCapture("results/result_voice_1.mp4")
            # _, frame = cv2.VideoCapture("results/result_voice_1.mp4").read()

        if audio_path:
            threading.Thread(target=play_audio, args=[audio_path]).start()  # play audio
        # 循环播放视频帧
        while True:
            if cap:
                ret, frame = cap.read()
            if frame is not None:#没有传音频过来时显示train.mp4的第一帧，建议替换成大约1秒左右的视频
                cv2.imshow('Fay-2d', frame)
                # 等待 38 毫秒
                cv2.waitKey(38)
            if not ret:
                break

# def play_video():
#     global video_list
#     video_path = None
#     audio_path = None
#     ret = None
#     frame = None
#     video_duration = 1.0  # 视频片段时长，单位秒
#     frame_rate = 30  # 帧率，可以根据实际情况修改
#     frame_count = int(frame_rate * video_duration)
#     frame_idx = 0
#     timer = time.time()  # 记录开始播放的时间

#     while True:
#         if len(video_list) > 0:
#             video_path = video_list[0].get("video")
#             audio_path = video_list[0].get("audio")
#             cap = cv2.VideoCapture(video_path)  # 打开视频文件
#             video_list.pop(0)
#             timer = time.time()  # 重置计时器
#             frame_idx = 0  # 重置帧计数
#         else:
#             audio_path = None
#             cap = None
#             if time.time() - timer > video_duration:
#                 frame_idx = 0  # 重置帧计数
#                 _, frame = cv2.VideoCapture("inputs/test/source_video/default.mp4").read()

#         if audio_path:
#             threading.Thread(target=play_audio, args=[audio_path]).start()  # play audio

#         if cap:
#             ret, frame = cap.read()
#         if frame is not None:
#             cv2.imshow('Fay-2d', frame)
#             # 等待 1000 / 帧率 毫秒
#             time.sleep(1.0 / frame_rate)

#         frame_idx += 1
#         if frame_idx >= frame_count:
#             frame_idx = 0  # 重置帧计数

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     if cap:
#         cap.release()
#     cv2.destroyAllWindows()

def play_audio(audio_file):
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.stop()


if __name__ == '__main__':

    audio_pre_process()

    threading.Thread(target=connet_fay, args=[]).start()
    play_video()


    
    
