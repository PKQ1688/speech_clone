#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2023/8/10 18:50
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2023/8/10 18:50
# @File         : base_data_process.py
import os

from pydub import AudioSegment


def convert_m4a_to_wav(m4a_path, wav_path):
    audio = AudioSegment.from_file(m4a_path, format="m4a")
    audio.export(wav_path, format="wav")


origin_file = "data/speech_clone/diandian_tmp"
origin_file_list = os.listdir(origin_file)
print(origin_file_list)

for index, file_name in enumerate(origin_file_list):
    print(file_name)
    m4a_path = os.path.join(origin_file, file_name)
    wav_path = os.path.join(origin_file, str(index) + ".wav")
    convert_m4a_to_wav(m4a_path, wav_path)
    os.remove(m4a_path)
