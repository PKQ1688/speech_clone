import struct

import librosa
import numpy as np
from modelscope.utils.audio.audio_utils import ndarray_pcm_to_wav


def get_sr_from_bytes(wav: bytes):
    sr = None
    data = wav
    if len(data) > 44:
        try:
            header_fields = {}
            header_fields['ChunkID'] = str(data[0:4], 'UTF-8')
            header_fields['Format'] = str(data[8:12], 'UTF-8')
            header_fields['Subchunk1ID'] = str(data[12:16], 'UTF-8')
            if header_fields['ChunkID'] == 'RIFF' and header_fields[
                    'Format'] == 'WAVE' and header_fields[
                        'Subchunk1ID'] == 'fmt ':
                header_fields['SampleRate'] = struct.unpack('<I',
                                                            data[24:28])[0]
                sr = header_fields['SampleRate']
        except Exception:
            # no treatment
            pass

    return sr


def wav_to_ndarray(wav_data, target_sr):
    # byte(PCM16) to float32, and resample
    value = wav_data[44:]
    middle_data = np.frombuffer(value, dtype=np.int16)
    middle_data = np.asarray(middle_data)
    if middle_data.dtype.kind not in 'iu':
        raise TypeError("'middle_data' must be an array of integers")
    dtype = np.dtype('float32')
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(middle_data.dtype)
    abs_max = 2**(i.bits - 1)
    offset = i.min + abs_max
    waveform = np.frombuffer((middle_data.astype(dtype) - offset) / abs_max,
                             dtype=np.float32)
    audio_sr = get_sr_from_bytes(wav_data)

    waveform = ndarray_resample(waveform, audio_sr, target_sr)

    return (target_sr, waveform)


def ndarray_to_wav(ndarray, sr, target_sr):
    # ndarray:int32 to float32, and output wav bytes
    #  print(ndarray, flush=True)
    #  print(ndarray.dtype, flush=True)

    ndarray_i16 = None
    ndarray_f32 = None
    if ndarray.dtype == np.int32:
        # ndarray:int32 to int16
        ndarray_i16 = (ndarray >> 16).astype(np.int16)
        print("dtype is int32", flush=True)
    elif ndarray.dtype == np.int16:
        ndarray_i16 = ndarray
        print("dtype is int16", flush=True)
    elif ndarray.dtype == np.float32:
        ndarray_f32 = ndarray
        print("dtype is float32", flush=True)
    else:
        raise TypeError("'dtype' must be int16 int32 float32 type")

    if ndarray_i16 is not None and ndarray_i16.dtype.kind not in 'iu':
        raise TypeError("'middle_data' must be an array of integers")
    dtype = np.dtype('float32')
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    if ndarray_i16 is not None:
        # ndarray:int16 to float32
        i = np.iinfo(ndarray_i16.dtype)
        abs_max = 2**(i.bits - 1)
        offset = i.min + abs_max
        ndarray_f32 = np.frombuffer((ndarray_i16.astype(dtype) - offset) / abs_max,
                                    dtype=np.float32)

    if ndarray_f32 is None:
        raise TypeError("invalid 'ndarray_f32'")

    data = ndarray_resample(ndarray_f32, sr, target_sr)
    return ndarray_pcm_to_wav(target_sr, data)


def ndarray_resample(audio_in: np.ndarray,
                     fs_in: int = 16000,
                     fs_out: int = 16000) -> np.ndarray:
    audio_out = audio_in
    if fs_in != fs_out:
        audio_out = librosa.resample(audio_in, orig_sr=fs_in, target_sr=fs_out)
    return audio_out