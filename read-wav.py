import wave
import struct

def parse_wave(filepath):
    with wave.open(filepath, "rb") as wave_file:
        sample_rate = wave_file.getframerate()
        length_in_sec = wave_file.getnframes() / sample_rate

        first_sample = struct.unpack('<h', wave_file.readframes(1))[0]
        second_sample = struct.unpack('<h', wave_file.readframes(1))[0]

        print('''
Parsed {filename}
-----------------------------------------------
Channels: {num_channels}
Sample Rate: {sample_rate}
First Sample: {first_sample}
Second Sample: {second_sample}
Length in Seconds: {length_in_seconds}'''.format(
            filename=filepath,
            num_channels=wave_file.getnchannels(),
            sample_rate=wave_file.getframerate(),
            first_sample=first_sample,
            second_sample=second_sample,
            length_in_seconds=length_in_sec))
        return

parse_wave("wav/major/scale_c_major.wav")
