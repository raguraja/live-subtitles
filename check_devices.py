import pyaudiowpatch as pyaudio
p = pyaudio.PyAudio()
print('WASAPI loopback devices:')
for i in range(p.get_device_count()):
    d = p.get_device_info_by_index(i)
    if d.get('isLoopbackDevice'):
        print(f'  [{i}] {d["name"]} ({int(d["defaultSampleRate"])}Hz, {d["maxInputChannels"]}ch)')
p.terminate()
