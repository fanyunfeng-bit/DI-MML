import os
import moviepy.editor as mp

def extract_audio(videos_file_path):
    my_clip = mp.VideoFileClip(videos_file_path)
    return my_clip






all_videos = r'D:\data\AVE_Dataset\Annotations.txt'
all_audio_dir = r'D:\data\AVE_Dataset\Audios'
if not os.path.exists(all_audio_dir):
    os.makedirs(all_audio_dir)

# train set processing
with open(all_videos, 'r') as f:
    files = f.readlines()

for i, item in enumerate(files[1:]):
    if i % 500 == 0:
        print('*******************************************')
        print('{}/{}'.format(i, len(files)))
        print('*******************************************')
    item = item.split('&')
    mp4_filename = os.path.join(r'D:\data\AVE_Dataset\AVE', item[1] + '.mp4')
    wav_filename = os.path.join(all_audio_dir, item[1]+'.wav')
    if os.path.exists(wav_filename):
        pass
    else:
        my_clip = extract_audio(mp4_filename)
        my_clip.audio.write_audiofile(wav_filename)

        #os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}'.format(mp4_filename, wav_filename))

