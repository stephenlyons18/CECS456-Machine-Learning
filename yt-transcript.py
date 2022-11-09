#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip3 install youtube_dl
# !pip install git+https://github.com/openai/whisper.git
# !pip3 install setuptools-rust
# !apt -y install ffmpeg


# In[ ]:


from __future__ import unicode_literals
import youtube_dl
import whisper
import warnings
warnings.filterwarnings("ignore")

# Download the video
ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': 'video.%(ext)s',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192',
    }],
}
link = input("Enter the link of the video: ")

# get the title of the video
with youtube_dl.YoutubeDL({'outtmpl': '%(id)s%(ext)s'}) as ydl:
    info_dict = ydl.extract_info(link, download=False)
    video_title = info_dict.get('title', None)



# In[ ]:


with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download([link])



# In[ ]:


# model
model = whisper.load_model("base")


# Transcribe the video into english text
transcript = model.transcribe("./video.wav")

# Print the transcript
print(transcript)


# In[8]:
# clean the title of the video to use it as the name of the file (remove special characters)
video_title = video_title.replace(" ", "_").replace(":", "").replace("?", "").replace("/", "").replace("\\", "").replace("|", "").replace("*", "").replace("<", "").replace(">", "").replace("\"", "").replace(".", "")



# save the transcript of the video in a text file with the name of the video
with open(video_title + ".txt", "w") as text_file:
    text_file.write(transcript['text'])


# delete the downloaded video file
import os
os.remove("video.wav")

