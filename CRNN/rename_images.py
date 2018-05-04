import pandas as pd
import os
import glob

df = pd.read_csv('/home/ahmed/Pictures/cogedis/cogedis_words_3/words.csv',sep=',')
df = df.astype(str)
path='/home/ahmed/Pictures/cogedis/cogedis_words_3/'
os.chdir(path)
images_name = glob.glob("*.png")
set_img = set([x.rsplit('.', 1)[0] for x in images_name])
i=0
for img in set_img:
    label=df.loc[df.id==img, 'manual_raw_value'].item()
    print('yes')
    os.rename(img+'.png',label+'.png')
    print(i)
    i +=1
    print('ok')

