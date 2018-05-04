# coding: utf-8
import glob
import os
import pandas as pd



df = pd.read_csv('/home/ahmed/Pictures/cogedis/cogedis_words_3/words.csv',sep=',')
df = df.astype(str)
print(len(df))
#df[['raw_value', 'manual_raw_value']] =  df[['raw_value', 'manual_raw_value']][~df[['raw_value', 'manual_raw_value']].applymap(lambda x: any([xx in ['{', ',','à','â','$','€', ';', ':', '\\', '/', '.', '%','_','-','}'] for xx in x]))]
#df.dropna(axis = 0, how = 'any', inplace = True)
#df=df.replace(['é','è'],'e', regex=True)
#df.dropna()
#df = df.applymap(lambda x: x.lower())

#df[['raw_value', 'manual_raw_value']]= df[['raw_value', 'manual_raw_value']].applymap(lambda x: x.replace('é','e'))
#df = df.applymap(lambda x: x.replace('è','e'))
#df['filtering'] = df['raw_value'].apply(lambda x : 1 if x.str.contains(',',',','à','â','$','€', ';', ':', '\\', '/', '.', '%','_','-') else 0)
#df['filtering'] = df['raw_value'].apply(lambda x : 1 if x.str.contains('à') else 0)

a = [ '\,','à','â','à','È','À','-','_', ';', '\:', '\\\\', '\/', '\.', '\$', '€', '\%', '_', '-','°','<','>']


joined = "|".join(a)
mask = ~df['manual_raw_value'].str.contains(joined)
cols = ['manual_raw_value']
df = df[mask].astype(str).replace(['é','è','È','É'],'e', regex=True).apply(lambda x: x.str.lower()).reset_index(drop=True)
df = df.astype(str)
df.to_csv('/home/ahmed/Pictures/cogedis/cogedis_words_large/words_processed.csv',index=False,sep=',')

#df['filtering'] = df['raw_value'].apply(lambda x : 1 if x.contains(['à', 'é']) else 0)