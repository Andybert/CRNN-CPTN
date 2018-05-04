# coding: utf-8
import glob
import os
import pandas as pd



df = pd.read_csv('/home/ahmed/Pictures/cogedis/cogedis_words_3/words.csv',sep=',')
print(len(df))
df = df.astype(str)
#df[['raw_value', 'manual_raw_value']] =  df[['raw_value', 'manual_raw_value']][~df[['raw_value', 'manual_raw_value']].applymap(lambda x: any([xx in ['{', ',','à','â','$','€', ';', ':', '\\', '/', '.', '%','_','-','}'] for xx in x]))]
#df.dropna(axis = 0, how = 'any', inplace = True)
#df=df.replace(['é','è'],'e', regex=True)
#df.dropna()
#df = df.applymap(lambda x: x.lower())

#df[['raw_value', 'manual_raw_value']]= df[['raw_value', 'manual_raw_value']].applymap(lambda x: x.replace('é','e'))
#df = df.applymap(lambda x: x.replace('è','e'))
#df['filtering'] = df['raw_value'].apply(lambda x : 1 if x.str.contains(',',',','à','â','$','€', ';', ':', '\\', '/', '.', '%','_','-') else 0)
#df['filtering'] = df['raw_value'].apply(lambda x : 1 if x.str.contains('à') else 0)



#a = ['0','1','2','3','4','5','6','7','8','9',':','/','.',',','%','$','€']
#joined = "|".join(a)
#mask = df['manual_raw_value'].str.contains(joined)
#cols = ['manual_raw_value']

#df = df[mask].astype(str).replace(['é','è','È','É'],'e', regex=True).apply(lambda x: x.str.lower()).reset_index(drop=True)

#b = ['a','b','c','d','e','f','g','h','i','g','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O'
 #    'P','Q','R','S','T','U','V','W','X','Y','Z']

#joined_2 = "|".join(b)
#mask_2 =  ~df['manual_raw_value'].str.contains(joined_2) | ~df['raw_value'].str.contains(joined_2)
#cols = ['raw_value','manual_raw_value']
#df = df[mask_2].reset_index(drop=True)

'''
a = ['0','1','2','3','4','5','6','7','8','9',':','/','.',',','%','$','€']
joined = "|".join(a)
mask =  df['manual_raw_value'].str.contains(joined)
cols = ['manual_raw_value']
df = df[mask].astype(str).replace(['é','è','È','É'],'e', regex=True).apply(lambda x: x.str.lower()).reset_index(drop=True)
'''

#df = df[~df['manual_raw_value'].str.isalpha().reset_index(drop=True)]
#df=df[~df.manual_raw_value.str.match(r'^[a-zA-Z-]*$')]
#a = r'^[\d]+$'
#a = r'^[\d:/.,%$€]+$'
#df=df[df.manual_raw_value.str.match(a)]
df.manual_raw_value=df.manual_raw_value.str.lower()
df.raw_value=df.raw_value.str.lower()
df=df.replace(['é','è','È','É'],'e', regex=True)
df = df[df.manual_raw_value.str.match(r'^[\da-z%,./@:]*$')]

print(len(df))
df.to_csv('/home/ahmed/Pictures/cogedis/cogedis_words_3/digit_alphabet_spec_char.csv',index=False,sep=',')
