# coding: utf-8
import pandas as pd



df = pd.read_csv('/home/ahmed/Pictures/bytel/bytel_words_large/words_processed_alphabet_only.csv',sep=',')
df = df.astype(str)
# df = df.astype(str)
# df = pd.read_csv('file', dtype=str)

df['manual_crnn'] = (df['crnn_pred']==df['manual_raw_value']).astype(int)
df['manual_abby'] = (df['raw_value']==df['manual_raw_value']).astype(int)
df['abby_crnn'] = (df['crnn_pred']==df['raw_value']).astype(int)
df['abby_manual_crnn'] = ((df['crnn_pred']==df['manual_raw_value']) & (df['manual_raw_value']==df['raw_value'])).astype(int)
df['wrong_crnn']=(df['crnn_pred'] !=df['manual_raw_value']).astype(int)
df['wrong_abby'] = (df['raw_value']!=df['manual_raw_value']).astype(int)


#df['wrong_abby'] = ((df['crnn_pred']==df['manual_raw_value']) & (df['manual_raw_value']!=df['raw_value'])).astype(int)
#df.to_csv('/home/ahmed/Pictures/bytel/bytel_words_large/words_processed_char_output.csv',sep=',',index=False)
'''
df['manual_abby'] = (df['raw_value']==df['manual_raw_value']).astype(int)
x = df[df.manual_raw_value == df.raw_value]
df.to_csv('/home/ahmed/Pictures/bytel/bytel_words_large/words_processed_digit_2.csv',index=False,sep=',')
'''
df['manual_raw_value'] = df['manual_raw_value'].astype(str)
df['raw_value'] = df['raw_value'].astype(str)
x = df[df.manual_raw_value == df.raw_value]
y = df[df.manual_raw_value == df.crnn_pred]
z= df[df.crnn_pred ==df.raw_value]
w = df[((df.manual_raw_value == df.crnn_pred) & (df.manual_raw_value == df.raw_value))]
t=df[((df['crnn_pred']==df['manual_raw_value']) & (df['manual_raw_value']!=df['raw_value']))]
k=df[df.raw_value != df.manual_raw_value]

print(" manual_raw_value == raw_value ", len(x), " out of ", len(df) )
print(" manual_raw_value == crnn_value ", len(y), " out of ", len(df) )
print("raw_value == crnn_value ",len(z), " out of ", len(df) )
print("raw_value == crnn_value==manual_raw_value ",len(w), " out of ", len(df))
print("abby wrong and crnn correct  ",len(t), " out of ", len(df) )
print("abby wrong  ",len(k), " out of ", len(df) )





