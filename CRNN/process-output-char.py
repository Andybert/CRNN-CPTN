# coding: utf-8
import glob
import os
import pandas as pd

exported_file= '/home/ahmed/Pictures/cogedis/cogedis_words_large/words.csv'

# The code will be changed : use parse argument to take functions name's as parameter

def all_data(input_path):
    df = pd.read_csv(input_path, sep=',')
    return input_path



def special_char(input_path,output_path):
    df = pd.read_csv(input, sep=',')
    df = df.astype(str)
    a = [ '\,','à','â','à','-','_', ';', '\:', '\\\\', '\/', '\.', '\$', '€', '\%', '_', '-'] # it remains ( )  *
    joined = "|".join(a)
    mask = df['raw_value'].str.contains(joined) | df['manual_raw_value'].str.contains(joined)
    cols = ['raw_value','manual_raw_value']
    df = df[mask].astype(str).replace(['é','è','È','É'],'e', regex=True).apply(lambda x: x.str.lower()).reset_index(drop=True)

    df.to_csv(output_path,sep=',')
    return output_path


def digit_alpha(input_path,output_path):
    df = pd.read_csv(input, sep=',')
    df = df.astype(str)
    df.manual_raw_value = df.manual_raw_value.str.lower()
    df.raw_value = df.raw_value.str.lower()
    #df.crnn_pred = df.crnn_pred.str.lower()
    df=df[df.manual_raw_value.str.match(r'^[\da-zA-Z,:/.%@]*$')]
    df=df.replace(['é','è','È','É'],'e', regex=True)
    df = df.replace(['à','â'], 'a', regex=True)
    df.to_csv(output_path, index=False, sep=',')
    return output_path

def only_digits(input_path,output_path):
    df = pd.read_csv(input, sep=',')
    df = df.astype(str)
    a = r'^[\d]+$'
    df = df[df.manual_raw_value.str.match(a)]
    df.to_csv(output_path, index=False, sep=',')

    return output_path
def only_alphabet(input, output):
    df = pd.read_csv(input, sep=',')
    df = df.astype(str)
    df.manual_raw_value = df.manual_raw_value.str.lower()
    df.raw_value = df.raw_value.str.lower()
    df.crnn_pred = df.crnn_pred.str.lower()
    df=df[df.manual_raw_value.str.match(r'^[a-zA-Z]*$')]
    df=df.replace(['é','è','È','É'],'e', regex=True)
    df.to_csv(output, index=False, sep=',')
    return output


def eval_pred_all_data(input):
    output_all_data=all_data(input)
    df = pd.read_csv(output_all_data, sep=',')
    df = df.astype(str)
    df['manual_raw_value'] = df['manual_raw_value'].astype(str)
    df['raw_value'] = df['raw_value'].astype(str)
    df['crnn_pred'] = df['crnn_pred'].astype(str)
    x = df[df.manual_raw_value == df.raw_value]
    y = df[df.manual_raw_value == df.crnn_pred]
    z = df[df.crnn_pred == df.raw_value]
    w = df[((df.manual_raw_value == df.crnn_pred) & (df.manual_raw_value == df.raw_value))]
    t = df[((df['crnn_pred'] == df['manual_raw_value']) & (df['manual_raw_value'] != df['raw_value']))]
    k = df[df.raw_value != df.manual_raw_value]

    #df['bin_crnn'] = (df['crnn_pred'] == df['manual_raw_value']).astype(int)

    print(" manual_raw_value == raw_value ", len(x), " out of ", len(df))
    print(" manual_raw_value == crnn_value ", len(y), " out of ", len(df))
    print("raw_value == crnn_value ", len(z), " out of ", len(df))
    print("raw_value == crnn_value==manual_raw_value ", len(w), " out of ", len(df))
    print("abby wrong and crnn correct  ", len(t), " out of ", len(df))
    print("abby wrong  ", len(k), " out of ", len(df))
    return x,y,z,w,t,k,len(df)



def eval_pred_spec_char(input):
    output_all_data=special_char(input)
    df = pd.read_csv(output_all_data, sep=',')
    df = df.astype(str)
    df['manual_raw_value'] = df['manual_raw_value'].astype(str)
    df['raw_value'] = df['raw_value'].astype(str)
    df['crnn_pred'] = df['crnn_pred'].astype(str)
    x = df[df.manual_raw_value == df.raw_value]
    y = df[df.manual_raw_value == df.crnn_pred]
    z = df[df.crnn_pred == df.raw_value]
    w = df[((df.manual_raw_value == df.crnn_pred) & (df.manual_raw_value == df.raw_value))]
    t = df[((df['crnn_pred'] == df['manual_raw_value']) & (df['manual_raw_value'] != df['raw_value']))]
    k = df[df.raw_value != df.manual_raw_value]

    print(" manual_raw_value == raw_value ", len(x), " out of ", len(df))
    print(" manual_raw_value == crnn_value ", len(y), " out of ", len(df))
    print("raw_value == crnn_value ", len(z), " out of ", len(df))
    print("raw_value == crnn_value==manual_raw_value ", len(w), " out of ", len(df))
    print("abby wrong and crnn correct  ", len(t), " out of ", len(df))
    print("abby wrong  ", len(k), " out of ", len(df))
    return x, y, z, w, t, k, len(df)



def eval_pred_digits(input):
    output_all_data=only_digits(input)
    df = pd.read_csv(output_all_data, sep=',')
    df = df.astype(str)
    df['manual_raw_value'] = df['manual_raw_value'].astype(str)
    df['raw_value'] = df['raw_value'].astype(str)
    df['crnn_pred'] = df['crnn_pred'].astype(str)
    x = df[df.manual_raw_value == df.raw_value]
    y = df[df.manual_raw_value == df.crnn_pred]
    z = df[df.crnn_pred == df.raw_value]
    w = df[((df.manual_raw_value == df.crnn_pred) & (df.manual_raw_value == df.raw_value))]
    t = df[((df['crnn_pred'] == df['manual_raw_value']) & (df['manual_raw_value'] != df['raw_value']))]
    k = df[df.raw_value != df.manual_raw_value]

    print(" manual_raw_value == raw_value ", len(x), " out of ", len(df))
    print(" manual_raw_value == crnn_value ", len(y), " out of ", len(df))
    print("raw_value == crnn_value ", len(z), " out of ", len(df))
    print("raw_value == crnn_value==manual_raw_value ", len(w), " out of ", len(df))
    print("abby wrong and crnn correct  ", len(t), " out of ", len(df))
    print("abby wrong  ", len(k), " out of ", len(df))
    return x, y, z, w, t, k, len(df)

def eval_pred_alphabet(input):
    output_all_data=only_alphabet(input)
    df = pd.read_csv(output_all_data, sep=',')
    df = df.astype(str)
    df['manual_raw_value'] = df['manual_raw_value'].astype(str)
    df['raw_value'] = df['raw_value'].astype(str)
    df['crnn_pred'] = df['crnn_pred'].astype(str)
    x = df[df.manual_raw_value == df.raw_value]
    y = df[df.manual_raw_value == df.crnn_pred]
    z = df[df.crnn_pred == df.raw_value]
    w = df[((df.manual_raw_value == df.crnn_pred) & (df.manual_raw_value == df.raw_value))]
    t = df[((df['crnn_pred'] == df['manual_raw_value']) & (df['manual_raw_value'] != df['raw_value']))]
    k = df[df.raw_value != df.manual_raw_value]

    print(" manual_raw_value == raw_value ", len(x), " out of ", len(df))
    print(" manual_raw_value == crnn_value ", len(y), " out of ", len(df))
    print("raw_value == crnn_value ", len(z), " out of ", len(df))
    print("raw_value == crnn_value==manual_raw_value ", len(w), " out of ", len(df))
    print("abby wrong and crnn correct  ", len(t), " out of ", len(df))
    print("abby wrong  ", len(k), " out of ", len(df))
    return x, y, z, w, t, k, len(df)






if __name__ == '__main__':
    man_abby_all,man_crnn_all,abby_crnn_all,man_abby_crnn_all,wrongabby_crnn_correct_all,wrong_abby_all,numb_sequence_all=eval_pred_all_data(exported_file)
    man_abby_all_spec_char, man_crnn_all_spec_char, abby_crnn_all_spec_char, man_abby_crnn_all_spec_char, wrongabby_crnn_correct_all_spec_char, wrong_abby_all_spec_char, numb_sequence_spec_char=eval_pred_spec_char(exported_file)
    man_abby_digits, man_crnn_digits, abby_crnn_digits, man_abby_crnn_digits, wrongabby_crnn_correct_digits, wrong_abby_digits, numb_sequence_digits=eval_pred_digits(exported_file)
    man_abby_alphabet, man_crnn_alphabet, abby_crnn_alphabet, man_abby_crnn_alphabet, wrongabby_crnn_correct_alphabet, wrong_abby_alphabet, numb_sequence_alphabet=eval_pred_alphabet(exported_file)


#df['filtering'] = df['raw_value'].apply(lambda x : 1 if x.contains(['à', 'é']) else 0)