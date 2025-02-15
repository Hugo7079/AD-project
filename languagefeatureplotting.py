# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 09:34:55 2024

@author: Anna Cheng

Goal: combining language feature extraction and picture plotting
"""

# -*- coding: utf-8 -*-
"""「LInguistic feature preprocessing.ipynb」的副本

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yQ59Q0DD6feGxaTb77zIzcCUI3P_U6IQ
"""

#from google.colab import drive
#drive.mount('/content/drive')

"""# Installing Whisper

The commands below will install the Python packages needed to use Whisper models and evaluate the transcription results.
"""

#! pip install git+https://github.com/openai/whisper.git
#! pip install jiwer
#!pip install opencc-python-reimplemented

import os
import numpy as np
import torch
import pandas as pd
import whisper
import torchaudio
# import pydub
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""# Running inference on the dataset using a base Whisper model

The following will take a few minutes to transcribe all utterances in the dataset.
"""

"""# Silero"""

torch.set_num_threads(1)

from IPython.display import Audio
from pprint import pprint

USE_ONNX = False # change this to True if you want to test onnx model
# if USE_ONNX:
#     !pip install -q onnxruntime

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=USE_ONNX)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

# Function to get silence segments longer than 2 seconds
def detect_silence_s(file_path, item):
    vad_iterator = VADIterator(model)
    audio_file = file_path + "/" + item
    audio, sample_rate = torchaudio.load(audio_file)
    audio_np = audio[0].numpy()
    # Get the time axis for the waveform
    time = np.linspace(0, len(audio_np) / sample_rate, len(audio_np))
    silence_times = list()
    sample_rate = 16000
    window_size_samples = 512
    wav = read_audio(audio_file, sampling_rate=sample_rate)
    # get speech timestamps from full audio file
    speech_timestamps = get_speech_timestamps(
        wav, model, return_seconds=True, visualize_probs=True,
        window_size_samples=window_size_samples, threshold=0.4,
        min_speech_duration_ms=200, sampling_rate=sample_rate
    )
    vad_iterator.reset_states()  # reset model states after each audio
    pprint(speech_timestamps)
    plt.figure(figsize=(25, 5))
    plt.plot(time, audio_np)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Waveform with Silence Periods')
    k = 0
    for timestamp in speech_timestamps:
        if k == 1:
            start_time = end_time
            end_time = timestamp['start']
        else:
            start_time = 0
            end_time = timestamp['start']
            k = 1
        if end_time - start_time >= 2:
            silence_times.append((start_time, end_time))
            plt.axvspan(start_time, end_time, facecolor='red', alpha=0.3)
        end_time = timestamp['end']
    print(len(silence_times))
    plt.grid()
    # plt.show()

"""# ckip 程式架構"""
# wav file 所在的地方
wav_file = "audio_files"
# Whisper model size
model_size = "medium"
model_wh = whisper.load_model(model_size)
print(
    f"Model is {'multilingual' if model_wh.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model_wh.parameters()):,} parameters."
)
#中研院分析詞頻資料
SD_database_path  = "WF_CD_SD_20210315_add_add_logCD.xlsx"

from datetime import datetime
current_dateTime = datetime.now()
date_time = str(current_dateTime.month)+'-'+str(current_dateTime.day)+'-'+str(current_dateTime.hour)+'-'+str(current_dateTime.minute)

def load_ckip_models():
    ws_driver = CkipWordSegmenter(model="bert-base")
    pos_driver = CkipPosTagger(model="bert-base")
    return ws_driver, pos_driver

#%% Settings
config = {
    #ckip model的路徑
    "model_path":"ckip",
    # wav file 所在的地方
    "wav_file": "audio_files",
    #待分析文件所在的資料夾的路徑
    "txt_file_folder_path": "audio_files/word",
    #輸出的檔案名稱
    "output_file_name": "audio_files/result_"+date_time+".xlsx",
    #輸入待分析的資料
    "for_analysis_data": "audio_files/result_"+date_time+".xlsx",
    #輸出待分析的資料
    "analysis_file_name": "audio_files/ana_"+date_time+".xlsx",
    #中研院分析詞頻資料
    "SD_database_path": SD_database_path,
    # remove unwanted elements
    "unwanted": {'、','。','，','／'},
    # grouping rule
    "group_rule" : {
        'N':['Na','Nf','Nc'],
        'V':['VA','VAC','VB','VC','VCL','VD','VE','VF','VG','VH','VHC','VI','VJ','VK','VL','SHI','V_2'],
        'ADV':['D','Df'],
        'ADJ':['A'],
        'Nb':['Nb','Ncd'],
        "punctuation" : [
            'COLONCATEGORY','COMMACATEGORY','DASHCATEGORY','DOTCATEGORY','ETCCATEGORY',
            'EXCLAMATIONCATEGORY','PARENTHESISCATEGORY','PAUSECATEGORY','PERIODCATEGORY',
            'QUESTIONCATEGORY','SEMICOLONCATEGORY','SPCHANGECATEGORY','SLASHCATEGORY','WHITESPACE'
        ],
        "content_word" : [
            'Na','Nf','Nc','VA','VAC','VB','VC','VCL','VD','VE','VF','VG','VH','VHC','VI','VJ','VK','VL',
            'D','Df','A','Nb','Ncd'
        ]
    },
    # pos count list
    "pos_count_list": {
        "否定詞數":{"D":["不","沒","沒有","不是","不可以","不能","不行","不可能"]},
        "Pronoun":["Nh"],
        "content_word":[
            'Na','Nf','Nc','VA','VAC','VB','VC','VCL','VD','VE','VF','VG','VH','VHC','VI','VJ','VK','VL',
            'D','Df','A','Nb','Ncd'
        ],
        'verb':['VA','VAC','VB','VC','VCL','VD','VE','VF','VG','VH','VHC','VI','VJ','VK','VL','SHI','V_2'],
        "被動句":{"P":["被","受到"]},
        "把字句":{"P":["把","將"]},
        "long_pause":["全形間隔號"],
        "short_pause":["PAUSECATEGORY"],
        "filler":["filler"]
    },
    "statistics_list": [
        "total_char", "total_word", "difficult_word","unique_word",
        "total_sentence_1", "total_sentence_2", "total_sentence_3",
        "total_sentence_4","total_sentence_5", "total_sentence_6", "total_sentence_7",
        "repeat_word",
        "avg_w_s_1", "avg_w_s_2", "avg_w_s_3", 'avg_w_s_4', 'avg_w_s_5', 'avg_w_s_6', 'avg_w_s_7',
        "content_word", "connect_word_pair", "connect_word", "被動型",'passive',
        '否定詞數', '代名詞', '實詞', '動詞', '連接詞', '正向連接詞', '負向連接詞', '成對連接詞',
        '被動句', '把字句', '比較', '連', '使役動詞',
        "長停頓", "短停頓", "filler",
        "avg_cwf_log(WF_ppm)", "avg_twf_log(WF_ppm)",
        "avg_cwf_WF_ppm", "avg_twf_WF_ppm",
        "tw_distr:", "cw_distr:"
    ]
}

using_config = {
    "Pronoun":["Nh"],
    'Verb':['VA','VAC','VB','VC','VCL','VD','VE','VF','VG','VH','VHC','VI','VJ','VK','VL','SHI','V_2'],
    "long_pause":["PAUSECATEGORY"],
    "filler":["filler"]
}

"""#Main part"""

def audio_to_text(path , audi):
    time_list = list()
    wav_path = path + "/" + audi
    result = model_wh.transcribe(wav_path, language="zh", beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4), no_speech_threshold=0.5)
    last_sentence = ""
    for item in result['segments']:
        if item['text'] == last_sentence:
            continue
        else:
            time_list.append([item['start'], item['end'], str(item['text'])])
            last_sentence = item['text']
    return time_list

import re
import glob
from opencc import OpenCC
tryon = 0
cc = OpenCC('s2t')
file_path = config["wav_file"]
files =  [i for i in os.listdir(file_path) if i[-4:]=='.wav']

for item in files:
    if os.path.exists(file_path+"/word_add_time/"+item[0:7]+'_time.txt'):
        continue
    else:
        string = str()
        print(item[0:7])
        time_list = audio_to_text(file_path, item)
        silence_segments = detect_silence_s(file_path, item)  # 無實際 return, 只是繪圖
        time_list = sorted(time_list, key=lambda x: x[0])
        if not os.path.isdir(file_path+"/word_add_time"):
            os.makedirs(file_path+"/word_add_time")
        with open (file_path+"/word_add_time/"+item[0:7]+'_time.txt','w',encoding='UTF-8')as f:
            for i in time_list:
                f.write(str(i[0])+" "+ str(i[1]) +" "+str(i[2])+'\n')
        for k in range(len(time_list)):
            if k+1 == len(time_list):
                string = string + time_list[k][2] + "。"
            elif time_list[k+1][2]=='、' or time_list[k][2]=='、':
                string = string + time_list[k][2]
            else:
                string = string + time_list[k][2] + "，"
        string = string.replace("?","，")
        string = string.replace(" ","，")
        string = re.sub(",","，", string, flags=re.IGNORECASE)
        string = re.sub("，?。?、+。?，?","、", string, flags=re.IGNORECASE)
        string = re.sub("，?。+，?","。", string, flags=re.IGNORECASE)
        if string[0] =="、":
            string = string[1:]
        string = cc.convert(string)
        if not os.path.isdir(file_path+"/word"):
            os.makedirs(file_path+"/word")
        with open (file_path+"/word/"+item[0:7]+'.txt','w',encoding='UTF-8')as f:
            f.write(string)

"""#安裝程式
ckiptagger - 分詞用
gdown - 下載預訓練模型
tensorflow - 模型框架
xlsxwriter - 寫入excel
"""

#!pip3 install tensorflow
#!pip3 install gdown
#!pip install xlsxwriter

"""下載預訓練模型
"""

from collections import OrderedDict
import time
import math
import xlsxwriter
from collections import Counter
#!pip install -U ckip-transformers
from ckip_transformers import __version__
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker

ws_driver = CkipWordSegmenter(model="bert-base")
pos_driver = CkipPosTagger(model="bert-base")
ner_driver = CkipNerChunker(model="bert-base")

"""# Function
計算程式需時
"""

def time_recorder(func):
    def new_function(*args, **kwargs):
        t = time.time()
        result = func(*args, **kwargs)
        print(func.__name__,"completed.","It took",round(time.time()-t,2),"s")
        return result
    return new_function

"""下載分詞模型"""
@time_recorder
def load_ckip_model():
    parser = {
        "ws": ws_driver,
        "pos": pos_driver,
        "ner": ner_driver
    }
    return parser

parser = load_ckip_model()

"""自動修正分詞字典"""
Auto_check_database = {
    'ws':{
        '小孩子':'小 孩子',
        '小女孩':'小 女孩',
        '盪鞦韆':'盪 鞦韆',
        '撈魚':'撈 魚',
        '喝水':'喝 水',
        '喝茶':'喝 茶'
    },
    'pos':{
        'pos_list':{
            '小 孩子':'A Na',
            '小 男孩':'A Na',
            '小 女孩':'A Na'
        }
    }
}
for item in Auto_check_database['ws'].keys():
    try:
        Auto_check_database['ws'][item] = Auto_check_database['ws'][item].split()
    except:
        pass
for item in Auto_check_database['pos']['pos_list'].keys():
    try:
        Auto_check_database['pos']['pos_list'][item] = Auto_check_database['pos']['pos_list'][item].split()
    except:
        pass

"""下載中研院詞頻excel檔"""
@time_recorder
def load_database(SD_database_path):
    db = pd.read_excel(SD_database_path, sheet_name='database')
    db.set_index("word", inplace=True)
    db = db[~db.index.duplicated()]
    database = db[["WF", "WF_ppm", "log(WF_ppm)"]].copy()
    database["Rank"] = database["WF"].rank(method='min', ascending=False)
    database["PR"] = database["WF"].rank(pct=True)
    for i in database.index:
        database.loc[i,"PR_rank_6"] = math.ceil(database.loc[i,"PR"]*100/17)
        database.loc[i,"PR_rank_8"] = math.ceil(database.loc[i,"PR"]*100/12.5)
        database.loc[i,"PR_rank_10"] = math.ceil(database.loc[i,"PR"]*100/10)
    database = database.sort_values("WF", ascending=False)
    return database

"""讀取文字檔"""
@time_recorder
def read_files(file_path):
    documents = {}
    for path, directory, files in os.walk(file_path):
        for file_name in files:
            if file_name.endswith(".txt"):
                pass
            else:
                continue
            if file_name[:-4] in documents.keys():
                print("There's a duplicate file:"+file_name)
            documents[file_name[:-4]] = {}
            documents[file_name[:-4]]["path"] = path
            with open(path+"/"+file_name, 'r', encoding='utf-8-sig') as f:
                raw_sentence_list = f.read().strip().split('\n')
            for i in range(len(raw_sentence_list)):
                raw_sentence_list[i] = raw_sentence_list[i].replace(" ","，")
            documents[file_name[:-4]]["sentence_list"] = raw_sentence_list
    return documents

"""## 進行分詞"""
@time_recorder
def parsing_WS(parser, documents, Auto_check=True):
    def flatten(lst):
        for sublist in lst:
            if isinstance(sublist, list):
                for item in sublist:
                    yield item
            else:
                yield sublist

    for document in documents.keys():
        documents[document]["word_sentence_list"] = parser["ws"](documents[document]["sentence_list"],)
        documents[document]["wa_list"] = [word for sentence in documents[document]["word_sentence_list"] for word in sentence]

        if Auto_check:
            for i in range(len(documents[document]["wa_list"])):
                if i+2 < len(documents[document]["wa_list"]):
                    if documents[document]['wa_list'][i]=='這' and documents[document]['wa_list'][i+1]=='個' and documents[document]['wa_list'][i+2]=='，':
                        documents[document]['wa_list'][i] = '這個'
                        documents[document]['wa_list'][i+1] = '，'
                        documents[document]['wa_list'][i+2] = ''
                    if documents[document]['wa_list'][i]=='那' and documents[document]['wa_list'][i+1]=='個' and documents[document]['wa_list'][i+2]=='，':
                        documents[document]['wa_list'][i] = '那個'
                        documents[document]['wa_list'][i+1] = '，'
                        documents[document]['wa_list'][i+2] = ''
        flatten_list = flatten(documents[document]["wa_list"])
        documents[document]["wa_list"] = [value for value in flatten_list if value != '']
        documents[document]["dataframe"] = pd.DataFrame(documents[document]["wa_list"], columns=["word"])
        documents[document]["word_sentence_list"] = [documents[document]["wa_list"].copy()]

    return documents

"""進行詞性標註 兩欄:word pos"""
@time_recorder
def parsing_POS(parser, documents, revised=False, POS_replace=False, Auto_check=False):
    if revised:
        for document in documents.keys():
            if document in ["count"]:
                continue
            documents[document]["pos_sentence_list"] = parser["pos"](documents[document]["word_sentence_list"])
            documents[document]["wa_list"] = [word for sentence in documents[document]["word_sentence_list"] for word in sentence]
            documents[document]["pa_list"] = [pos for sentence in documents[document]["pos_sentence_list"] for pos in sentence]
            documents[document]["dataframe"]["re_POS"] = documents[document]["pa_list"]
            if POS_replace:
                documents[document]["dataframe"]["POS"] = documents[document]["dataframe"]["re_POS"]
        return documents

    for document in documents.keys():
        documents[document]["pos_sentence_list"] = parser["pos"](documents[document]["word_sentence_list"])
        documents[document]["wa_list"] = [word for sentence in documents[document]["word_sentence_list"] for word in sentence]
        documents[document]["pa_list"] = [pos for sentence in documents[document]["pos_sentence_list"] for pos in sentence]
        documents[document]["dataframe"]["POS"] = documents[document]["pa_list"]

        if Auto_check:
            for i in range(len(documents[document]["pa_list"])):
                if documents[document]["wa_list"][i] in ['然後','這個','齁','那個','對','這樣子','喔','對不對','嗯','阿這個']:
                    documents[document]["pa_list"][i] = 'filler'
            documents[document]["dataframe"]["POS"] = documents[document]["pa_list"]

    return documents

"""讀取修正後的excel檔"""
@time_recorder
def read_revised_documents(file_path):
    revised_documents = pd.read_excel(file_path, sheet_name=None)
    documents = {}
    for file_name in revised_documents.keys():
        if file_name in ["config"]:
            continue
        if file_name in ["count"]:
            documents["count"] = revised_documents["count"]
            continue
        if file_name in documents.keys():
            print("There's a duplicate file:"+file_name)
        documents[file_name] = {}
        documents[file_name]["article"] = []
        documents[file_name]["path"] = ""
        documents[file_name]["dataframe"] = revised_documents[file_name][["word", "POS"]].copy()
        documents[file_name]["word_sentence_list"] = [[word for word in documents[file_name]["dataframe"]["word"]]]
    print("Parsed documents read completed")
    return documents

"""計算各種詞類數目"""
@time_recorder
def pos_count(documents, count_list):
    for document in documents.keys():
        if document in ["count"]:
            continue
        for count in count_list.keys():
            if count in ["Pronoun","content_word","verb","long_pause", "short_pause", "filler"]:
                documents[document]["dataframe"][count] = 0
                for i in range(documents[document]["dataframe"].shape[0]):
                    if documents[document]["dataframe"].loc[i,"POS"] in count_list[count]:
                        documents[document]["dataframe"].loc[i,count] = 1
            else:
                documents[document]["dataframe"][count] = 0
                for i in range(documents[document]["dataframe"].shape[0]):
                    if documents[document]["dataframe"].loc[i,"POS"] in count_list[count].keys():
                        if documents[document]["dataframe"].loc[i,"word"] in count_list[count][documents[document]["dataframe"].loc[i,"POS"]]:
                            documents[document]["dataframe"].loc[i,count] += 1
    return documents

"""將中研院詞頻資料和dataframe結合"""
@time_recorder
def refer_to_database(documents, database):
    for document in documents.keys():
        if document in ["count"]:
            continue
        documents[document]["dataframe"] = pd.merge(
            left=documents[document]["dataframe"],
            right=database,
            how="left",
            on="word"
        )
    return documents

"""## 產出統計 primary 資料"""
@time_recorder
def statistics(documents, statistics_list, group_rule, pos_count, revised=False):
    for document in documents.keys():
        if document in ["count"]:
            continue
        count_class = Counter(documents[document]['dataframe']['POS'])
        period = Counter(documents[document]['dataframe']['word'])['、']
        documents[document]["statistics"] = OrderedDict()

        # using_config計算
        for i in using_config.keys():
            add = 0
            for item in using_config[i]:
                add += count_class[item]
            documents[document]["statistics"][i] = add

        documents[document]["statistics"]['long_pause'] = period

        for i in statistics_list:
            if i == "total_char":
                total_char = 0
                for word in documents[document]["dataframe"][
                    (~documents[document]["dataframe"]["POS"].isin(group_rule["punctuation"])) &
                    (~documents[document]["dataframe"]["word"].isnull())
                ]["word"]:
                    total_char += len(word)
                documents[document]["statistics"][i] = total_char

            elif i == "total_word":
                documents[document]["statistics"][i] = documents[document]["dataframe"][
                    ~documents[document]["dataframe"]["POS"].isin(group_rule["punctuation"])
                ].shape[0]

            elif i == "unique_word":
                documents[document]["statistics"][i] = documents[document]["dataframe"][
                    ~documents[document]["dataframe"]["POS"].isin(group_rule["punctuation"])
                ].drop_duplicates(subset=['word']).shape[0]

            elif i == "total_sentence_1":
                documents[document]["statistics"][i] = documents[document]["dataframe"][
                    documents[document]["dataframe"]["POS"].isin(group_rule["punctuation"])
                ].shape[0]

            elif i == "total_sentence_7":
                total_s = 0
                state = 0
                for j in range(documents[document]["dataframe"].shape[0]):
                    if state == 0:
                        if documents[document]["dataframe"].loc[j,"POS"] in group_rule["N"]:
                            state = 1
                        if documents[document]["dataframe"].loc[j,"POS"] in group_rule["V"]:
                            state = 2
                    elif state == 1:
                        if documents[document]["dataframe"].loc[j,"POS"] in group_rule["V"]:
                            total_s += 1
                            state = 3
                    elif state == 2:
                        if documents[document]["dataframe"].loc[j,"POS"] in group_rule["N"]:
                            total_s += 1
                            state = 4
                    if documents[document]["dataframe"].loc[j,"POS"] in group_rule["punctuation"]:
                        state = 0
                documents[document]["statistics"][i] = total_s

            elif i == "passive":
                documents[document]["statistics"][i] = sum([x for x in documents[document]["dataframe"]["被動句"]]) + \
                                                        sum([x for x in documents[document]["dataframe"]["把字句"]])

            elif i == "content_word":
                documents[document]["statistics"][i] = documents[document]["dataframe"][
                    documents[document]["dataframe"]["POS"].isin(group_rule["content_word"])
                ].shape[0]

            elif i == "avg_cwf_WF_ppm":
                cw = documents[document]["dataframe"][
                    (documents[document]["dataframe"]["POS"].isin(group_rule["content_word"])) &
                    (~documents[document]["dataframe"]["WF_ppm"].isnull())
                ]
                sum_WF = sum([x for x in cw["WF_ppm"]])
                num_cw = cw.shape[0]
                if num_cw == 0:
                    documents[document]["statistics"][i] = 0
                else:
                    documents[document]["statistics"][i] = sum_WF/num_cw

            elif i == "avg_twf_WF_ppm":
                tw = documents[document]["dataframe"][
                    (~documents[document]["dataframe"]["POS"].isin(group_rule["punctuation"])) &
                    (~documents[document]["dataframe"]["WF_ppm"].isnull())
                ]
                sum_WF = sum([x for x in tw["WF_ppm"]])
                num_tw = tw.shape[0]
                if num_tw == 0:
                    documents[document]["statistics"][i] = 0
                else:
                    documents[document]["statistics"][i] = sum_WF/num_tw

    return documents

"""寫入excel檔，only for primary data"""
@time_recorder
def write_data(documents , file_path, revised=False):
    stat_df = pd.DataFrame()
    if revised:
        file_path = file_path[:-5] + '_revised' + '.xlsx'
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        for article_name in documents.keys():
            if revised == False:
                data = documents[article_name]['dataframe']
                if 'statistics' in documents[article_name].keys():
                    stat = documents[article_name]['statistics']
                else:
                    stat = {}
                stat['file'] = article_name
                data.to_excel(writer, sheet_name=article_name)
                stat_df = stat_df.append(stat, ignore_index=True)
                columns_name = stat_df.columns.tolist()
                columns_name = columns_name[-1:] + columns_name[:-1]
                stat_df = stat_df.reindex(columns=columns_name)
                stat_df.to_excel(writer, sheet_name='count', index=False)
            else:
                if article_name == 'count':
                    continue
                else:
                    data = documents[article_name]['dataframe']
                    stat = documents[article_name]['statistics']
                    stat['file'] = article_name
                    data.to_excel(writer, sheet_name=article_name)
                    stat_df = stat_df.append(stat, ignore_index=True)
                    stat_df.to_excel(writer, sheet_name='count', index=False)
    return stat_df

# 開始執行程式
database = load_database(config['SD_database_path'])
documents = read_files(config["txt_file_folder_path"])
documents = parsing_WS(parser,documents,Auto_check=True)
documents = parsing_POS(parser,documents,Auto_check=True)
documents = pos_count(documents, config["pos_count_list"])
documents = refer_to_database(documents, database)
documents = statistics(documents, config["statistics_list"], config["group_rule"], config["pos_count_list"])
stat_df = write_data(documents, config["output_file_name"])

"""## 讀取primary資料，產出secondary資料
"""

stat_df = pd.read_excel(config["for_analysis_data"], sheet_name='count')
stat_df = stat_df.reset_index()

import pandas
@time_recorder
def pd_nan(stat_df,file_path):
    ana_df = pd.DataFrame()
    ana_df['file'] = pd.DataFrame(stat_df['file'])
    ana_df['total_word'] = stat_df['total_word']
    ana_df['unique_word'] = stat_df['unique_word']
    ana_df['TTR'] = stat_df['unique_word'] / stat_df['total_word']
    ana_df['content_word'] = stat_df['content_word']
    ana_df['frequency'] = stat_df['avg_twf_WF_ppm']
    ana_df['frequency_c'] = stat_df['avg_cwf_WF_ppm']
    ana_df['U'] = stat_df['total_sentence_1']
    ana_df['S'] = stat_df['total_sentence_7']
    ana_df['MLU'] = stat_df['total_char'] / ana_df['U']
    ana_df['MLS'] = stat_df['total_char'] / ana_df['S']
    ana_df['filler_ratio'] = stat_df['filler'] / ana_df['total_word']
    ana_df['long_pauses_ratio'] = stat_df['long_pause'] / ana_df['total_word']
    ana_df['content_density'] = stat_df['content_word'] / ana_df['total_word']
    ana_df['passive_ratio'] = stat_df['passive'] / ana_df['U']
    ana_df['verb_ratio'] = stat_df['Verb'] / ana_df['total_word']
    ana_df['pronoun_ratio'] = stat_df['Pronoun'] / ana_df['total_word']
    ana_df = round(ana_df,4)
    print(ana_df.head(10))
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        ana_df.to_excel(writer)
    return ana_df

analysis_df = pd_nan(stat_df, config["analysis_file_name"])

"""## 以下為畫特徵雷達圖的程式碼
畫兩組資料：固定背景數據 & 實際資料 於同一張雷達圖
"""
import math
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as io

io.renderers.default='browser'

class ComplexRadar():
    """
    Create a complex radar chart with different scales for each variable
    Parameters
    ----------
    fig : figure object
        A matplotlib figure object to add the axes on
    variables : list
        A list of variables
    ranges : list
        A list of tuples (min, max) for each variable
    n_ring_levels: int, defaults to 5
        Number of ordinate or ring levels to draw
    show_scales: bool, defaults to True
        Indicates if the ranges for each variable are plotted
    """
    def __init__(self, fig, variables, ranges, n_ring_levels=5, show_scales=True):
        angles = np.arange(0, 360, 360./len(variables))
        axes = [fig.add_axes([0.1,0.1,0.9,0.9], polar=True, label = "axes{}".format(i)) for i in range(len(variables)+1)]
        
        # Ensure clockwise rotation
        for ax in axes:
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_axisbelow(True)
        
        # Setting the ranges
        for i, ax in enumerate(axes):
            j = 0 if (i==0 or i==1) else i-1
            ax.set_ylim(*ranges[j])
            grid = np.linspace(*ranges[j], num=n_ring_levels, endpoint=False)
            gridlabel = ["{}".format(round(x,2)) for x in grid]
            gridlabel[0] = "" 
            lines, labels = ax.set_rgrids(grid, labels=gridlabel, angle=angles[j])
            ax.set_ylim(*ranges[j])
            ax.spines["polar"].set_visible(False)
            ax.grid(visible=False)
            if show_scales == False:
                ax.set_yticklabels([])

        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.xaxis.set_visible(False)
            
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
        self.ax1 = axes[1]
        self.plot_counter = 0
        
        # Draw the lines
        self.ax.yaxis.grid()
        self.ax.xaxis.grid()

        # Outer circle
        self.ax.spines['polar'].set_visible(True)
        
        self.ax1.axis('off')
        self.ax1.set_zorder(9)
        
        l, text_ = self.ax.set_thetagrids(angles, labels=variables)
        labels_ = [t.get_text() for t in self.ax.get_xticklabels()]
        labels_ = ['\n'.join(textwrap.wrap(lab, 15, break_long_words=False)) for lab in labels_]
        self.ax.set_xticklabels(labels_)

        for t,a in zip(self.ax.get_xticklabels(),angles):
            if a == 0:
                t.set_ha('center')
            elif a > 0 and a < 180:
                t.set_ha('left')
            elif a == 180:
                t.set_ha('center')
            else:
                t.set_ha('right')
        self.ax.tick_params(axis='both', pad=15)

    def _scale_data(self, data, ranges):
        """Scales data[1:] to ranges[0]"""
        sdata = []
        for d, (y1, y2) in zip(data[1:], ranges[1:]):
            if not ((y1 <= d <= y2) or (y2 <= d <= y1)):
                print(f"Clipping value {d} to range ({y1}, {y2})")
                d = max(min(d, max(y1, y2)), min(y1, y2))
            sdata.append((d - y1) / (y2 - y1) * (ranges[0][1] - ranges[0][0]) + ranges[0][0])
        # Handle data[0] separately
        sdata = [data[0]] + sdata
        return sdata

    def plot(self, data, *args, **kwargs):
        sdata = self._scale_data(data, self.ranges)
        self.ax1.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kwargs)
        self.plot_counter += 1
    
    def fill(self, data, *args, **kwargs):
        sdata = self._scale_data(data, self.ranges)
        self.ax1.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kwargs)
        
    def use_legend(self, *args, **kwargs):
        self.ax1.legend(*args, **kwargs)
    
    def set_title(self, title, pad=25, **kwargs):
        self.ax.set_title(title, pad=pad, **kwargs)

# 從analysis_file_name中載入 secondary feature 資料
df = pd.read_excel(config["analysis_file_name"])
df_reduced = df.drop(['Unnamed: 0','file','frequency'], axis=1)

# 新增 average, max, min 三行
df_reduced.loc['average'] = df_reduced.mean()
df_reduced.loc['max'] = df_reduced.max()
df_reduced.loc['min'] = df_reduced.min()

# 取得 min, max range
zipped_minmax = zip(df_reduced.loc['min'], df_reduced.loc['max'])

# 固定背景數據
fixed_data = {
    'total_word': 760.25,
    'unique_word': 282,
    'TTR': 0.390875,
    'content_word': 303.75,
    'frequency_c': 668.4344,
    'U': 135,
    'S': 74.25,
    'MLU': 8.323875,
    'MLS': 15.13563,
    'filler_ratio': 0.01345,
    'long_pauses_ratio': 0.0024,
    'content_density': 0.407375,
    'passive_ratio': 0.025475,
    'verb_ratio': 0.260775,
    'pronoun_ratio': 0.076875
}

# Radar圖上對應的變數順序（自行調整名稱長度）
variables = ('TW','UW','TTR','CW','CWF','NU','NS','MLU','MLS','FR','LPR','CD','PaR','VR','PrR')

# 依照上面變數順序，組出fixed_data
data_fixed = (
    fixed_data['total_word'],       # TW
    fixed_data['unique_word'],      # UW
    fixed_data['TTR'],              # TTR
    fixed_data['content_word'],     # CW
    fixed_data['frequency_c'],      # CWF
    fixed_data['U'],                # NU
    fixed_data['S'],                # NS
    fixed_data['MLU'],              # MLU
    fixed_data['MLS'],              # MLS
    fixed_data['filler_ratio'],     # FR
    fixed_data['long_pauses_ratio'],# LPR
    fixed_data['content_density'],  # CD
    fixed_data['passive_ratio'],    # PaR
    fixed_data['verb_ratio'],       # VR
    fixed_data['pronoun_ratio']     # PrR
)

def describe_chart(stats):
    text = (
        f"**語言特徵說明：**\n\n"
        f"- 總詞數 (TW) = {stats['TW']}\n"
        f"- 獨特詞數 (UW) = {stats['UW']}\n"
        f"- 詞彙多樣性 (TTR) = {stats['TTR']}\n"
        f"- 內容詞數 (CW) = {stats['CW']}\n"
        f"- 內容詞平均詞頻 (CWF) = {stats['CWF']}\n"
        f"- 標點句數 (U) = {stats['NU']}\n"
        f"- N+V或V+N結構數 (S) = {stats['NS']}\n"
        f"- 平均語句長度 (MLU) = {stats['MLU']}\n"
        f"- 平均句長 (MLS) = {stats['MLS']}\n"
        f"- 填充詞比例 (FR) = {stats['FR']}\n"
        f"- 長停頓比例 (LPR) = {stats['LPR']}\n"
        f"- 內容密度 (CD) = {stats['CD']}\n"
        f"- 被動句比例 (PaR) = {stats['PaR']}\n"
        f"- 動詞使用比例 (VR) = {stats['VR']}\n"
        f"- 代名詞使用比例 (PrR) = {stats['PrR']}\n"
    )
    return text
    
# 實際資料(這裡使用 df_reduced loc 'average'當示範)
data_ad = tuple(df_reduced.loc['average'])
ranges = list(zipped_minmax)

# 建立雷達圖
# adjusted_ranges = [(min(y1, min(data_fixed[i], data_ad[i])), 
#                     max(y2, max(data_fixed[i], data_ad[i]))) 
#                    for i, (y1, y2) in enumerate(ranges)]

adjusted_ranges = [
    (345,1725),    # TW
    (163,546),     # UW
    (0.2958,0.4903), # TTR
    (176,748),     # CW
    (499.4797,934.8998), # CWF
    (98,355),      # NU
    (39,195),      # NS
    (5, 11),       # MLU - Adjusted
    (12.2105,18.1282), # MLS
    (0,0.0394),    # FR
    (0,0.0145),    # LPR
    (0.3733,0.513),# CD
    (0,0.0714),    # PaR
    (0.2251,0.3159),# VR
    (0.0316,0.1216) # PrR
]

# 更新範圍並打印
print("Adjusted ranges:", adjusted_ranges)

# 繪製雷達圖
fig1 = plt.figure(figsize=(4, 4))
radar = ComplexRadar(fig1, variables, adjusted_ranges, show_scales=True)
radar.set_title("Linguistic Features Radar Chart", pad=40)

# 繪製內容
radar.plot(data_fixed, label='Fixed Background', color='blue', linestyle='--')
radar.fill(data_fixed, alpha=0.3, color='blue')
radar.plot(data_ad, label='AD Data', color='green')
radar.fill(data_ad, alpha=0.3, color='green')

# 替換圖例設置
plt.legend(['Fixed Background', 'AD Data'], loc='upper right', bbox_to_anchor=(1.3, 1.2))

# 保存圖片
plt.savefig('radar_chart_complete.png', bbox_inches='tight', dpi=300)

plt.show()

# 將 data_ad 轉換為 stats 字典
stats = dict(zip(variables, data_ad))

# 呼叫 describe_chart 並印出結果
description = describe_chart(stats)
print(description)

