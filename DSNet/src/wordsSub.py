
import warnings
from konlpy.tag import *
import pandas as pd
warnings.simplefilter("ignore")

class NounSub:
  def __init__(self,sentence,tokenizer):
    self.tokenizer = tokenizer
    self.token_list = sentence
    self.final_result = []
    self.pos_table = ['NNG','NNP','NNB']
    self.sub_table = ['JKG','JKV','JKQ','JKB','XSV']
    self.stopwords = ['니다','남자','여자','사람','청년','집단','반사','새끼','복도',
                      '젊은이','마리','남성','여성','소년','소녀','아이템','남녀','단어']
  
  def __len__(self):
    return len(self.final_result)

  def window_func(self,left,seq):
    pos = left
    result = []
    while True:
      if pos >= len(seq): break 
      if seq[pos][1] in self.sub_table: 
        result = []
        break
      if seq[pos][1] not in self.pos_table: break
      elif seq[pos][1] in self.pos_table: result.append(seq[pos][0])
      pos += 1
    return pos,result

  def run(self):
    for x in self.token_list:
      words = self.tokenizer.pos(x)
      idx = 0
      output = []
      while True:
        if idx >= len(words): break
        if words[idx][1] in self.pos_table:
          idx,word_list = self.window_func(idx,words)
          if len(word_list) > 0 and len(word_list[-1]) >= 2:
            if word_list[-1] not in self.stopwords: 
              output.append(word_list[-1])
        else: idx += 1
      output = [''.join(word.split()) for word in output] 
      output = ' '.join(output)
      if len(output) != 0:
        self.final_result.append(output)

  def __getitem__(self,idx):
    return self.final_result[idx]