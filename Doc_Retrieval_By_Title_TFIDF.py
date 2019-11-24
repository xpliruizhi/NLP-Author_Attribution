import datetime
import zipfile
import json
import collections
import gc
import pickle
import pymysql

import pdb
import heapq
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords 
import nltk
from nltk.tokenize import wordpunct_tokenize

from collections import Counter
ISOTIMEFORMAT = '%H:%M:%S'
#nltk.download('stopwords')
stop_words = set(stopwords.words('english')) 
def Stemmer_text(Sentence,stop_words):
    ps = PorterStemmer()
    words = wordpunct_tokenize(Sentence)
    new_sentence = ""
    for word in words:
        if word in stop_words:
            continue
        else:
            if word.isalpha() or word.isalnum():
                new_sentence = new_sentence+" "+ps.stem(word)
    return new_sentence
    
def Read_from_WIKI():
    my_docs_key = {}
    print("begin to read wiki files,time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    myzip = zipfile.ZipFile('wiki-pages-text.zip')
    ziplist = myzip.namelist()
    ## loop each txt file
    for i in range(1,len(ziplist)):
        myzip = zipfile.ZipFile('wiki-pages-text.zip')
        fileobj = myzip.open(ziplist[i])
    #    print("start read "+str(i)+"file:"+ ziplist[i])
    #    loop each line from txt
        j = 0
        key_count = 0
        previous_key = ""
        for line in fileobj:
            # must use utf-8 to store different languages
            # remove "/n" at each line 
            myline = line.decode('utf-8').strip()
            # use first 2 blanks to cut the string
            # j += 1
            # if j>90:
                # break
            
            line_list = myline.split(' ',2)
            #temp = line_list[0]+'$'+line_list[1]
            #mydocument[temp] = line_list[2]
            if line_list[0] == "Hispanic_and_Latino_Americans" or line_list[0] == "Carrie_Mathison":
                print(line_list)
            if previous_key != line_list[0]:
                previous_key = line_list[0]
                key = line_list[0].replace("_"," ")
                key = key.replace("-LRB-","(")
                key = key.replace("-RRB-",")")
                
                ### Using Stemmer###
                #key = Stemmer_text(key,stop_words)
                ### Using Stemmer###
                if key not in my_docs_key.keys():
                    #key_count = key_count + 1
                    my_docs_key[key]= line_list[0]
            else:
                continue
        #break
        myzip.close()
    print("File Reading Finish,time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    return my_docs_key

def Doc_retrival_invert_list_by_title(my_docs_key):
########################################################################################
    print("Start building TITLE TFIDF matrix......"+datetime.datetime.now().strftime(ISOTIMEFORMAT))
    my_document_keys = list(my_docs_key.keys())
    vectorizer = TfidfVectorizer(stop_words='english')
    my_response = vectorizer.fit_transform(my_document_keys)
    word_list = vectorizer.get_feature_names()
    print("TITLE TFIDF matrix finished......"+datetime.datetime.now().strftime(ISOTIMEFORMAT))
    t_len =  len(my_response.nonzero()[1])
    invert_index_dict = defaultdict(Counter)
    N = 0
    for col in zip(my_response.nonzero()[1],my_response.nonzero()[0]):
        invert_index_dict[word_list[col[0]]][col[1]] = float(my_response[col[1],col[0]])
        N = N+1
        if N%100000 == 0:
            print("Time:"+datetime.datetime.now().strftime(ISOTIMEFORMAT)+", "+str(N)+'data processed'+' total:'+ str(t_len))
    print("Inverted index finished"+datetime.datetime.now().strftime(ISOTIMEFORMAT))
    return invert_index_dict, my_document_keys
    
def tokenizer_claim_by_sklearn(claim):
    #my_c = claim.split(' ')
    count_vect = CountVectorizer(stop_words='english')
    #count_vect = CountVectorizer()
    
    ###Stemmer used
    #claim = Stemmer_text(claim,stop_words)
    ###Stemmer used
    
    count_vect.fit_transform([claim])

    #pdb.set_trace()
    return count_vect.get_feature_names()

def doc_by_tf_idf(query_list,invert_index_dict, document_keys):
#    pdb.set_trace()
    word_index_counter = Counter()
    for word in query_list:
        if word in invert_index_dict.keys():    
            word_index_counter = word_index_counter + invert_index_dict[word]
    #print(word_index_counter)
    doc_ranking = word_index_counter.most_common(5)
    rank_list = []
    for element in doc_ranking:
        TITLE = my_docs_key[document_keys[element[0]]]
        
        ##### 减少内存消耗###在没有Stemmer的时候可行
        # TITLE = document_keys[element[0]]
        # TITLE = TITLE.replace(")","-RRB-")
        # TITLE = TITLE.replace("(","-LRB-")
        # TITLE = TITLE.replace(" ","_")
        
        rank_list.append([TITLE,1])
    return rank_list  

my_docs_key = Read_from_WIKI()
invert_index_dict, my_document_keys = Doc_retrival_invert_list_by_title(my_docs_key)

#print(my_docs_key)

# del my_docs_key
# gc.collect()

# q_list = tokenizer_claim_by_sklearn('is American Thighs bullshit')
# print(q_list)
# rank_list = doc_by_tf_idf(q_list,invert_index_dict, my_document_keys)
# print(rank_list)

with open('devset.json', 'r') as f:
    print('read json file starting'+datetime.datetime.now().strftime(ISOTIMEFORMAT))
    data = json.load(f)
    total_len = len(data)
    print('total'+str(total_len)+'data')
    i=0
    with open("record_title.json","w") as dg:
        for json_key in data.keys():
            myclaim = data[json_key]['claim']
            query_list = tokenizer_claim_by_sklearn(myclaim)
            my_document_l = doc_by_tf_idf(query_list,invert_index_dict, my_document_keys)
            data[json_key]['evidence'] = my_document_l
            i = i+1
            if i%100 == 0:
                print(str(i)+'data processed'+datetime.datetime.now().strftime(ISOTIMEFORMAT))
        json.dump(data,dg,indent=1)
        f.close()
        dg.close()
f.close()
    
    
    
    
    