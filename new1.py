# import keras.preprocessing.text as T
# from keras.preprocessing.text import Tokenizer

import datetime
import zipfile
import json
import collections
import gc
import pickle
import pymysql
import hashlib

import pdb
import heapq
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np

import nltk
from collections import Counter
from DatabaseTool import DataToMysql

vocab = {}
stemmer = nltk.stem.PorterStemmer()
total_tokens = 0
doc_term_freqs = []
    # norm_doc stores the normalized tokens of a doc
norm_doc = []
term_id = 0

#vectorizer = TfidfVectorizer(stop_words='english')
#vectorizer = TfidfVectorizer()
ISOTIMEFORMAT = '%H:%M:%S'
######################Read Wiki ZIP File#####################################
def Read_from_WIKI():
    mydocument = collections.defaultdict(dict)
    
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
        for line in fileobj:
            # must use utf-8 to store different languages
            # remove "/n" at each line 
            myline = line.decode('utf-8').strip()
            # use first 2 blanks to cut the string
            # j += 1
            # if j>90:
                # break
            line_list = myline.split(' ',2)
            temp = line_list[0]+'$'+line_list[1]
            mydocument[temp] = line_list[2]
            
            # key = line_list[0].replace("_"," ")
            # key = key.replace("-LRB-","(")
            # key = key.replace("-RRB-",")")
            # if key in my_docs_key.keys():
                # key_count = key_count
            # else:
                # key_count = key_count + 1
                # my_docs_key[key]= key_count
        #break
        myzip.close()
    print("File Reading Finish,time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    return mydocument

def Build_tfidf_model_from_WIKI(mydocument):
    print("Start building matrix......")
    vectorizer = TfidfVectorizer(stop_words='english')
    my_response_1 = vectorizer.fit_transform(mydocument.values())

    print("Matrix build success,time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    print("Writing.....")
    #store the content
    with open("model_tfidf.npy", 'wb') as handle:
                        pickle.dump(my_response_1, handle)
    handle.close()
    with open("get_feature_names.npy", 'wb') as handle:
                        pickle.dump(vectorizer.get_feature_names(), handle)
    handle.close()
    print("Writing Finished,time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))

def Build_tfidf_model_from_WIKI_v2(mydocument):
    print("Start building matrix......")
    #vectorizer = TfidfVectorizer()
    vectorizer = TfidfVectorizer(stop_words='english')
    my_response_1 = vectorizer.fit_transform(mydocument.values())

    print("Matrix build success,time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    return my_response_1, vectorizer.get_feature_names()

    
def Load_tfidf_model():
    #load the content
    print("Start Load TFIDFmodel, Time:"+datetime.datetime.now().strftime(ISOTIMEFORMAT))
    my_response = pickle.load(open("model_tfidf.npy","rb"))
    vector_feature_name = pickle.load(open("get_feature_names.npy","rb"))
    print("Finish Load TFIDFmodel, Time:"+datetime.datetime.now().strftime(ISOTIMEFORMAT))
    return my_response,vector_feature_name

def Build_Inverted_Index_Momory_Files(my_response,word_name):
    print("Construct invert_index_dict")
    
    invert_index_dict = defaultdict(Counter)
    N = 0
    J = 0
    t_len =  len(my_response.nonzero()[1])
    str_ti = ""
   
    ####Build Inverted Index in Memory and Files####
    for col in zip(my_response.nonzero()[1],my_response.nonzero()[0]):
    # np.array(scaled, dtype=np.float16)
        
        invert_index_dict[word_name[col[0]]][col[1]] = float(my_response[col[1],col[0]])
        N = N+1
        
        if N%100000 == 0:
            print("Time:"+datetime.datetime.now().strftime(ISOTIMEFORMAT)+", "+str(N)+'data processed'+' total:'+ str(t_len))
        if N%100000000 == 0:
            J = J+1
            str_ti = "Invert_index"+str(J)
            print("Start writing Files"+str(J)+"Time:"+datetime.datetime.now().strftime(ISOTIMEFORMAT))
            with open(str_ti+"model.npy", 'wb') as handle:
                                pickle.dump(invert_index_dict, handle)
            invert_index_dict = defaultdict(Counter)
            handle.close()
            print(str_ti+"Finished Time:"+datetime.datetime.now().strftime(ISOTIMEFORMAT))
    #store the content
    J = J+1
    str_ti = "Invert_index"+str(J)
    with open(str_ti+"model.npy", 'wb') as handle:
                        pickle.dump(invert_index_dict, handle)
    del invert_index_dict
    gc.collect()
    handle.close()
    return J

def Load_inverted_index(J):
    #load the content
    i = 0
    Dict_list = []
    for i in range(0,J): 
        invert_index_dict_new = pickle.load(open("invert_list.mod", "rb" ))
        i = i+1

def Build_Inverted_Index_Database(my_response,word_name):
    N = 0
    t_len =  len(my_response.nonzero()[1])
    mysql = DataToMysql('localhost','root','@lrzxp123','pythonDB')
    Single_data = []
    data_list = []
    #####hash key value
    gg = hashlib.md5()
    #####
    for col in zip(my_response.nonzero()[1],my_response.nonzero()[0]):
        # if word_name[col[0]] == 'amīrnān' and int(col[1])==307:
            # print("col[0]="+str(col[0]))
            # print("col[1]="+str(col[1]))
            # print("Response_val="+str(my_response[col[1],col[0]]))
        my_key_word = word_name[col[0]].encode('utf-8')
        gg.update(my_key_word)
        Single_data = [str(gg.hexdigest()),int(col[1]),float(my_response[col[1],col[0]])]
        #pdb.set_trace()
        data_list.append(Single_data)
        
        del col
        del Single_data       
        
        N = N+1
        #pdb.set_trace()
        if N%100000 == 0:
            print("Time:"+datetime.datetime.now().strftime(ISOTIMEFORMAT)+", "+str(N)+'data processed'+' total:'+ str(t_len))
        if N%100000 == 0:
            print("Start insert to DB, Time:"+datetime.datetime.now().strftime(ISOTIMEFORMAT))
            mysql.writemany_list(data_list)
            print(str(N)+"data DB WRITE finished"+"Time:"+datetime.datetime.now().strftime(ISOTIMEFORMAT))
            Single_data = []
            data_list = []
    print("Start insert to DB_FINAL, Time:"+datetime.datetime.now().strftime(ISOTIMEFORMAT))
    mysql.writemany_list(data_list)

    del Single_data
    gc.collect() 

    del data_list
    gc.collect() 
    #pdb.set_trace()
    print("Invert_index_dict build success,time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))

def TFIDF_PREPROCESSING():
    my_document,my_docs_key = Read_from_WIKI()
    Build_tfidf_model_from_WIKI(my_document)
    document_keys = list(my_document.keys())
    del my_document
    gc.collect()
    return document_keys

#TFIDF_PREPROCESSING()

my_response,word_name = Load_tfidf_model()
#my_document = Read_from_WIKI()
#my_response,word_name = Build_tfidf_model_from_WIKI_v2(my_document)
Build_Inverted_Index_Database(my_response,word_name)
#J = Build_Inverted_Index_Momory_Files(my_response,word_name)

del my_response
gc.collect() 


# with open("tfidfmodel.json","w") as TFg:
    # json.dump(invert_index_dict,TFg)
    # TFg.close()
# print("Write Finished")
#print(invert_index_dict[0][33])
#print(my_response[0,33])  
#print(my_response)
# print(mydocument.values())
# my_response_list = response.toarray()
#print(my_response.nonzero()[1])
#print(vectorizer.get_feature_names())
#print(n)
  
def doc_by_tf_idf(query_list):
#    pdb.set_trace()
    word_index_counter = Counter()
    for word in query_list:
        if word in invert_index_dict.keys():    
            word_index_counter = word_index_counter + invert_index_dict[word]
    doc_ranking = word_index_counter.most_common(5)
    rank_list = []
    for element in doc_ranking:
        TITLE = document_keys[element[0]].split('$')
        rank_list.append([TITLE[0],TITLE[1]])
    return rank_list    
    
def tokenizer_claim_by_sklearn(claim):
    #my_c = claim.split(' ')
    count_vect = CountVectorizer(stop_words='english')
    #count_vect = CountVectorizer()
    count_vect.fit_transform([claim])
    #pdb.set_trace()
    return count_vect.get_feature_names()

# q_list = tokenizer_claim_by_sklearn('is Alta Outcome Document bullshit')
# print(q_list)
# rank_list = doc_by_tf_idf(q_list)
# print(rank_list)

######################Read json File##################################

# with open('devset.json', 'r') as f:
    # print('read json file starting')
    # data = json.load(f)
    # total_len = len(data)
    # print('total'+str(total_len)+'data')
    # i=0
    # with open("record.json","w") as dg:
        # for json_key in data.keys():
            # myclaim = data[json_key]['claim']
            # query_list = tokenizer_claim_by_sklearn(myclaim)
            # my_document_l = doc_by_tf_idf(query_list)
            # data[json_key]['evidence'] = my_document_l
            # i = i+1
            # if i%100 == 0:
                # print(str(i)+'data processed')
        # json.dump(data,dg)
# #        json.dumps(data, sort_keys = True, indent = 4, separators=(',', ':'))
        # f.close()
        # dg.close()
# f.close()
# print(data['137557'].keys())
# dict_keys(['claim', 'label', 'evidence'])

# print(data['121708'])
# {'claim': "Directing is a profession of Alfred Hitchcock's.", 
# 'label': 'SUPPORTS', 
# 'evidence': [['Alfred_Hitchcock', 2], ['Alfred_Hitchcock', 0], ['Alfred_Hitchcock', 19]]}

## 嵌套字典
# print(data['121708']['evidence'])
# [['Alfred_Hitchcock', 2], ['Alfred_Hitchcock', 0], ['Alfred_Hitchcock', 19]]   ## 二维数组

# def doc_by_tf_idf(query):
    ##resp_matrix = np.array(my_response.toarray())    
    # temp = np.zeros(len(resp_matrix))
    # for token in query:
        # if token in vectorizer.vocabulary_.keys():
            # myindex = vectorizer.vocabulary_[token]
           ##pdb.set_trace()
            # temp = temp + (resp_matrix.T)[myindex]
            
    # rank_list = heapq.nlargest(3, range(len(temp)), temp.take) 
    # mylist = []
    # return [document,score] list
    # for index in rank_list:
        # mylist.append([document_keys[index],temp[index]])
    # return mylist