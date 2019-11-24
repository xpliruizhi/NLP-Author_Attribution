from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
import datetime
import pdb
import gc
import zipfile
import json
from nltk.corpus import stopwords 
from collections import Counter,defaultdict
from gensim import summarization
from nltk.tag import StanfordPOSTagger
import pickle
import simplejson
### 提升效率，数组连接用 extend()
### 少用 not in,多用 in 替代
ISOTIMEFORMAT = '%H:%M:%S'
stop_words = set(stopwords.words('english'))
st = StanfordPOSTagger('english-bidirectional-distsim.tagger')
Meaningful_Tag = {'NN','NNP','NNS','NNPS','FW','CD'}
def BM25_PREPROCESSING(Sentence):
    Sentence = Sentence.replace("_"," ")
    Sentence = Sentence.replace("-LRB-"," ")
    Sentence = Sentence.replace("-RRB-"," ")
    Sentence = Sentence.replace("-RSB-"," ")
    Sentence = Sentence.replace("-LSB-"," ")
    words = wordpunct_tokenize(Sentence.lower())
    #pdb.set_trace()
    New_line = []
    ## IF is title, we do not filter the word by tagging
    ## Else use POS tag to filter the word from text, leave the text
    for word in words:
        if word in stop_words:
            continue
        else:
            if word.isalpha() or word.isalnum():
                New_line.append(word)                
    return New_line

def Read_from_WIKI():
    #my_docs_key = {}
    my_docs_key = []
    my_docs_detail = []
    print("begin to read wiki files,time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    myzip = zipfile.ZipFile('wiki-pages-text.zip')
    ziplist = myzip.namelist()
    ## loop each txt file
    for i in range(1,len(ziplist)):
        myzip = zipfile.ZipFile('wiki-pages-text.zip')
        fileobj = myzip.open(ziplist[i])
        print("start read "+str(i)+"file:"+ ziplist[i])
    #    loop each line from txt
        j = 0
        key_count = 0
        for line in fileobj:
            value = []
            # must use utf-8 to store different languages
            # remove "/n" at each line 
            myline = line.decode('utf-8').strip()
            # use first 2 blanks to cut the string
            # j += 1
            # if j>90:
                # break
            line_list = myline.split(' ',2)
            temp = line_list[0]+'$$'+line_list[1]
            #key_count = key_count + 1
            #pdb.set_trace()
            value = BM25_PREPROCESSING(line_list[0])
            value_txt = BM25_PREPROCESSING(line_list[2])
            my_docs_key.append(temp)
            #pdb.set_trace()
            value.extend(value_txt)
            my_docs_detail.append(value)
            #gc.collect()
        # break
        myzip.close()
    print("File Reading Finish,time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    #pdb.set_trace()
    gc.collect()
    return my_docs_key, my_docs_detail

def Read_from_WIKI_Title():
    #my_docs_key = {}
    my_docs_key = []
    #my_docs_detail = []
    print("begin to read wiki files title,time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    myzip = zipfile.ZipFile('wiki-pages-text.zip')
    ziplist = myzip.namelist()
    ## loop each txt file
    for i in range(1,len(ziplist)):
        myzip = zipfile.ZipFile('wiki-pages-text.zip')
        fileobj = myzip.open(ziplist[i])
        print("start read "+str(i)+"file:"+ ziplist[i])
    #    loop each line from txt
        j = 0
        key_count = 0
        
        for line in fileobj:
            #value = []
            # must use utf-8 to store different languages
            # remove "/n" at each line 
            myline = line.decode('utf-8').strip()
            # use first 2 blanks to cut the string
            # j += 1
            # if j>90:
                # break
            line_list = myline.split(' ',2)
            temp = line_list[0]+'$$'+line_list[1]
            #key_count = key_count + 1
            #pdb.set_trace()
            #value = BM25_PREPROCESSING(line_list[0])
            #value_txt = BM25_PREPROCESSING(line_list[2])
            my_docs_key.append(temp)
            #pdb.set_trace()
            #value.extend(value_txt)
            #my_docs_detail.append(value)
        # break
        myzip.close()
    print("File Reading Finish,time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    #pdb.set_trace()
    return my_docs_key

def Read_from_WIKI_Sent():
    my_docs_detail = []
    print("begin to read wiki sentences,time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    myzip = zipfile.ZipFile('wiki-pages-text.zip')
    ziplist = myzip.namelist()
    ## loop each txt file
    for i in range(1,len(ziplist)):
        myzip = zipfile.ZipFile('wiki-pages-text.zip')
        fileobj = myzip.open(ziplist[i])
        print("start read "+str(i)+"file:"+ ziplist[i])
    #    loop each line from txt
        j = 0
        for line in fileobj:
            value = []
            # must use utf-8 to store different languages
            # remove "/n" at each line 
            myline = line.decode('utf-8').strip()
            # use first 2 blanks to cut the string
            # j += 1
            # if j>90:
                # break
            line_list = myline.split(' ',2)
            #pdb.set_trace()
            value = BM25_PREPROCESSING(line_list[0])
            value_txt = BM25_PREPROCESSING(line_list[2])
            #pdb.set_trace()
            value.extend(value_txt)
            my_docs_detail.append(value)
        # break
        myzip.close()
    print("File Reading Finish,time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    #pdb.set_trace()
    #gc.collect()
    return my_docs_detail

def average_idf_cal(idf_dic):
    total_len = len(idf_dic.keys())
    return sum(idf_dic.values())/total_len

def build_inverted_index_BM25(new_bm):
    inverted_index = defaultdict(Counter)
    ave_idf = average_idf_cal(new_bm.idf)
    for i in range(0,new_bm.corpus_size-1):
        single_doc = list(new_bm.f[i].keys())
        for word in single_doc:
            inverted_index[word][i] = new_bm.get_score([word],i,ave_idf)
    print("Store the model to local file")
    #store the content
    with open("model_bm25_inverted_list.npy", 'wb') as handle:
                        pickle.dump(inverted_index, handle)
    #gc.collect()
    handle.close()
    print("Store the model Success")
    return inverted_index

def doc_by_BM25(query_list,invert_index_dict, document_keys):
    #pdb.set_trace()
    word_index_counter = Counter()
    for word in query_list:
        if word in invert_index_dict.keys():    
            word_index_counter = word_index_counter + invert_index_dict[word]
    #print(word_index_counter)
    doc_ranking = word_index_counter.most_common(5)
    rank_list = []
    #pdb.set_trace()
    for element in doc_ranking:
        TITLE = document_keys[element[0]].split('$$')
        rank_list.append([TITLE[0],TITLE[1]])
    return rank_list  

def doc_by_BM25_diff_title(query_list,invert_index_dict, document_keys):
    #pdb.set_trace()
    word_index_counter = Counter()
    for word in query_list:
        if word in invert_index_dict.keys():    
            word_index_counter = word_index_counter + invert_index_dict[word]
    #print(word_index_counter)
    doc_ranking = word_index_counter.most_common(150)
    rank_list = []

    mytitle = set()

    #pdb.set_trace()
    for element in doc_ranking:
        TITLE = document_keys[element[0]].split('$$')
        if TITLE[0] in mytitle:
            continue
        else:
            mytitle.add(TITLE[0])
            rank_list.append([TITLE[0],TITLE[1]])
            if len(rank_list) == 5:
                #pdb.set_trace()
                break    
    return rank_list

def load_model_inverted_index():
    #load the content
    print("Start Load BM25 invertedIndex model, Time:"+datetime.datetime.now().strftime(ISOTIMEFORMAT))
    with open('model_bm25_inverted_list.json', 'r') as f:
        my_invert_index = simplejson.load(f)   
    doc_feature_name = pickle.load(open("get_feature_names_BM25.npy","rb"))
    print("Finish Load BM25 model, Time:"+datetime.datetime.now().strftime(ISOTIMEFORMAT))
    #simplejson.load(fp, **kwargs)
    return my_invert_index,doc_feature_name


def run_for_the_first_time():
    #my_docs, my_docs_detail = Read_from_WIKI()
    my_docs_detail = Read_from_WIKI_Sent()
    # pdb.set_trace()
    print("BM25 initialise....TIME:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    my_bm25 = summarization.bm25.BM25(my_docs_detail)
    print("BM25 Initialization Finished....TIME:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    my_docs_detail = None
    del my_docs_detail
    #gc.collect()
    
    print("Building Inverted index...TIME:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    invert_index_dict = build_inverted_index_BM25(my_bm25)

    print("Inverted index finished...TIME:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    my_bm25 = None
    del my_bm25
    gc.collect()

    my_docs = Read_from_WIKI_Title()
    with open("get_feature_names_BM25.npy", 'wb') as handle:
                        pickle.dump(my_docs, handle)
    handle.close()
    #gc.collect()
    return invert_index_dict,my_docs

def load_model_by_json():
    print("Start Load BM25 invertedIndex model, Time: "+datetime.datetime.now().strftime(ISOTIMEFORMAT))
    with open('model_bm25_inverted_list.json', 'r') as f:
        doc_feature_name = simplejson.load(f)
    print("Finish Time"+datetime.datetime.now().strftime(ISOTIMEFORMAT))
    
# def transfer_npy():
    # #load the content
    # print("Loading BM25 invertedIndex model, Time: "+datetime.datetime.now().strftime(ISOTIMEFORMAT))
    # my_invert_index = pickle.load(open("model_bm25_inverted_list.npy","rb"))
    # #invert_index_dict, my_docs= load_model_inverted_index()
    # print("Load Finish, Start convert BM25 invertedIndex model, Time: "+datetime.datetime.now().strftime(ISOTIMEFORMAT))
    # with open("model_bm25_inverted_list.json","w") as dg:
        # simplejson.dump(my_invert_index,dg,indent=1)
    # dg.close()
    # print("Convert Success!!!, Time:"+datetime.datetime.now().strftime(ISOTIMEFORMAT))    
    
# def run_for_the_second_time():
    # invert_index_dict,my_docs = load_model_inverted_index()
    # return invert_index_dict,my_docs
    
# transfer_npy()
load_model_by_json()
# invert_index_dict,my_docs = run_for_the_second_time()
# with open('devset.json', 'r') as f:
    # print('read json file starting'+datetime.datetime.now().strftime(ISOTIMEFORMAT))
    # data = json.load(f)
    # total_len = len(data)
    # print('total'+str(total_len)+'data')
    # #gc.collect()
    # i=0
    # with open("record_title_bm25_use_all_sentence.json","w") as dg:
        # for json_key in data.keys():
            # myclaim = data[json_key]['claim']
            # query_list = BM25_PREPROCESSING(myclaim)
            # #my_document_l = doc_by_BM25(query_list,invert_index_dict, my_docs)
            # my_document_l = doc_by_BM25_diff_title(query_list,invert_index_dict, my_docs)
            # data[json_key]['evidence'] = my_document_l
            # i = i+1
            # if i%100 == 0:
                # print(str(i)+'data processed'+datetime.datetime.now().strftime(ISOTIMEFORMAT))
        # #gc.collect()
        # json.dump(data,dg,indent=1)
        # f.close()
        # dg.close()
# f.close()

# q_list = BM25_PREPROCESSING('is American Thighs bullshit')
# print(q_list)
# rank_list = doc_by_BM25_diff_title(q_list,invert_index_dict, my_docs)
# print(rank_list)

# query_list = BM25_PREPROCESSING(myclaim)
# print(query_list)
# my_rank = my_bm25.ranked(query_list,5)
# print(my_rank)
# my_document_l = []
# for it in my_rank:
    # my_document_l.append(my_docs[it])
# print(my_document_l)






