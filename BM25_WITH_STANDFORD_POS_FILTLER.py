from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
import datetime
import pdb
import gc
import zipfile
import json
import simplejson
from nltk.corpus import stopwords 
from collections import Counter,defaultdict
from gensim import summarization
from nltk.tag import StanfordPOSTagger
import nltk

ISOTIMEFORMAT = '%H:%M:%S'
stop_words = set(stopwords.words('english'))
st = StanfordPOSTagger('english-bidirectional-distsim.tagger')
Meaningful_Tag = {'NN','NNP','NNS','NNPS','FW','CD'}
from nltk.tag import StanfordPOSTagger
from sner import POSClient
#nltk.download('stanford-tagger')
# tagger = POSClient(host='localhost', port=9198)

def BM25_PREPROCESSING(Sentence,Istitle):
    Sentence = Sentence.replace("_"," ").replace("-LRB-","(").replace("-RRB-",")").replace("-LSB-","]").replace("-RSB-","[")
    #pdb.set_trace()
    New_line = []
    ## IF is title, we do not filter the word by tagging
    ## Else use POS tag to filter the word from text, leave the text
    if Istitle:
        words = wordpunct_tokenize(Sentence.lower())
        for word in words:
            if word in stop_words:
                continue
            else:
                if word.isalpha() or word.isalnum():
                    New_line.append(word)
    # else:
        # words = wordpunct_tokenize(Sentence.lower())
        # ####Standford POStagging using
        # #tagged_word_list =st.tag(words)
        # #tagged_word_list = st.tag_sents([words])
        # tagged_word_list = nltk.pos_tag(words)
        # #tagged_word_list = tagger.tag(Sentence.lower())
        # ####Standford POStagging
        # for word,tag in tagged_word_list:
                # if tag in Meaningful_Tag and(word.isalpha() or word.isalnum()):
                    # New_line.append(word)                        
    return New_line

def BM25_PREPROCESSING_FOR_TAGGER(Sentence):

    Sentence[0] = Sentence[0].replace("_"," ").replace("-LRB-","(").replace("-RRB-",")").replace("-LSB-","]")
    Sentence[2] = Sentence[2].replace("_"," ").replace("-LRB-","(").replace("-RRB-",")").replace("-LSB-","]")
    #pdb.set_trace()
    New_line = []
    ## IF is title, we do not filter the word by tagging
    ## Else use POS tag to filter the word from text, leave the text
    words = wordpunct_tokenize(Sentence[0].lower())
    for word in words:
        if word in stop_words:
            continue
        else:
            if word.isalpha() or word.isalnum():
                New_line.append(word)
                
    words = wordpunct_tokenize(Sentence[2].lower())
    ####Standford POStagging using
    #tagged_word_list =st.tag(words)
    #tagged_word_list = st.tag_sents([words])
    tagged_word_list = nltk.pos_tag(words)
    #tagged_word_list = tagger.tag(Sentence.lower())
    ####Standford POStagging
    for word,tag in tagged_word_list:
        if tag in Meaningful_Tag:
            if(word.isalpha() or word.isalnum()):
                New_line.append(word)                        
    return New_line

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
        # key_count = 0
        # previous_key = ""
        for line in fileobj:
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
            value = BM25_PREPROCESSING(line_list[0],True)
            value_txt = BM25_PREPROCESSING(line_list[2],False)
            my_docs_key[temp]= value.extend(value_txt)
        # break
        myzip.close()
    print("File Reading Finish,time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    return my_docs_key

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
        print("start read "+str(i)+"file:"+ ziplist[i]+datetime.datetime.now().strftime(ISOTIMEFORMAT))
    #    loop each line from txt
        j = 0      
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
    #my_docs_key = {}
    #my_docs_key = []
    my_docs_detail = []
    print("begin to read wiki sentences,time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    myzip = zipfile.ZipFile('wiki-pages-text.zip')
    ziplist = myzip.namelist()
    ## loop each txt file
    title_list = []
    for i in range(1,len(ziplist)):
        myzip = zipfile.ZipFile('wiki-pages-text.zip')
        fileobj = myzip.open(ziplist[i])
        print("start read "+str(i)+"file:"+ ziplist[i]+datetime.datetime.now().strftime(ISOTIMEFORMAT))
    #    loop each line from txt
        j = 0
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
            #temp = line_list[0]+'$$'+line_list[1]
            #value_txt = BM25_PREPROCESSING_FOR_TAGGER(line_list)
            #my_docs_key.append(temp)
            #pdb.set_trace()
            value_txt = sent_tag_prop(line_list[2])
            my_docs_detail.append(value_txt)
            title_list.append(BM25_PREPROCESSING(line_list[0],True))
        # break
        myzip.close()
    print("read sentence finished,Start tagging"+datetime.datetime.now().strftime(ISOTIMEFORMAT))
    my_docs_detail = nltk.pos_tag_sents(my_docs_detail)
    print("Tagging finished,Start add title"+datetime.datetime.now().strftime(ISOTIMEFORMAT))
    i = 0
    my_docs_detail_new = []
    lengh = len(my_docs_detail)
    for single_sentence in my_docs_detail:
        New_line = []
        for word,tag in single_sentence:
            if tag in Meaningful_Tag:
                if(word.isalpha() or word.isalnum()):
                    New_line.append(word)
        New_line.extend(title_list[i])
        i = i+1
        if i%1000000 == 0:
            print(str(i)+'data processed, total'+str(lengh)+"##"+datetime.datetime.now().strftime(ISOTIMEFORMAT))
        my_docs_detail_new.append(New_line)
    print("Add title Finish,time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    return my_docs_detail_new
    
def sent_tag_prop(Sentence):
    Sentence = Sentence.replace("_"," ").replace("-LRB-","(").replace("-RRB-",")").replace("-LSB-","]")
    words = wordpunct_tokenize(Sentence.lower())
    return words
    
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
    return inverted_index

def doc_by_BM25(query_list,invert_index_dict, document_keys):
    #pdb.set_trace()
    word_index_counter = Counter()
    for word in query_list:
        if word in invert_index_dict.keys():    
            word_index_counter = word_index_counter + Counter(invert_index_dict[word])
    #print(word_index_counter)
    doc_ranking = word_index_counter.most_common(5)
    rank_list = []
    #pdb.set_trace()
    for element in doc_ranking:
        TITLE = document_keys[int(element[0])].split('$$')
        rank_list.append([TITLE[0],TITLE[1]])
    return rank_list  
def run_first():
    my_docs_detail = Read_from_WIKI_Sent()
    print("BM25 initialise....TIME:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    my_bm25 = summarization.bm25.BM25(my_docs_detail)
    print("BM25 Initialization Finished....TIME:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    my_docs_detail = None

    print("Building Inverted index...TIME:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    invert_index_dict = build_inverted_index_BM25(my_bm25)
    my_bm25 = None
    print("Inverted index finished_NOW_STORE THE CONTENT...TIME:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))

    with open("model_bm25_tagger_invelist.json","w") as dg:
        simplejson.dump(invert_index_dict,dg,indent=1)
    dg.close()
    print("Inverted index model STORE THE CONTENT...TIME:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))

def load_model_by_json():
    print("Start Load BM25 invertedIndex model, Time: "+datetime.datetime.now().strftime(ISOTIMEFORMAT))
    with open('model_bm25_tagger_invelist.json', 'r') as f:
        invert_index_dict = simplejson.load(f)
    print("Finish Time"+datetime.datetime.now().strftime(ISOTIMEFORMAT))
    return invert_index_dict
#run_first()    
invert_index_dict = load_model_by_json()
# pdb.set_trace()
my_docs = Read_from_WIKI_Title()

# myclaim = 'is Alta Outcome Document bullshit'
# print(list(my_docs_key.keys()))
# print("####################################################")
# print(list(my_docs_key.values()))

# q_list = BM25_PREPROCESSING('is American Thighs bullshit',True)
# print(q_list)
# rank_list = doc_by_BM25(q_list,invert_index_dict, my_docs)
# print(rank_list)

# query_list = BM25_PREPROCESSING(myclaim)
# print(query_list)
# my_rank = my_bm25.ranked(query_list,5)
# print(my_rank)
# my_document_l = []
# for it in my_rank:
    # my_document_l.append(my_docs[it])
# print(my_document_l)

with open('devset.json', 'r') as f:
    print('read json file starting'+datetime.datetime.now().strftime(ISOTIMEFORMAT))
    data = json.load(f)
    total_len = len(data)
    print('total'+str(total_len)+'data')
    i=0
    with open("record_title_bm25_postag.json","w") as dg:
        for json_key in data.keys():
            myclaim = data[json_key]['claim']
            query_list = BM25_PREPROCESSING(myclaim,True)
            pdb.set_trace()
            my_document_l = doc_by_BM25(query_list,invert_index_dict, my_docs)
            data[json_key]['evidence'] = my_document_l
            i = i+1
            if i%100 == 0:
                print(str(i)+'data processed'+datetime.datetime.now().strftime(ISOTIMEFORMAT))
        json.dump(data,dg,indent=1)
        f.close()
        dg.close()
f.close()








