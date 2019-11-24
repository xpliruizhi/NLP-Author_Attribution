from NLI_Model import ESIM
from Corpus import SNLICorpus
from Corpus import WikiCorpus

corpus_snli = SNLICorpus()
embedding_matrix, word_embedding = corpus_snli.get_embedding_matrix()
tokenizer = corpus_snli.get_tokenizer()

corpus = WikiCorpus(tokenizer)
train_x, train_y, dev_x, dev_y, class_weights = corpus.generate_training_data(word_embedding)

print(len(corpus.tokenizer.word_index))


NLI_model = ESIM(3, 100, embedding_matrix, corpus.get_voc_size())
model = NLI_model.model
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
model.load_weights('NLI100.h5')
model.fit(train_x, train_y, epochs = 2, batch_size = 128, validation_data=(dev_x, dev_y),class_weight=class_weights)
model.save_weights('NLI100_finetune2.h5')

