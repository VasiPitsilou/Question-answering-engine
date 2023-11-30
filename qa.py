import pandas as pd
import random 
import torch
import torch.nn as nn
import warnings
from SPARQLWrapper import SPARQLWrapper, JSON
from transformers import BertTokenizer, BertModel
from sklearn.metrics import f1_score

train_data = pd.read_csv('wikidata_train.txt', sep="\t", names=['entity_id','relation_id', 'answer', 'question']) #read train dataset
val_data = pd.read_csv('wikidata_test.txt', sep="\t", names=['entity_id','relation_id', 'answer', 'question']) #read validation or test dataset (enter test set here)

train_data = train_data.drop(labels='answer', axis=1) #drop answer
val_data = val_data.drop(labels='answer', axis=1) #drop answer

sparql = SPARQLWrapper("https://query.wikidata.org/sparql", agent='ChicoBot Test agent')
entity_names = []
entity_ids = []
for i in range(0,len(train_data),100):  #retrieve entities' corresponding ids
  str = ""
  for entity_id in train_data['entity_id'][i:i+100]:
    str = str + "wd:" + entity_id + " "
  sparql.setQuery("""
  SELECT ?item ?itemLabel 
  WHERE
  {
    VALUES ?item {""" + str + """}
    SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
  }
  """)
  sparql.setReturnFormat(JSON)
  results = sparql.query().convert()
  results_df = pd.json_normalize(results['results']['bindings'])
  for j in results_df['item.value']:
      entity_ids.append(j)
  for j in results_df['itemLabel.value']:
      entity_names.append(j)

val_entity_names = []
val_entity_ids = []
for i in range(0,len(val_data),100):  #same for validation set
  str = ""
  for entity_id in val_data['entity_id'][i:i+100]:
    str = str + "wd:" + entity_id + " "
  sparql.setQuery("""
  SELECT ?item ?itemLabel 
  WHERE
  {
    VALUES ?item {""" + str + """}
    SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
  }
  """)
  sparql.setReturnFormat(JSON)
  results = sparql.query().convert()
  results_df = pd.json_normalize(results['results']['bindings'])
  for j in results_df['item.value']:
      val_entity_ids.append(j)
  for j in results_df['itemLabel.value']:
      val_entity_names.append(j)

"""Since SPARQL returns URI, extract ID from URI"""

entity_ids_without_link = []  #get ids from URIs
for j in entity_ids:
  entity_ids_without_link.append(j.split("http://www.wikidata.org/entity/",1)[1])

val_entity_ids_without_link = []
for j in val_entity_ids:
  val_entity_ids_without_link.append(j.split("http://www.wikidata.org/entity/",1)[1])

"""New column containing ID for train and validation/test set"""

new_df_column = []  #this will become the train_data new column, containing entities' labels
for entity_id in train_data['entity_id']: #for every entity id in initial dataset
  j=0
  for parse in entity_ids_without_link: #find its id on list
    if entity_id==parse:
      new_df_column.append(entity_names[j]) #and find corresponding label
      break
    j += 1

val_new_df_column = []  #this will become the val_data new column, containing entities' labels
for entity_id in val_data['entity_id']: #for every entity id in initial dataset
  j=0
  for parse in val_entity_ids_without_link: #find its id on list
    if entity_id==parse:
      val_new_df_column.append(val_entity_names[j]) #and find corresponding label
      break
    j += 1

train_data['entity_label'] = new_df_column  #new column containing entity id
val_data['entity_label'] = val_new_df_column

columns_titles = ["entity_id","entity_label","relation_id","question"]
train_data=train_data.reindex(columns=columns_titles)
val_data = val_data.reindex(columns=columns_titles)

"""Create array of 0s and 1s representing entity label span"""

entity_span = []  #create entity span column
parse = 0
for question in train_data['question']:
  current_entity = []
  for word in question.split():
    if word in train_data.iloc[parse]['entity_label'].casefold().split():
      current_entity.append(1)
    else:
      current_entity.append(0)
  entity_span.append(current_entity)
  parse += 1

train_data['entity_span'] = entity_span

val_entity_span = []  #create entity span column for validation set
parse = 0
for question in val_data['question']:
  current_entity = []
  for word in question.split():
    if word in val_data.iloc[parse]['entity_label'].casefold().split():
      current_entity.append(1)
    else:
      current_entity.append(0)
  val_entity_span.append(current_entity)
  parse += 1

val_data['entity_span'] = val_entity_span

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

"""Relation vocabulary for multi-label relation classification"""

relation_vocabulary = []  #this will be the relations vocabulary, and the fine-tuned BERT model will classify each question relation as one of these categories 
for relation in train_data['relation_id']:
  flag = 0
  for already_saved_relation in relation_vocabulary:
    if relation == already_saved_relation:
      flag = 1
      break
  if flag == 0:
    relation_vocabulary.append(relation)

"""Save questions and their encodings in lists for convenience"""

train_questions = []
for question in train_data['question']:
  train_questions.append(question)
train_encodings = tokenizer(train_questions, truncation=True, padding=True, return_tensors='pt')
train_relations = []
for relation in train_data['relation_id']:
  count = 0
  for relation_id in relation_vocabulary:
    if relation == relation_id:
      train_relations.append(count)
    count += 1

val_questions = []
for question in val_data['question']:
  val_questions.append(question)
val_encodings = tokenizer(val_questions, truncation=True, padding=True, return_tensors='pt')
val_relations = []
for relation in val_data['relation_id']:
  count = 0
  for relation_id in relation_vocabulary:
    if relation == relation_id:
      val_relations.append(count)
    count += 1

"""Make one-hot encoded matrices for entity span start index (train set)"""

train_span_start = [] #find span start
for span in train_data['entity_span']:
  train_span_cur = []
  set_zero = 0
  for flag in span:
    if (flag == 1 and set_zero == 0):
      set_zero = 1
      train_span_cur.append(1)
      continue
    if (flag == 0 or (flag == 1 and set_zero == 1)):
      train_span_cur.append(0)
  train_span_start.append(train_span_cur)

max = 0 
for span in train_span_start:  #padding
  if (len(span) > max):
    max = len(span)
for span in train_span_start:
  for i in range(max - len(span)):
    span.append(0)

"""Find position of span start"""

#span start index (position of entity span's start)

train_span_start_index = []
for i in train_span_start:
  counter = 0
  for parser in i:
    if parser == 1:
      train_span_start_index.append(counter)
      break
    counter += 1
    if counter == len(i):   #if no span is detected (array is full of 0's, set the span start as a random number)
      train_span_start_index.append(random.randint(1, 10))

"""Make one-hot encoded matrices for entity span end index (train set)"""

train_span_end = [] #find span end
for span in train_data['entity_span']:
  train_span_cur = []
  one_area = 0
  counter = 0
  for flag in span:
    if flag == 0 and one_area == 0:
      train_span_cur.append(0)
    if flag == 1:
      one_area = 1
      train_span_cur.append(0)
    if flag == 0 and one_area == 1:
      train_span_cur[counter-1] = 1
      train_span_cur.append(0)
      one_area = 0
    counter += 1
  train_span_end.append(train_span_cur)

max = 0
for span in train_span_end:
  if (len(span) > max):
    max = len(span)
for span in train_span_end:
  for i in range(max - len(span)):
    span.append(0)

"""Find position of span end"""

train_span_end_index = [] #span end index (position of entity span's end)
for i in train_span_end:
  counter = 0
  for parser in i:
    if parser == 1:
      train_span_end_index.append(counter)
      break
    counter += 1
    if counter == len(i):
      train_span_end_index.append(random.randint(1, 10))

"""Same procedure as above for validation set"""

val_span_start = [] #find span start on validation set
for span in val_data['entity_span']:
  val_span_cur = []
  set_zero = 0
  for flag in span:
    if (flag == 1 and set_zero == 0):
      set_zero = 1
      val_span_cur.append(1)
      continue
    if (flag == 0 or (flag == 1 and set_zero == 1)):
      val_span_cur.append(0)
  val_span_start.append(val_span_cur)

max = 0
for span in val_span_start:
  if (len(span) > max):
    max = len(span)
for span in val_span_start:
  for i in range(max - len(span)):
    span.append(0)

val_span_start_index = [] #index of span start on validation set questions
for i in val_span_start:
  counter = 0
  for parser in i:
    if parser == 1:
      val_span_start_index.append(counter)
      break
    counter += 1
    if counter == len(i):
      val_span_start_index.append(random.randint(1, 10))

val_span_end = [] #find span end on validation set
for span in val_data['entity_span']:
  val_span_cur = []
  one_area = 0
  counter = 0
  for flag in span:
    if flag == 0 and one_area == 0:
      val_span_cur.append(0)
    if flag == 1:
      one_area = 1
      val_span_cur.append(0)
    if flag == 0 and one_area == 1:
      val_span_cur[counter-1] = 1
      val_span_cur.append(0)
      one_area = 0
    counter += 1
  val_span_end.append(val_span_cur)

max = 0
for span in val_span_end:   #padding
  if (len(span) > max):
    max = len(span)
for span in val_span_end:
  for i in range(max - len(span)):
    span.append(0)

val_span_end_index = [] #index of span ending on validation set questions
for i in val_span_end:
  counter = 0
  for parser in i:
    if parser == 1:
      val_span_end_index.append(counter)
      break
    counter += 1
    if counter == len(i):
      val_span_end_index.append(random.randint(1, 10))

"""Dataset for relation prediction model"""

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_encodings, train_relations)
val_dataset = Dataset(val_encodings, val_relations)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True) 
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)

"""Neural network for relation prediction"""

class BertNetwork(nn.Module): 
    def __init__(self):
        super(BertNetwork, self).__init__()
      
        self.bert = BertModel.from_pretrained("bert-base-uncased")     
        self.out = nn.Linear(768, len(relation_vocabulary)) # we know that BERT ouput size is 768
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1) 

    def forward(self, input_ids, attention):
        unused, h1 = self.bert(input_ids = input_ids, attention_mask = attention, return_dict = False)
        h2 = self.out(h1) 
        h3 = self.relu(h2)
        return h3

device = torch.device('cuda' if (torch.cuda.is_available) else 'cpu')
device = 'cpu'
print('device: ', device)
model = BertNetwork()
model.to(device)

"""Train model for relation prediction"""

# warnings.filterwarnings("ignore")
# model.train()
# optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
# loss_func = nn.CrossEntropyLoss()
# for epoch in range(1):
#   num = 0
#   loss_list = []
#   accuracy_list = []
#   f1_list = []
#   accuracy_list_val = []
#   f1_list_val = []
#   for batch in train_dataloader:
#     num += 1
#     optimizer.zero_grad() 
#     output = model(batch['input_ids'],batch['attention_mask'])  
#     loss = loss_func(output, batch['labels']) 
#     loss_list.append(loss.item())
#     accuracy_list.append((torch.argmax(output, dim=1) == batch['labels']).sum().item()) 
#     f1_list.append(f1_score(batch['labels'], torch.argmax(output, dim=1), average='micro'))
#     loss.backward() 
#     optimizer.step()
#   with torch.no_grad():
#     for val_batch in val_dataloader:
#       output = model(val_batch['input_ids'],val_batch['attention_mask'])  
#       accuracy_list_val.append((torch.argmax(output, dim=1) == val_batch['labels']).sum().item()) 
#       f1_list_val.append(f1_score(val_batch['labels'],torch.argmax(output, dim=1), average='micro'))
#   print('Finished epoch ', epoch, 'with loss ', sum(loss_list)/len(loss_list), ', train accuracy', sum(accuracy_list)/len(train_dataloader.dataset), ', validation accuracy', sum(accuracy_list_val)/len(val_dataloader.dataset))
#   print('Train F1 score', sum(f1_list)/len(train_dataloader), ', validation F1 score ', sum(f1_list_val)/len(val_dataloader))

"""Save model"""

# torch.save(model.state_dict(),'drive/MyDrive/relation_model')

"""Entity span prediction

"""

class SpanDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, span_start_indexes, span_end_indexes):
        self.encodings = encodings
        self.span_start_indexes = span_start_indexes
        self.span_end_indexes = span_end_indexes

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['span_start_indexes'] = torch.tensor(self.span_start_indexes[idx])
        item['span_end_indexes'] = torch.tensor(self.span_end_indexes[idx])
        return item

    def __len__(self):
        return len(self.span_start_indexes)

entity_train_dataset = SpanDataset(train_encodings, train_span_start_index, train_span_end_index)
entity_val_dataset = SpanDataset(val_encodings, val_span_start_index, val_span_end_index)

entity_train_dataloader = torch.utils.data.DataLoader(entity_train_dataset, batch_size=16, shuffle=True) 
entity_val_dataloader = torch.utils.data.DataLoader(entity_val_dataset, batch_size=16, shuffle=True)

class EntityBertNetwork(nn.Module): 
    def __init__(self):
        super(EntityBertNetwork, self).__init__()
      
        self.bert = BertModel.from_pretrained("bert-base-uncased")     
        self.out_start = nn.Linear(768, len(train_span_start[1])) # we know that BERT ouput size is 768
        self.out_end = nn.Linear(768, len(train_span_end[1]))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1) 

    def forward(self, input_ids, attention):
        unused, h1 = self.bert(input_ids = input_ids, attention_mask = attention, return_dict = False)
        h2_start = self.out_start(h1) 
        h3_start = self.relu(h2_start)
        h2_end = self.out_end(h1) 
        h3_end = self.relu(h2_end)
        return h3_start, h3_end

device = torch.device('cuda' if (torch.cuda.is_available) else 'cpu')
device = 'cpu'
print('device: ', device)
entity_model = EntityBertNetwork()
entity_model.to(device)

"""Train entity span prediction model"""

# warnings.filterwarnings("ignore")
# entity_model.train()
# optimizer = torch.optim.Adam(entity_model.parameters(), lr=5e-5)
# loss_func = nn.CrossEntropyLoss()

# for epoch in range(1):
#   loss_list = []  #initialize evaluation metrics for new epoch
#   accuracy_list = []
#   f1_list = []
#   f1_list_end = []
#   accuracy_list_val = []
#   f1_list_val = []
#   f1_list_val_end = []

#   for batch in entity_train_dataloader:
#     optimizer.zero_grad() 

#     output_start, output_end = entity_model(batch['input_ids'], batch['attention_mask'])  #predict values

#     loss = loss_func(output_start, batch['span_start_indexes']) + loss_func(output_end, batch['span_end_indexes'])
#     loss_list.append(loss.item())

#     accuracy_list.append((torch.argmax(output_start, dim=1) == batch['span_start_indexes']).sum().item()) 

#     f1_list.append(f1_score(batch['span_start_indexes'], torch.argmax(output_start, dim=1), average='micro')) #batch F1 score for entity span start
#     f1_list_end.append(f1_score(batch['span_end_indexes'], torch.argmax(output_end, dim=1), average='micro')) #batch F1 score for entity span end

#     loss.backward() 
#     optimizer.step()
#   with torch.no_grad(): #validation set evaluation
#     for val_batch in entity_val_dataloader:

#       output_start, output_end = entity_model(val_batch['input_ids'],val_batch['attention_mask'])
#       accuracy_list_val.append((torch.argmax(output_start, dim=1) == val_batch['span_start_indexes']).sum().item()) 
#       f1_list_val.append(f1_score(val_batch['span_start_indexes'],torch.argmax(output_start, dim=1), average='micro'))
#       f1_list_val_end.append(f1_score(val_batch['span_end_indexes'],torch.argmax(output_end, dim=1), average='micro'))
#   #print epoch results
#   print('Finished epoch ', epoch, 'with loss ', sum(loss_list)/len(loss_list))
#   print('Train accuracy start', sum(accuracy_list)/len(train_dataloader.dataset), ', validation accuracy start', sum(accuracy_list_val)/len(val_dataloader.dataset))
#   print('Train F1 score start', sum(f1_list)/len(train_dataloader), ', validation F1 score start ', sum(f1_list_val)/len(val_dataloader))
#   print('Train F1 score end', sum(f1_list_end)/len(train_dataloader), ', validation F1 score end ', sum(f1_list_val_end)/len(val_dataloader))

"""Save model"""

# torch.save(entity_model.state_dict(),'drive/MyDrive/entity_model_2')

"""Evaluate models on test set"""

#Load already trained model
model.load_state_dict(torch.load('relation_model'))
entity_model.load_state_dict(torch.load('entity_model'))
accuracy_list_val = []
f1_list_val = []

with torch.no_grad():
    for val_batch in val_dataloader:
      output = model(val_batch['input_ids'],val_batch['attention_mask'])  
      accuracy_list_val.append((torch.argmax(output, dim=1) == val_batch['labels']).sum().item()) 
      f1_list_val.append(f1_score(val_batch['labels'],torch.argmax(output, dim=1), average='micro'))
print('Test accuracy for relation prediction', sum(accuracy_list_val)/len(val_dataloader.dataset))
print('Test F1 score for relation prediction', sum(f1_list_val)/len(val_dataloader))

accuracy_list = []
accuracy_list_val = []
f1_list_val = []
f1_list_val_end = []

with torch.no_grad(): #validation set evaluation
    for val_batch in entity_val_dataloader:

      output_start, output_end = entity_model(val_batch['input_ids'],val_batch['attention_mask'])
      accuracy_list_val.append((torch.argmax(output_start, dim=1) == val_batch['span_start_indexes']).sum().item()) 
      f1_list_val.append(f1_score(val_batch['span_start_indexes'],torch.argmax(output_start, dim=1), average='micro'))
      f1_list_val_end.append(f1_score(val_batch['span_end_indexes'],torch.argmax(output_end, dim=1), average='micro'))
print('Test accuracy for span prediction start', sum(accuracy_list_val)/len(val_dataloader.dataset))
print('Test F1 score for span prediction start', sum(f1_list_val)/len(val_dataloader))
print('Test F1 score for span prediction end', sum(f1_list_val_end)/len(val_dataloader))

"""QA engine ☺ """

print('Welcome to the question answering engine. Type \'exit\' to quit ')
while(1):
  question = input()  #read question
  if question == "exit":
    break
  question_bert_ready = tokenizer(question, return_tensors = 'pt')  #prepare question for BERT model
  relation_pred = model(question_bert_ready['input_ids'],question_bert_ready['attention_mask'])  #predict question relation
  entity_start, entity_end = entity_model(question_bert_ready['input_ids'],question_bert_ready['attention_mask']) #predict question entity span

  entity_name = ''  #initialize entity name
  counter = 0
  for question_word in question.split():  #save entity label
    if counter >= torch.argmax(entity_start, dim=1).item() and counter <= torch.argmax(entity_end, dim=1).item():
      if entity_name == '':
        entity_name = entity_name + question_word
      else:
        entity_name = entity_name + ' ' + question_word #add gap between words
    counter += 1
  print('entity_name ', entity_name)

  relation = relation_vocabulary[torch.argmax(relation_pred, dim=1).item()] #save relation
  print('relation ',relation_vocabulary[torch.argmax(relation_pred, dim=1).item()])
  counter = 0
  for parser in train_data['entity_label']: #for given entity label, find its ID for sparql
    if parser.lower().find(entity_name.lower()) >= 0:
      entity_id = train_data['entity_id'][counter]
    counter += 1
  print('entity_id ', entity_id)

  #ask for an answer to given entity and relation
  sparql.setQuery(""" 
  SELECT ?item ?itemLabel 
  WHERE 
  {
    wd:""" + entity_id + """ wdt:""" + relation + """ ?item.
    SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". } 
  }
    """)
  sparql.setReturnFormat(JSON)
  results = sparql.query().convert()
  results_df = pd.json_normalize(results['results']['bindings'])
  if results_df.empty:  #if sparql returned no answer, print message
    print('Sorry, no answer available!')
    continue
  for j in results_df['item.value']:
      answer_id = j
      break
  for j in results_df['itemLabel.value']:
      answer_name = j
      break
  print('answer: ', answer_name)  #else print first returned answer :)
