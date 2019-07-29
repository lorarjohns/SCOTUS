class JSONReader:
    def __init__(self, *args, **kwargs):
      self.attributes = {'absolute_url': 'unknown','author': 'unknown','author_str': 'unknown','cluster': 'unknown','date_created': 'unknown','date_modified': 'unknown','download_url': 'unknown','extracted_by_ocr': 'unknown','html': 'unknown','html_columbia': 'unknown','html_lawbox': 'unknown','html_with_citations': 'unknown','id': 'unknown','joined_by': 'unknown','local_path': 'unknown','opinions_cited': 'unknown','page_count': 'unknown','per_curiam': 'unknown','plain_text': 'unknown','resource_uri': 'unknown','sha1': 'unknown','type': 'unknown'}

    def read_json(self, files): ## the reading
        for file in files:
          with open(file, 'r') as f:
            data = json.loads(f.read())
        yield data
    
         
    def no_nulls(self, files):
        data = read_json(files)
        for d in data:
          record = self.attributes
          for k,v in record.values():
          if data.get(k) is not None:
            record[k] = data[k]
            yield record
    
    def json_to_df(self, files):
      records = no_nulls(files)
      for record in records:
          df = pd.concat(record)
      return df
            




def __iter__(self):
        for file in enumerate(self.files,1):
            with open(file, 'r') as f:
                data = json.loads(f.read())
            yield data


# # Tokenize, tag, lemmatize, parse, named entity recognition, stop words
# 
# import spacy
# 
# nlp = spacy.load('en_core_web_lg')
# 
# def lemmatizer(doc):
#     doc = [token.lemma_ for token in doc if (not token.lemma_ == '-PRON-' and not token.is_stop and not token.is_digit)]
#     doc = u' '.join(doc)
#     return nlp.make_doc(doc)
# 
# nlp.add_pipe(lemmatizer, 'lemmatizer', after='tagger')
# nlp.disable_pipe('ner')
# spacy_docs = []
# for doc in tqdm_notebook(nlp.pipe(documents)):    
#     doc = [token.text for token in doc]
#     doc = u' '.join(doc)
#     spacy_docs.append(nlp.make_doc(doc))
# 
# spacy_df = pd.DataFrame([]).append([spacy_docs])
# spacy_df.columns = ['spacy_docs']

# spacy_df.reset_index(drop=True, inplace=True)
# mini_df.reset_index(drop=True, inplace=True)

# mini_df = pd.concat([mini_df, spacy_df], axis=1)

# def preprocess_chunk(chunk):
#     doclist = []
#     cleaned = []
#     for doc in tqdm.tqdm(chunk):
#         doc = replace_citations(doc)
#         doclist.append(doc)
#     for doc in tqdm.tqdm(nlp.pipe(doclist)):
#         doc = [token.lemma_ for token in doc if (not token.is_stop and not token.is_digit and not token.is_punct)]
#         doc = " ".join(doc)
#         cleaned.append(doc)
#     return cleaned
# 
# def build_corpora(file_path, chunksize=200):
#     corpora = []
#     for chunk in pd.read_csv(file_path, chunksize=chunksize):
#         chunk = preprocess_chunk(chunk.clean_text)
#         corpora.append(chunk)
#        # corpora = [y for x in corpora for y in x] # flatten list
#     return corpora