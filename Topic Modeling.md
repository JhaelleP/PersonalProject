Imported all of the modules


```python
# Importing modules---------------------------------------------
import pandas as pd #To import dataset
import re #To modify the text, i.e to remove punctuations
from wordcloud import WordCloud #To create wordclouds
import gensim #For topic modeling and natural language processing
from gensim.utils import simple_preprocess #Converts document into unicode strings
import gensim.corpora as corpora #A corpus library to store text documents
from gensim.models import CoherenceModel #To calculate topic coherence for topic models
import nltk #To process natural language
from nltk.corpus import stopwords #To remove unnecessary words from the string
from pprint import pprint #To maintain the format of non fundamental Python types
import spacy #To preprocess the string for later analysis
import pickle #To convert python objects into byte streams
import pyLDAvis #Aids in interpreting the topics created in a topic model
import pyLDAvis.gensim_models #For topic modelling, document indexing and similarity retrieval
import numpy as np #For manipulating arrays
import tqdm as tqdm #To show the progress of iteration loops (The percentage that is completed)
import matplotlib.pyplot as plt #For data visualization


```

    /Users/jhaelle/opt/anaconda3/lib/python3.8/site-packages/ipykernel/ipkernel.py:287: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
      and should_run_async(code)


The .csv file is uploaded, then removed the columns that were not needed for analysis. I only kept the bill number and bill text


```python

#Obtaining and filtering data-------------------------------------
docs = pd.read_csv('/Users/jhaelle/Desktop/PythonRB/bills.csv')
docs = docs.drop(columns=['Success', 'Sponsors', 'Type of Bill','Committees'], axis=1).sample(42)

```

    /Users/jhaelle/opt/anaconda3/lib/python3.8/site-packages/ipykernel/ipkernel.py:287: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
      and should_run_async(code)


From the "Text" column in "docs" all punctuations are removed and all words are lowercased. Then the first 5 rows are printed to ensure I obtained the desired preprocessed result.


```python

#Preprocessing---------------------------------------------------
# Punctuation
docs['Text_processed'] = \
docs['Text'].map(lambda x: re.sub('[,\.!?]', '', x))
# Lowercase
docs['Text_processed'] = \
docs['Text_processed'].map(lambda x: x.lower())
# Print
docs['Text_processed'].head()

```

    /Users/jhaelle/opt/anaconda3/lib/python3.8/site-packages/ipykernel/ipkernel.py:287: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
      and should_run_async(code)
    <>:4: DeprecationWarning: invalid escape sequence \.
    <>:4: DeprecationWarning: invalid escape sequence \.
    <ipython-input-8-428a8c300192>:4: DeprecationWarning: invalid escape sequence \.
      docs['Text'].map(lambda x: re.sub('[,\.!?]', '', x))





    9     as passed senate a bill to be entitled an act ...
    41    a bill to be entitled an act 1 to amend chapte...
    6     a bill to be entitled an act 1 to amend code s...
    42    a bill to be entitled an act 1 to amend chapte...
    29    a bill to be entitled an act 1 to amend chapte...
    Name: Text_processed, dtype: object



A word cloud is produced to visually reveal any other unnecessary words that needs to be removed.


```python

#Verify Preprocessing---------------------------------------------
# Join all texts together
cloud = ','.join(list(docs['Text_processed'].values))
# WordCloud object
wordcloud = WordCloud(width=500, height= 500, background_color="white", max_words=5000, contour_width=2, contour_color='red')
# Produce
wordcloud.generate(cloud)
# Visualize
wordcloud.to_image()

```

    /Users/jhaelle/opt/anaconda3/lib/python3.8/site-packages/ipykernel/ipkernel.py:287: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
      and should_run_async(code)





    
![png](output_7_1.png)
    



Removes the stopwords identified in the word cloud as unnecessary to create a string of words that will be most significant in developing topics


```python

#Remove Stopwords---------------------------------------------
nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(['georgia', 'section', 'b','c','shall','may' 'related','matters','code','annotated','relating','article','member','law','amended','state','chapter','title','subsection'])
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]
data = docs.Text_processed.values.tolist()
data_words = list(sent_to_words(data))
data_words = remove_stopwords(data_words)
print(data_words[:1][0][:30])

```

    /Users/jhaelle/opt/anaconda3/lib/python3.8/site-packages/ipykernel/ipkernel.py:287: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
      and should_run_async(code)
    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/jhaelle/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!


    ['passed', 'senate', 'bill', 'entitled', 'act', 'amend', 'official', 'general', 'provisions', 'regarding', 'parks', 'historic', 'areas', 'memorials', 'recreation', 'juvenile', 'programs', 'protection', 'children', 'youth', 'respectively', 'strengthen', 'laws', 'supports', 'foster', 'children', 'foster', 'families', 'provide', 'definitions']


A collection of the bill text was created, each word in the collection was given an ID number, and the frequency of each term was calculated. Then it printed out the the frequency of each ID number (ID number, frequency).


```python

#Create Corpus----------------------------------------------------
#Dictionary
id2word = corpora.Dictionary(data_words)
# Corpus
texts = data_words
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus[:1][0][:30])

```

    [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 6), (8, 1), (9, 5), (10, 1), (11, 1), (12, 1), (13, 1), (14, 2), (15, 2), (16, 4), (17, 1), (18, 4), (19, 1), (20, 3), (21, 2), (22, 13), (23, 2), (24, 1), (25, 1), (26, 4), (27, 7), (28, 6), (29, 1), (30, 1)]


    /Users/jhaelle/opt/anaconda3/lib/python3.8/site-packages/ipykernel/ipkernel.py:287: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
      and should_run_async(code)


To build the Latent Dirichlet allocation model the number of topics was set to 10 for simplicity, other parameters are default. Then, it printed the weight each keyword has within its topic.


```python

#Topic Modeling----------------------------------------------------
#LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=10)
# Prints weighted keywords
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]
```

    /Users/jhaelle/opt/anaconda3/lib/python3.8/site-packages/ipykernel/ipkernel.py:287: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
      and should_run_async(code)


    [(0,
      '0.029*"court" + 0.018*"shall" + 0.017*"juvenile" + 0.015*"child" + '
      '0.010*"judge" + 0.008*"provide" + 0.007*"care" + 0.007*"violation" + '
      '0.006*"judges" + 0.006*"services"'),
     (1,
      '0.016*"shall" + 0.013*"sexual" + 0.013*"offense" + 0.012*"employee" + '
      '0.012*"violation" + 0.011*"agent" + 0.009*"person" + 0.009*"child" + '
      '0.007*"court" + 0.007*"children"'),
     (2,
      '0.028*"child" + 0.020*"shall" + 0.010*"court" + 0.009*"violation" + '
      '0.009*"services" + 0.008*"juvenile" + 0.008*"school" + 0.008*"care" + '
      '0.008*"provide" + 0.007*"offense"'),
     (3,
      '0.028*"shall" + 0.022*"child" + 0.016*"court" + 0.010*"care" + '
      '0.008*"juvenile" + 0.008*"provide" + 0.007*"judge" + 0.007*"health" + '
      '0.007*"services" + 0.006*"provided"'),
     (4,
      '0.023*"shall" + 0.010*"committee" + 0.009*"child" + 0.008*"court" + '
      '0.007*"provide" + 0.006*"juvenile" + 0.005*"department" + 0.005*"services" '
      '+ 0.005*"health" + 0.005*"school"'),
     (5,
      '0.020*"child" + 0.016*"shall" + 0.015*"violation" + 0.010*"offense" + '
      '0.010*"court" + 0.009*"services" + 0.009*"care" + 0.007*"sexual" + '
      '0.007*"children" + 0.006*"age"'),
     (6,
      '0.024*"child" + 0.023*"shall" + 0.010*"department" + 0.009*"court" + '
      '0.009*"services" + 0.008*"school" + 0.007*"care" + 0.007*"children" + '
      '0.006*"provide" + 0.006*"juvenile"'),
     (7,
      '0.020*"shall" + 0.015*"court" + 0.012*"juvenile" + 0.012*"child" + '
      '0.009*"violation" + 0.008*"provide" + 0.007*"care" + 0.007*"judge" + '
      '0.007*"services" + 0.007*"provided"'),
     (8,
      '0.016*"shall" + 0.014*"child" + 0.012*"school" + 0.009*"provide" + '
      '0.009*"court" + 0.007*"care" + 0.007*"department" + 0.007*"health" + '
      '0.007*"services" + 0.006*"public"'),
     (9,
      '0.019*"shall" + 0.012*"child" + 0.010*"juvenile" + 0.010*"court" + '
      '0.008*"school" + 0.008*"care" + 0.007*"department" + 0.007*"services" + '
      '0.006*"violation" + 0.006*"health"')]


Similar to the process above, except this groups 2 and 3 word phrases together to be used for the Latent Dirichlet allocation model. To prepare these phrases for analysis, the stopwords are removed and some keywords are converted to their base form. The final keywords are printed in a string.


```python
#Phrase Modeling for Bigrams and Trigrams----------------------------------------------------------
# Build the models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) 
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

stop_words = stopwords.words('english')
stop_words.extend(['georgia', 'section', 'b','c' 'related','matters','code','annotated','relating','article','member','law','amended','state','chapter','title','subsection'])
# Define functions for stopwords, bigrams, trigrams and lemmatization
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]
def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove stopwords
data_words_nostops = remove_stopwords(data_words)
# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)
# Initialize spacy model
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
# Do lemmatization keeping only noun, adjective, verb and adverb
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
print(data_lemmatized[:1])
```

    /Users/jhaelle/opt/anaconda3/lib/python3.8/site-packages/ipykernel/ipkernel.py:287: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
      and should_run_async(code)


    [['pass', 'bill_entitled', 'act', 'amend_official', 'general', 'provision', 'regard', 'park', 'historic', 'area', 'memorial', 'recreation', 'juvenile', 'program', 'protection', 'child', 'youth', 'respectively', 'strengthen', 'law', 'support', 'foster', 'child', 'foster', 'family', 'provide', 'definition', 'provide', 'free', 'access', 'park', 'foster', 'parent', 'provide', 'report', 'certain', 'datum', 'juvenile', 'court', 'clerk', 'foster', 'child', 'allege', 'adjudicated', 'child', 'need', 'service', 'delinquent', 'child', 'provide', 'attorney', 'conflict', 'resolution', 'certain', 'juvenile', 'court', 'hearing', 'authorize', 'department', 'human', 'service', 'partner', 'child', 'place', 'agency', 'assist', 'casework', 'service', 'provide', 'varying', 'level', 'training', 'require', 'experienced', 'foster', 'parent', 'respite', 'caregiver', 'provide', 'conflicting_law', 'purposes_enacte', 'general_assembly', 'official', 'general', 'provision', 'regard', 'park', 'historic', 'area', 'memorial', 'recreation', 'use', 'term', 'fictive_kin', 'meaning_set', 'forth', 'foster', 'parent', 'meaning_set', 'fee', 'admission', 'park', 'historic', 'site', 'recreational', 'area', 'operate', 'pursuant', 'authority', 'department', 'waive', 'foster', 'parent', 'relative', 'fictive_kin', 'serve', 'primary', 'placement', 'child', 'temporary', 'permanent', 'custody', 'division', 'family', 'child', 'service', 'department', 'human', 'service', 'sb_csfa', 'official', 'juvenile', 'revise', 'collection', 'information', 'juvenile', 'court', 'clerk', 'report', 'requirement', 'datum', 'collection', 'follow', 'clerk', 'juvenile', 'court', 'collect', 'follow', 'information', 'child', 'need', 'service', 'delinquent', 'child', 'child', 'accuse', 'class', 'designate', 'felony', 'act', 'class', 'designate', 'felony', 'act', 'provide', 'information', 'djj', 'frequently', 'request', 'djj', 'name', 'date', 'birth', 'sex', 'race', 'offense', 'charge', 'location', 'offense', 'include', 'name', 'school', 'offense', 'occur', 'school', 'safety', 'zone', 'define', 'name', 'referral', 'source', 'include', 'name', 'school', 'refer', 'source', 'school', 'disposition', 'case', 'date', 'authority', 'commitment', 'applicable', 'clerk', 'juvenile', 'court', 'report', 'administrative', 'office', 'court', 'total', 'number', 'petition', 'motion', 'file', 'previous', 'calendar', 'year', 'number', 'number', 'court', 'appoint', 'litem', 'number', 'court', 'appoint', 'counsel', 'number', 'judge', 'issue', 'order', 'authorize', 'abortion', 'notification', 'number', 'judge', 'deny', 'order', 'last', 'number', 'denial', 'appeal', 'file', 'number', 'appeal', 'result', 'denial', 'affirm', 'number', 'appeal', 'result', 'reversal', 'denial', 'clerk', 'make', 'report', 'march', 'year', 'previous', 'calendar', 'year', 'individual', 'report', 'make', 'administrative', 'office', 'court', 'hold', 'confidential', 'disclosure', 'open', 'record', 'administrative', 'office', 'court', 'provide', 'aggregate', 'statistic', 'accordance', 'individual', 'report', 'destroy', 'month', 'submission', 'administrative', 'office', 'court', 'pursuant', 'rule', 'promulgate', 'adopt', 'judicial', 'council', 'court', 'january', 'clerk', 'juvenile', 'court', 'collect', 'datum', 'sb_csfa', 'child', 'allege', 'adjudicate', 'delinquent', 'child', 'transmit', 'datum', 'require', 'rule', 'judicial', 'council', 'supreme', 'court', 'make', 'publish', 'print', 'electronically', 'wide', 'minimum', 'standard', 'rule', 'deem', 'necessary', 'carry', 'clerk', 'juvenile', 'court', 'develop', 'enact', 'policy', 'procedure', 'necessary', 'carry', 'standard', 'rule', 'create', 'judicial', 'council', 'court', 'pursuant', 'rule', 'adopt', 'supreme', 'court', 'january', 'clerk', 'juvenile', 'court', 'collect', 'datum', 'case', 'child', 'allege', 'adjudicated', 'child', 'need', 'service', 'delinquent', 'child', 'place', 'foster', 'care', 'also', 'allege', 'child', 'transmit', 'datum', 'require', 'rule', 'datum', 'include', 'minimum', 'adherence', 'case', 'court', 'time', 'frame', 'contain', 'say', 'revise', 'continuance', 'hearing', 'dependency', 'proceeding', 'follow', 'stipulation', 'attorney', 'convenience', 'party', 'constitute_good', 'cause', 'otherwise', 'provide', 'judicial', 'rule', 'govern', 'attorney', 'conflict', 'resolution', 'pende', 'criminal', 'prosecution', 'family', 'matter', 'cause', 'hearing', 'dependency', 'case', 'time', 'limitation', 'require', 'termination_parental', 'right', 'hearing', 'take', 'priority', 'attorney', 'conflict', 'resolution', 'civil', 'criminal', 'hearing', 'nonjury', 'appearance', 'class', 'trial', 'court', 'need', 'discovery', 'cause', 'court', 'find', 'person', 'entity', 'fail', 'comply', 'order', 'discovery', 'official', 'program', 'protection', 'child', 'youth', 'power', 'duty', 'department', 'human', 'service', 'revise', 'follow', 'department', 'human', 'service', 'authorize', 'empowered', 'program', 'program', 'county', 'district', 'department', 'family', 'child', 'service', 'establish', 'maintain', 'extend', 'improve', 'limit', 'funds_appropriate', 'therefor', 'program', 'provide', 'preventive', 'service', 'follow', 'collect', 'disseminate', 'information', 'problem', 'child', 'youth', 'provide', 'consultative', 'assistance', 'group', 'public', 'private', 'interested', 'sb_csfa', 'develop', 'program', 'service', 'prevention', 'control', 'treatment', 'dependency', 'delinquency', 'child', 'research', 'demonstration', 'project', 'design', 'add', 'store', 'information', 'social', 'emotional', 'problem', 'child', 'youth', 'improve', 'method', 'deal', 'problem', 'child', 'welfare', 'service', 'follow', 'casework', 'service', 'child', 'youth', 'mothers_beare', 'child', 'wedlock', 'live', 'home', 'elsewhere', 'help', 'overcome', 'problem', 'result', 'dependency', 'delinquency', 'department', 'authorize', 'contract', 'certify', 'partner', 'license', 'child', 'place', 'agency', 'assist', 'provide', 'casework', 'service', 'protective', 'service', 'investigate', 'complaint', 'abuse', 'abandonment', 'child', 'youth', 'parent', 'guardian', 'custodian', 'person', 'serve', 'loco', 'parentis', 'basis', 'finding', 'investigation', 'offer', 'social', 'service', 'parent', 'guardian', 'custodian', 'person', 'serve', 'loco', 'parentis', 'relation', 'problem', 'bring', 'situation', 'attention', 'enforcement', 'agency', 'appropriate', 'court', 'community', 'agency', 'supervise', 'provide', 'require', 'service', 'care', 'involve', 'interstate', 'placement', 'child', 'homemaker', 'service', 'payment', 'cost', 'service', 'need', 'due', 'absence', 'incapacity', 'mother', 'boarding', 'care', 'payment_maintenance', 'cost', 'foster', 'family', 'home', 'group', 'care', 'facility', 'child', 'youth', 'adequately', 'care', 'home', 'boarding', 'care', 'payment_maintenance', 'cost', 'mothers_beare', 'child', 'wedlock', 'prior', 'reasonable', 'period', 'childbirth', 'day', 'care', 'service', 'care', 'protection', 'child', 'parent', 'absent', 'home', 'unable', 'reason', 'provide', 'parental', 'supervision', 'casework', 'service', 'care', 'child', 'youth', 'parent', 'custodian', 'guardian', 'place', 'child', 'custody', 'department', 'voluntary', 'agreement', 'agreement', 'revoke', 'parent', 'custodian', 'guardian', 'request', 'child', 'return', 'parent', 'custodian', 'guardian', 'relative', 'voluntary', 'agreement', 'expire', 'provide', 'however', 'subparagraph', 'prohibit', 'department', 'obtaining', 'order', 'place', 'child', 'custody', 'accordance', 'department', 'authorize', 'contract', 'certify', 'partner', 'license', 'child', 'place', 'agency', 'assist', 'provide', 'casework', 'service', 'sb_csfa', 'service', 'court', 'request', 'follow', 'accept', 'casework', 'service', 'care', 'child', 'youth', 'legal', 'custody', 'vest', 'department', 'court', 'provide', 'shelter', 'custodial', 'care', 'child', 'prior', 'examination', 'study', 'pende', 'court', 'hearing', 'make', 'social', 'study', 'report', 'court', 'respect', 'child', 'youth', 'petition', 'file', 'provide', 'casework', 'service', 'care', 'payment_maintenance', 'cost', 'child', 'youth', 'run', 'away', 'home', 'community', 'home', 'community', 'home', 'community', 'pay', 'cost', 'return', 'runaway', 'child', 'youth', 'home', 'community', 'provide', 'service', 'care', 'cost', 'runaway', 'child', 'youth', 'require', 'regional', 'group', 'care', 'facility', 'purpose', 'provide', 'local', 'authority', 'alternative', 'place', 'child', 'common', 'jail', 'shelter', 'care', 'prior', 'examination', 'study', 'pende', 'hear', 'juvenile', 'court', 'detention', 'prior', 'examination', 'study', 'pende', 'hear', 'juvenile', 'court', 'study', 'diagnosis', 'pende', 'determination', 'treatment', 'hear', 'juvenile', 'court', 'facility', 'design', 'afford', 'specialize', 'diversified', 'program', 'forestry', 'camp', 'ranch', 'group', 'residence', 'care', 'treatment', 'training', 'child', 'youth', 'different', 'age', 'different', 'emotional', 'mental', 'physical', 'condition', 'regulation', 'child', 'place', 'agency', 'child', 'care', 'institution', 'maternity', 'home', 'establish', 'rules_regulation', 'provide', 'consultation', 'rules_regulation', 'agency', 'institution', 'home', 'license', 'inspect', 'periodically', 'agency', 'institution', 'home', 'ensure', 'adherence', 'establish', 'standard', 'prescribe', 'department', 'adoption', 'service', 'follow', 'supervise', 'work', 'child', 'place', 'agency', 'fund', 'make', 'available', 'provide', 'service', 'parent', 'desire', 'surrender', 'child', 'adoption', 'provide', 'adoption', 'statute', 'provide', 'care', 'payment_maintenance', 'cost', 'mothers_beare', 'child', 'wedlock', 'child', 'consider', 'adoption', 'sb_csfa', 'inquire', 'character', 'reputation', 'person', 'make', 'application', 'adoption', 'child', 'place', 'child', 'adoption', 'provide', 'financial', 'assistance', 'family', 'adopt', 'child', 'child', 'place', 'adoption', 'determine', 'eligible', 'assistance', 'adoption', 'assistance', 'agreement', 'sign', 'prior', 'finalization', 'adoption', 'party', 'financial', 'assistance', 'grant', 'hard', 'place', 'child', 'physical', 'mental', 'emotional', 'disability', 'problem', 'difficult', 'find', 'permanent', 'home', 'financial', 'assistance', 'exceed', 'percent', 'amount', 'pay', 'boarding', 'child', 'family', 'foster', 'home', 'special', 'service', 'medical', 'care', 'available', 'insurance', 'public', 'facility', 'supplement', 'available', 'family', 'provide', 'child', 'adequately', 'continued', 'financial', 'assistance', 'department', 'review', 'supplement', 'pay', 'time', 'review', 'least', 'annually', 'determine', 'continued', 'assistance', 'provide', 'payment', 'license', 'child', 'place', 'agency', 'place', 'child', 'special', 'need', 'jurisdiction', 'department', 'adoption', 'payment', 'exceed', 'adoption', 'arrange', 'agency', 'board', 'define', 'special', 'need', 'child', 'half', 'payment', 'make', 'time', 'placement', 'remain', 'amount', 'pay', 'adoption', 'finalize', 'adoption', 'disrupt', 'prior', 'finalization', 'reimburse', 'child', 'place', 'agency', 'amount', 'calculate', 'prorated', 'basis', 'base', 'length', 'time', 'child', 'home', 'service', 'provide', 'provide', 'payment', 'agency', 'recruit', 'educate', 'train', 'potential', 'adoptive', 'foster', 'parent', 'preparation', 'anticipation', 'adopt', 'foster', 'special', 'need', 'child', 'board', 'define', 'special', 'need', 'child', 'set', 'payment', 'amount', 'rule', 'regulation', 'appropriate', 'documentation', 'preplacement', 'service', 'timely', 'manner', 'payment', 'set', 'board', 'make', 'enrollment', 'potential', 'adoptive', 'foster', 'parent', 'service', 'staff', 'development', 'recruitment', 'program', 'service', 'train', 'educational', 'scholarship', 'personnel', 'necessary', 'assure', 'efficient', 'effective', 'administration', 'service', 'care', 'child', 'youth', 'authorize', 'department', 'authorize', 'disburse', 'fund', 'match', 'federal', 'fund', 'order', 'provide', 'qualified', 'employee', 'graduate', 'postgraduate', 'educational', 'scholarship', 'accordance', 'rules_regulation', 'adopt', 'board', 'pursuant', 'viii', 'vii', 'paragraph', 'constitution', 'sb_csfa', 'miscellaneous', 'service', 'provide', 'medical', 'hospital', 'psychiatric', 'surgical', 'dental', 'service', 'payment', 'cost', 'service', 'consider', 'appropriate', 'necessary', 'competent', 'medical', 'authority', 'child', 'supervision', 'control', 'department', 'secure', 'prior', 'consent', 'parent', 'legal', 'guardian', 'preparation', 'education', 'training', 'foster', 'parent', 'provide', 'appropriate', 'knowledge', 'skill', 'provide', 'need', 'foster', 'child', 'include', 'knowledge', 'skill', 'reasonable', 'prudent', 'parent', 'standard', 'participation', 'child', 'age', 'developmentally', 'appropriate', 'activity', 'continue', 'preparation', 'necessary', 'placement', 'child', 'department', 'authorize', 'require', 'vary', 'level', 'initial', 'annual', 'training', 'base', 'experience', 'foster', 'parent', 'age', 'need', 'foster', 'child', 'child', 'foster', 'parent', 'provide', 'respite', 'care', 'part', 'training', 'offer', 'online', 'youth', 'leave', 'foster', 'care', 'reason', 'attain', 'child', 'foster', 'care', 'less', 'month', 'child', 'eligible', 'receive', 'document', 'official', 'birth', 'certificate', 'child', 'social', 'security', 'card', 'issue', 'commissioner', 'social', 'security', 'health', 'insurance', 'information', 'copy', 'child', 'medical', 'record', 'driver', 'license', 'identification', 'card', 'issue', 'accordance', 'requirement', 'real', 'd', 'act', 'official', 'documentation', 'necessary', 'prove', 'child', 'previously', 'foster', 'care', 'provision', 'record', 'accordance', 'paragraph', 'consider', 'violation', 'extend', 'care', 'youth', 'service', 'youth', 'receive', 'federal', 'reimbursement', 'provide', 'service', 'accordance', 'usc', 'exist', 'law', 'conflict', 'act_repeale']]


A collection of the phrases and keywords was created, each phrase/keyword in the collection was given an ID number, and the frequency of each was calculated. Then it printed out the the frequency of each ID number (ID number, frequency).


```python
#Create Corpus----------------------------------------------------
# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)
# Create Corpus
texts = data_lemmatized
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus[:2])
```

    [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 6), (8, 1), (9, 4), (10, 1), (11, 1), (12, 1), (13, 2), (14, 2), (15, 1), (16, 2), (17, 1), (18, 4), (19, 1), (20, 5), (21, 13), (22, 2), (23, 1), (24, 1), (25, 3), (26, 13), (27, 1), (28, 4), (29, 4), (30, 1), (31, 1), (32, 1), (33, 4), (34, 1), (35, 1), (36, 1), (37, 3), (38, 1), (39, 1), (40, 1), (41, 2), (42, 5), (43, 3), (44, 1), (45, 3), (46, 8), (47, 1), (48, 1), (49, 1), (50, 4), (51, 4), (52, 8), (53, 3), (54, 1), (55, 2), (56, 2), (57, 1), (58, 2), (59, 4), (60, 3), (61, 1), (62, 1), (63, 2), (64, 1), (65, 2), (66, 25), (67, 1), (68, 2), (69, 4), (70, 7), (71, 3), (72, 2), (73, 1), (74, 2), (75, 1), (76, 1), (77, 76), (78, 1), (79, 1), (80, 3), (81, 8), (82, 4), (83, 2), (84, 1), (85, 1), (86, 1), (87, 5), (88, 1), (89, 1), (90, 1), (91, 1), (92, 1), (93, 4), (94, 1), (95, 1), (96, 3), (97, 1), (98, 1), (99, 1), (100, 1), (101, 1), (102, 1), (103, 1), (104, 2), (105, 2), (106, 2), (107, 1), (108, 1), (109, 8), (110, 3), (111, 1), (112, 1), (113, 29), (114, 1), (115, 2), (116, 1), (117, 5), (118, 4), (119, 1), (120, 2), (121, 7), (122, 1), (123, 1), (124, 1), (125, 3), (126, 1), (127, 2), (128, 4), (129, 1), (130, 3), (131, 1), (132, 1), (133, 17), (134, 4), (135, 2), (136, 2), (137, 1), (138, 1), (139, 1), (140, 1), (141, 2), (142, 2), (143, 1), (144, 1), (145, 1), (146, 2), (147, 1), (148, 1), (149, 1), (150, 1), (151, 2), (152, 1), (153, 1), (154, 1), (155, 1), (156, 1), (157, 1), (158, 2), (159, 1), (160, 2), (161, 1), (162, 1), (163, 1), (164, 1), (165, 1), (166, 2), (167, 1), (168, 1), (169, 1), (170, 2), (171, 1), (172, 3), (173, 1), (174, 1), (175, 1), (176, 1), (177, 1), (178, 1), (179, 1), (180, 3), (181, 3), (182, 2), (183, 1), (184, 1), (185, 1), (186, 1), (187, 2), (188, 4), (189, 1), (190, 8), (191, 2), (192, 1), (193, 2), (194, 2), (195, 3), (196, 2), (197, 1), (198, 4), (199, 2), (200, 1), (201, 8), (202, 1), (203, 1), (204, 21), (205, 1), (206, 1), (207, 1), (208, 3), (209, 1), (210, 2), (211, 1), (212, 1), (213, 1), (214, 1), (215, 4), (216, 6), (217, 1), (218, 1), (219, 1), (220, 3), (221, 6), (222, 1), (223, 3), (224, 1), (225, 14), (226, 1), (227, 1), (228, 1), (229, 4), (230, 1), (231, 2), (232, 1), (233, 4), (234, 2), (235, 6), (236, 1), (237, 1), (238, 1), (239, 3), (240, 2), (241, 1), (242, 1), (243, 1), (244, 1), (245, 1), (246, 3), (247, 1), (248, 2), (249, 2), (250, 4), (251, 1), (252, 13), (253, 2), (254, 1), (255, 2), (256, 1), (257, 1), (258, 2), (259, 1), (260, 1), (261, 2), (262, 5), (263, 1), (264, 1), (265, 1), (266, 1), (267, 1), (268, 1), (269, 2), (270, 1), (271, 8), (272, 1), (273, 1), (274, 1), (275, 1), (276, 1), (277, 2), (278, 4), (279, 2), (280, 2), (281, 1), (282, 2), (283, 1), (284, 2), (285, 1), (286, 3), (287, 1), (288, 4), (289, 6), (290, 11), (291, 1), (292, 1), (293, 9), (294, 1), (295, 1), (296, 3), (297, 2), (298, 4), (299, 5), (300, 1), (301, 1), (302, 1), (303, 5), (304, 1), (305, 1), (306, 2), (307, 18), (308, 1), (309, 2), (310, 4), (311, 1), (312, 1), (313, 3), (314, 2), (315, 1), (316, 4), (317, 8), (318, 4), (319, 5), (320, 1), (321, 1), (322, 1), (323, 2), (324, 4), (325, 1), (326, 2), (327, 2), (328, 15), (329, 4), (330, 1), (331, 1), (332, 2), (333, 1), (334, 3), (335, 1), (336, 1), (337, 1), (338, 1), (339, 2), (340, 1), (341, 1), (342, 1), (343, 7), (344, 1), (345, 1), (346, 6), (347, 1), (348, 1), (349, 8), (350, 1), (351, 1), (352, 1), (353, 1), (354, 1), (355, 3), (356, 1), (357, 1), (358, 35), (359, 3), (360, 1), (361, 1), (362, 2), (363, 1), (364, 1), (365, 1), (366, 4), (367, 1), (368, 1), (369, 1), (370, 1), (371, 2), (372, 2), (373, 2), (374, 3), (375, 2), (376, 1), (377, 1), (378, 1), (379, 1), (380, 1), (381, 2), (382, 1), (383, 2), (384, 1), (385, 1), (386, 1), (387, 2), (388, 1), (389, 7), (390, 1), (391, 3), (392, 7), (393, 2), (394, 1), (395, 1), (396, 3), (397, 1), (398, 1), (399, 2), (400, 3), (401, 2), (402, 1), (403, 2), (404, 3), (405, 1), (406, 1), (407, 8), (408, 3), (409, 1), (410, 2), (411, 1), (412, 1), (413, 6), (414, 2), (415, 4), (416, 1), (417, 2), (418, 3), (419, 40), (420, 2), (421, 1), (422, 2), (423, 1), (424, 1), (425, 1), (426, 2), (427, 5), (428, 2), (429, 5), (430, 1), (431, 1), (432, 4), (433, 1), (434, 1), (435, 1), (436, 1), (437, 1), (438, 5), (439, 1), (440, 1), (441, 2), (442, 2), (443, 2), (444, 1), (445, 2), (446, 1), (447, 1), (448, 1), (449, 1), (450, 1), (451, 1), (452, 1), (453, 5), (454, 1), (455, 1), (456, 2), (457, 5), (458, 2), (459, 3), (460, 1), (461, 1), (462, 1), (463, 1), (464, 1), (465, 1), (466, 1), (467, 1), (468, 1), (469, 1), (470, 2), (471, 1), (472, 3), (473, 1), (474, 1), (475, 1), (476, 3), (477, 18), (478, 1)], [(9, 3), (10, 1), (19, 1), (25, 12), (26, 1), (30, 2), (32, 3), (37, 1), (40, 4), (41, 4), (46, 4), (50, 6), (51, 1), (53, 3), (55, 2), (58, 6), (65, 3), (69, 1), (72, 3), (73, 9), (74, 1), (76, 3), (77, 15), (79, 1), (91, 2), (92, 1), (93, 3), (94, 1), (95, 3), (101, 2), (105, 5), (106, 1), (108, 2), (112, 2), (113, 20), (115, 1), (117, 6), (118, 1), (120, 6), (122, 1), (125, 3), (126, 1), (130, 1), (133, 5), (136, 1), (137, 4), (141, 1), (142, 1), (161, 1), (165, 1), (176, 1), (180, 3), (187, 2), (190, 10), (191, 1), (192, 3), (195, 8), (198, 4), (199, 3), (200, 1), (201, 8), (210, 1), (211, 1), (214, 2), (216, 1), (219, 6), (221, 2), (227, 1), (229, 3), (230, 1), (233, 17), (234, 2), (235, 8), (243, 1), (245, 4), (246, 10), (249, 7), (252, 1), (253, 2), (254, 1), (255, 2), (256, 6), (258, 4), (259, 1), (262, 13), (264, 3), (265, 2), (266, 1), (271, 2), (278, 1), (280, 1), (282, 1), (284, 2), (288, 6), (289, 3), (295, 2), (296, 1), (299, 2), (303, 14), (304, 1), (306, 5), (307, 7), (308, 3), (311, 3), (313, 1), (314, 14), (315, 1), (321, 1), (324, 16), (326, 20), (332, 1), (339, 2), (340, 1), (342, 1), (343, 3), (346, 1), (348, 2), (349, 4), (350, 1), (352, 1), (356, 3), (357, 1), (358, 29), (359, 2), (362, 7), (365, 1), (366, 12), (371, 1), (373, 1), (374, 2), (381, 2), (386, 1), (387, 1), (389, 1), (392, 9), (393, 4), (395, 1), (404, 9), (406, 4), (408, 2), (412, 7), (415, 6), (417, 1), (418, 2), (419, 3), (422, 1), (423, 2), (426, 2), (427, 8), (429, 1), (444, 5), (449, 1), (453, 2), (456, 2), (471, 1), (476, 2), (479, 4), (480, 1), (481, 1), (482, 3), (483, 3), (484, 5), (485, 1), (486, 1), (487, 4), (488, 2), (489, 3), (490, 1), (491, 1), (492, 1), (493, 1), (494, 1), (495, 1), (496, 10), (497, 4), (498, 2), (499, 1), (500, 2), (501, 3), (502, 1), (503, 1), (504, 1), (505, 1), (506, 2), (507, 1), (508, 1), (509, 1), (510, 2), (511, 6), (512, 1), (513, 1), (514, 1), (515, 1), (516, 2), (517, 1), (518, 8), (519, 1), (520, 1), (521, 1), (522, 1), (523, 1), (524, 2), (525, 1), (526, 5), (527, 6), (528, 1), (529, 1), (530, 1), (531, 1), (532, 1), (533, 5), (534, 1), (535, 2), (536, 2), (537, 3), (538, 1), (539, 1), (540, 4), (541, 1), (542, 1), (543, 2), (544, 1), (545, 5), (546, 1), (547, 1), (548, 1), (549, 1), (550, 2), (551, 1), (552, 1), (553, 2), (554, 4), (555, 4), (556, 1), (557, 3), (558, 1), (559, 4), (560, 1), (561, 27), (562, 1), (563, 1), (564, 5), (565, 1), (566, 1), (567, 1), (568, 3), (569, 7), (570, 1), (571, 4), (572, 3), (573, 2), (574, 2), (575, 4), (576, 2), (577, 1), (578, 1), (579, 1), (580, 2), (581, 1), (582, 1), (583, 2), (584, 1), (585, 1), (586, 1), (587, 1), (588, 1), (589, 1), (590, 1), (591, 7), (592, 1), (593, 1), (594, 3), (595, 2), (596, 2), (597, 1), (598, 3), (599, 1), (600, 4), (601, 1), (602, 1), (603, 1), (604, 1), (605, 1), (606, 1), (607, 2), (608, 1), (609, 1), (610, 1), (611, 1), (612, 4), (613, 2), (614, 2), (615, 31), (616, 13), (617, 5), (618, 3), (619, 1), (620, 1), (621, 1), (622, 1), (623, 1), (624, 1), (625, 2), (626, 2), (627, 1), (628, 2), (629, 1), (630, 3), (631, 2), (632, 1), (633, 1), (634, 4), (635, 2), (636, 2), (637, 1), (638, 1), (639, 3), (640, 1), (641, 1), (642, 3), (643, 31), (644, 4), (645, 1), (646, 1), (647, 1), (648, 1), (649, 1), (650, 1), (651, 24), (652, 2), (653, 1), (654, 2), (655, 1), (656, 1), (657, 1), (658, 2), (659, 5), (660, 8), (661, 1), (662, 2), (663, 4), (664, 2), (665, 4), (666, 1), (667, 2), (668, 1), (669, 1), (670, 1), (671, 2), (672, 1), (673, 1), (674, 1), (675, 1), (676, 1), (677, 5), (678, 1), (679, 1), (680, 4), (681, 1), (682, 1), (683, 1), (684, 2), (685, 1), (686, 1), (687, 2), (688, 1), (689, 1), (690, 3), (691, 1), (692, 1), (693, 1), (694, 1), (695, 4), (696, 1), (697, 1), (698, 2), (699, 1), (700, 3), (701, 1), (702, 1), (703, 2), (704, 4), (705, 1), (706, 1), (707, 1), (708, 8), (709, 1), (710, 1), (711, 1), (712, 1), (713, 2), (714, 1), (715, 4), (716, 4), (717, 1), (718, 1), (719, 2), (720, 1), (721, 1), (722, 3), (723, 1), (724, 1), (725, 2), (726, 1), (727, 1), (728, 5), (729, 12), (730, 1), (731, 1), (732, 1), (733, 2), (734, 1), (735, 1), (736, 1), (737, 1), (738, 5), (739, 3)]]


    /Users/jhaelle/opt/anaconda3/lib/python3.8/site-packages/ipykernel/ipkernel.py:287: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
      and should_run_async(code)


To build the Latent Dirichlet allocation model for phrases the general parameters and the number of topics were maintained. The chunksize determines how many documents are processed at one time, 100 was chosen for efficancy and speed. The passes determines how often the model is trained on the documents, 10 was chosen to maintain efficancy while using the highest amount possible. The model also computed a list of topics in descending order of most likely topics for each word with their phi values multiplied by the word count


```python
#LDA model--------------------------------------------------------
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=10, 
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)
```

    /Users/jhaelle/opt/anaconda3/lib/python3.8/site-packages/ipykernel/ipkernel.py:287: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
      and should_run_async(code)


It printed the weight each phrase/keyword has within its topic.


```python
#Print the Keyword in the 10 topics-------------------------------
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]
```

    [(0,
      '0.053*"violation" + 0.044*"offense" + 0.039*"sexual" + 0.021*"person" + '
      '0.020*"child" + 0.017*"individual" + 0.017*"provide" + 0.013*"contact" + '
      '0.013*"aggravate" + 0.010*"convict"'),
     (1,
      '0.041*"child" + 0.020*"family" + 0.018*"treatment" + 0.012*"duty" + '
      '0.012*"program" + 0.012*"qualified_residential" + 0.012*"service" + '
      '0.011*"provide" + 0.011*"need" + 0.010*"officer"'),
     (2,
      '0.022*"provide" + 0.019*"child" + 0.014*"care" + 0.012*"court" + '
      '0.010*"woman" + 0.009*"person" + 0.008*"health" + 0.008*"mean" + '
      '0.007*"include" + 0.007*"order"'),
     (3,
      '0.082*"school" + 0.032*"safety" + 0.027*"public" + 0.026*"provide" + '
      '0.022*"agency" + 0.018*"plan" + 0.015*"board" + 0.015*"local" + '
      '0.015*"education" + 0.011*"student"'),
     (4,
      '0.001*"court" + 0.001*"provide" + 0.001*"child" + 0.001*"service" + '
      '0.001*"judge" + 0.001*"juvenile" + 0.001*"care" + 0.001*"health" + '
      '0.001*"school" + 0.001*"follow"'),
     (5,
      '0.032*"committee" + 0.016*"judge" + 0.014*"service" + 0.011*"member" + '
      '0.011*"resolution" + 0.011*"health" + 0.011*"report" + 0.009*"chairperson" '
      '+ 0.008*"mental" + 0.008*"court"'),
     (6,
      '0.050*"court" + 0.050*"judge" + 0.040*"juvenile" + 0.016*"provide" + '
      '0.014*"commission" + 0.014*"department" + 0.013*"afterschool" + '
      '0.012*"appoint" + 0.011*"time" + 0.011*"health"'),
     (7,
      '0.025*"department" + 0.024*"benefit" + 0.022*"care" + 0.022*"medicaid" + '
      '0.018*"cmo" + 0.018*"pharmacy" + 0.018*"provide" + 0.015*"program" + '
      '0.013*"pay" + 0.011*"cmos"'),
     (8,
      '0.056*"child" + 0.029*"service" + 0.025*"provide" + 0.020*"care" + '
      '0.020*"court" + 0.015*"foster" + 0.014*"youth" + 0.013*"parent" + '
      '0.011*"department" + 0.011*"place"'),
     (9,
      '0.068*"child" + 0.025*"court" + 0.024*"placement" + 0.015*"provide" + '
      '0.013*"care" + 0.012*"include" + 0.011*"change" + 0.011*"service" + '
      '0.011*"dfc" + 0.010*"foster"')]


    /Users/jhaelle/opt/anaconda3/lib/python3.8/site-packages/ipykernel/ipkernel.py:287: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
      and should_run_async(code)


The Perplexity was computed to understand the accuracy of the model. The lower the perplexity is, the better.
The Coherence Score was computed to measure the relative distance between keywords within a topic. 


```python
# Perplexity and Coherence Score--------------------------------------------------
# Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  
# a measure of how good the model is. lower the better.

# Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
```

    
    Perplexity:  -6.446167795299904


    /Users/jhaelle/opt/anaconda3/lib/python3.8/site-packages/ipykernel/ipkernel.py:287: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
      and should_run_async(code)


    
    Coherence Score:  0.4241751167352971


To have a better understanding of the hyperparameters to use, the algorithm will perform multiple sensitivity tests to determine the number of topics, the value of alpha(document-topic density) and the value of beta (word-topic density).


```python
def compute_coherence_values(corpus, dictionary, k, a, b):
    
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    
    return coherence_model_lda.get_coherence()
```

    /Users/jhaelle/opt/anaconda3/lib/python3.8/site-packages/ipykernel/ipkernel.py:287: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
      and should_run_async(code)


The function is called, and iterated over the range of topics, alpha, and beta parameter values. The range for topics was determined by subjective reasoning, anything less than 2 would create broad topics while anything over 11 would create unnecessary topics that does not give useful information because the same keywords will be repeated in multiple topics. After the iterations, the results are saved in a .csv file.


```python
grid = {}
grid['Validation_Set'] = {}
# Topics range
min_topics = 2
max_topics = 11
step_size = 1
topics_range = range(min_topics, max_topics, step_size)
# Alpha parameter
alpha = list(np.arange(0.01, 1, 0.3))
alpha.append('symmetric')
alpha.append('asymmetric')
# Beta parameter
beta = list(np.arange(0.01, 1, 0.3))
beta.append('symmetric')
# Validation sets
num_of_docs = len(corpus)
corpus_sets = [ gensim.utils.ClippedCorpus(corpus, int(num_of_docs*0.75)), 
               corpus]
corpus_title = ['75% Corpus', '100% Corpus']
model_results = {'Validation_Set': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []
                }

if 1 == 1:
    pbar = tqdm.tqdm(total=540)
    
    # Iterate through validation corpuses
    for i in range(len(corpus_sets)):
        # Iterate through number of topics
        for k in topics_range:
            # Iterate through alpha values
            for a in alpha:
                # Iterate through beta values
                for b in beta:
                    # Coherence score for the given parameters
                    cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word, 
                                                  k=k, a=a, b=b)
                    # Save model 
                    model_results['Validation_Set'].append(corpus_title[i])
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Beta'].append(b)
                    model_results['Coherence'].append(cv)
                    
                    pbar.update(1)
    pd.DataFrame(model_results).to_csv('lda_tuning_results.csv', index=False)
    pbar.close()

```

    /Users/jhaelle/opt/anaconda3/lib/python3.8/site-packages/ipykernel/ipkernel.py:287: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
      and should_run_async(code)
    
      0%|          | 0/540 [00:00<?, ?it/s][A
      0%|          | 1/540 [00:07<1:04:03,  7.13s/it][A
      0%|          | 2/540 [00:13<1:02:13,  6.94s/it][A
      1%|          | 3/540 [00:20<1:02:23,  6.97s/it][A
      1%|          | 4/540 [00:27<1:02:05,  6.95s/it][A
      1%|          | 5/540 [00:34<1:03:10,  7.08s/it][A
      1%|          | 6/540 [00:41<1:01:27,  6.91s/it][A
      1%|▏         | 7/540 [00:47<58:16,  6.56s/it]  [A
      1%|▏         | 8/540 [00:53<56:56,  6.42s/it][A
      2%|▏         | 9/540 [00:59<55:37,  6.29s/it][A
      2%|▏         | 10/540 [01:05<54:57,  6.22s/it][A
      2%|▏         | 11/540 [01:11<54:13,  6.15s/it][A
      2%|▏         | 12/540 [01:17<53:12,  6.05s/it][A
      2%|▏         | 13/540 [01:22<52:32,  5.98s/it][A
      3%|▎         | 14/540 [01:29<53:15,  6.08s/it][A
      3%|▎         | 15/540 [01:35<52:46,  6.03s/it][A
      3%|▎         | 16/540 [01:41<52:20,  5.99s/it][A
      3%|▎         | 17/540 [01:46<51:43,  5.93s/it][A
      3%|▎         | 18/540 [01:52<51:12,  5.89s/it][A
      4%|▎         | 19/540 [01:58<51:26,  5.92s/it][A
      4%|▎         | 20/540 [02:04<51:43,  5.97s/it][A
      4%|▍         | 21/540 [02:10<52:18,  6.05s/it][A
      4%|▍         | 22/540 [02:16<51:37,  5.98s/it][A
      4%|▍         | 23/540 [02:22<51:09,  5.94s/it][A
      4%|▍         | 24/540 [02:28<51:12,  5.95s/it][A
      5%|▍         | 25/540 [02:34<51:15,  5.97s/it][A
      5%|▍         | 26/540 [02:40<51:58,  6.07s/it][A
      5%|▌         | 27/540 [02:47<53:22,  6.24s/it][A
      5%|▌         | 28/540 [02:53<52:23,  6.14s/it][A
      5%|▌         | 29/540 [02:59<52:00,  6.11s/it][A
      6%|▌         | 30/540 [03:05<51:42,  6.08s/it][A
      6%|▌         | 31/540 [03:11<51:15,  6.04s/it][A
      6%|▌         | 32/540 [03:17<51:00,  6.02s/it][A
      6%|▌         | 33/540 [03:23<50:52,  6.02s/it][A
      6%|▋         | 34/540 [03:29<50:28,  5.99s/it][A
      6%|▋         | 35/540 [03:35<50:48,  6.04s/it][A
      7%|▋         | 36/540 [03:42<52:15,  6.22s/it][A
      7%|▋         | 37/540 [03:47<50:56,  6.08s/it][A
      7%|▋         | 38/540 [03:53<50:21,  6.02s/it][A
      7%|▋         | 39/540 [03:59<49:39,  5.95s/it][A
      7%|▋         | 40/540 [04:05<49:18,  5.92s/it][A
      8%|▊         | 41/540 [04:12<52:16,  6.29s/it][A
      8%|▊         | 42/540 [04:18<51:43,  6.23s/it][A
      8%|▊         | 43/540 [04:24<51:23,  6.20s/it][A
      8%|▊         | 44/540 [04:30<50:44,  6.14s/it][A
      8%|▊         | 45/540 [04:36<50:00,  6.06s/it][A
      9%|▊         | 46/540 [04:42<49:58,  6.07s/it][A
      9%|▊         | 47/540 [09:07<11:26:43, 83.58s/it][A
      9%|▉         | 48/540 [09:15<8:19:39, 60.93s/it] [A
      9%|▉         | 49/540 [09:21<6:05:02, 44.61s/it][A
      9%|▉         | 50/540 [09:28<4:30:20, 33.10s/it][A
      9%|▉         | 51/540 [09:34<3:23:37, 24.98s/it][A
     10%|▉         | 52/540 [09:40<2:37:00, 19.30s/it][A
     10%|▉         | 53/540 [09:46<2:05:44, 15.49s/it][A
     10%|█         | 54/540 [09:52<1:42:27, 12.65s/it][A
     10%|█         | 55/540 [09:59<1:27:59, 10.89s/it][A
     10%|█         | 56/540 [10:05<1:15:54,  9.41s/it][A
     11%|█         | 57/540 [10:11<1:07:32,  8.39s/it][A
     11%|█         | 58/540 [10:17<1:02:08,  7.74s/it][A
     11%|█         | 59/540 [10:23<58:11,  7.26s/it]  [A
     11%|█         | 60/540 [10:29<55:12,  6.90s/it][A
     11%|█▏        | 61/540 [10:35<52:59,  6.64s/it][A
     11%|█▏        | 62/540 [10:41<51:22,  6.45s/it][A
     12%|█▏        | 63/540 [10:48<50:49,  6.39s/it][A
     12%|█▏        | 64/540 [10:54<50:06,  6.32s/it][A
     12%|█▏        | 65/540 [11:00<49:52,  6.30s/it][A
     12%|█▏        | 66/540 [11:06<49:18,  6.24s/it][A
     12%|█▏        | 67/540 [11:12<48:44,  6.18s/it][A
     13%|█▎        | 68/540 [11:19<50:00,  6.36s/it][A
     13%|█▎        | 69/540 [11:25<49:04,  6.25s/it][A
     13%|█▎        | 70/540 [11:31<48:33,  6.20s/it][A
     13%|█▎        | 71/540 [24:17<30:30:24, 234.17s/it][A
     13%|█▎        | 72/540 [24:30<21:48:36, 167.77s/it][A
     14%|█▎        | 73/540 [24:41<15:39:08, 120.66s/it][A
     14%|█▎        | 74/540 [24:48<11:13:28, 86.71s/it] [A
     14%|█▍        | 75/540 [24:55<8:05:22, 62.63s/it] [A
     14%|█▍        | 76/540 [25:01<5:53:30, 45.71s/it][A
     14%|█▍        | 77/540 [25:07<4:21:48, 33.93s/it][A
     14%|█▍        | 78/540 [25:14<3:18:33, 25.79s/it][A
     15%|█▍        | 79/540 [25:21<2:33:19, 19.95s/it][A
     15%|█▍        | 80/540 [25:27<2:01:26, 15.84s/it][A
     15%|█▌        | 81/540 [25:33<1:38:33, 12.88s/it][A
     15%|█▌        | 82/540 [25:39<1:22:53, 10.86s/it][A
     15%|█▌        | 83/540 [25:45<1:11:58,  9.45s/it][A
     16%|█▌        | 84/540 [25:51<1:04:46,  8.52s/it][A
     16%|█▌        | 85/540 [25:57<58:59,  7.78s/it]  [A
     16%|█▌        | 86/540 [26:03<54:50,  7.25s/it][A
     16%|█▌        | 87/540 [26:10<52:00,  6.89s/it][A
     16%|█▋        | 88/540 [26:16<50:09,  6.66s/it][A
     16%|█▋        | 89/540 [26:22<49:33,  6.59s/it][A
     17%|█▋        | 90/540 [26:28<48:33,  6.47s/it][A
     17%|█▋        | 91/540 [26:35<48:56,  6.54s/it][A
     17%|█▋        | 92/540 [26:41<47:48,  6.40s/it][A
     17%|█▋        | 93/540 [26:47<47:08,  6.33s/it][A
     17%|█▋        | 94/540 [26:53<46:40,  6.28s/it][A
     18%|█▊        | 95/540 [26:59<46:09,  6.22s/it][A
     18%|█▊        | 96/540 [27:05<45:28,  6.15s/it][A
     18%|█▊        | 97/540 [31:41<10:41:53, 86.94s/it][A
     18%|█▊        | 98/540 [31:52<7:53:01, 64.21s/it] [A
     18%|█▊        | 99/540 [31:59<5:45:52, 47.06s/it][A
     19%|█▊        | 100/540 [32:06<4:15:48, 34.88s/it][A
     19%|█▊        | 101/540 [32:13<3:15:29, 26.72s/it][A
     19%|█▉        | 102/540 [32:20<2:30:22, 20.60s/it][A
     19%|█▉        | 103/540 [32:26<1:59:05, 16.35s/it][A
     19%|█▉        | 104/540 [32:32<1:37:02, 13.35s/it][A
     19%|█▉        | 105/540 [32:39<1:21:41, 11.27s/it][A
     20%|█▉        | 106/540 [32:45<1:10:41,  9.77s/it][A
     20%|█▉        | 107/540 [32:52<1:03:22,  8.78s/it][A
     20%|██        | 108/540 [32:58<58:11,  8.08s/it]  [A
     20%|██        | 109/540 [33:04<54:27,  7.58s/it][A
     20%|██        | 110/540 [33:11<52:28,  7.32s/it][A
     21%|██        | 111/540 [33:17<50:11,  7.02s/it][A
     21%|██        | 112/540 [33:24<49:03,  6.88s/it][A
     21%|██        | 113/540 [33:30<48:08,  6.76s/it][A
     21%|██        | 114/540 [33:37<47:15,  6.66s/it][A
     21%|██▏       | 115/540 [33:43<46:11,  6.52s/it][A
     21%|██▏       | 116/540 [33:49<45:24,  6.42s/it][A
     22%|██▏       | 117/540 [33:56<45:18,  6.43s/it][A
     22%|██▏       | 118/540 [34:03<46:12,  6.57s/it][A
      0%|          | 0/540 [1:13:33<?, ?it/s].45s/it][A
      0%|          | 0/540 [58:41<?, ?it/s]
    
     22%|██▏       | 120/540 [50:08<34:06:22, 292.34s/it][A
     22%|██▏       | 121/540 [50:21<24:14:54, 208.34s/it][A
     23%|██▎       | 122/540 [50:29<17:14:20, 148.47s/it][A
     23%|██▎       | 123/540 [50:36<12:16:06, 105.91s/it][A
     23%|██▎       | 124/540 [50:42<8:47:06, 76.03s/it]  [A
     23%|██▎       | 125/540 [50:49<6:21:17, 55.13s/it][A
     23%|██▎       | 126/540 [50:55<4:39:03, 40.44s/it][A
     24%|██▎       | 127/540 [51:02<3:29:02, 30.37s/it][A
     24%|██▎       | 128/540 [51:08<2:39:32, 23.23s/it][A
     24%|██▍       | 129/540 [51:15<2:04:39, 18.20s/it][A
     24%|██▍       | 130/540 [51:21<1:39:53, 14.62s/it][A
     24%|██▍       | 131/540 [51:27<1:22:48, 12.15s/it][A
     24%|██▍       | 132/540 [51:34<1:11:45, 10.55s/it][A
     25%|██▍       | 133/540 [51:41<1:03:06,  9.30s/it][A
     25%|██▍       | 134/540 [51:47<56:50,  8.40s/it]  [A
     25%|██▌       | 135/540 [51:53<52:23,  7.76s/it][A
     25%|██▌       | 136/540 [52:00<49:39,  7.38s/it][A
     25%|██▌       | 137/540 [52:06<48:31,  7.22s/it][A
     26%|██▌       | 138/540 [52:13<46:46,  6.98s/it][A
     26%|██▌       | 139/540 [52:19<45:19,  6.78s/it][A
     26%|██▌       | 140/540 [52:26<44:25,  6.66s/it][A
     26%|██▌       | 141/540 [52:32<43:54,  6.60s/it][A
     26%|██▋       | 142/540 [52:40<45:54,  6.92s/it][A
     26%|██▋       | 143/540 [52:46<43:57,  6.64s/it][A
     27%|██▋       | 144/540 [52:52<42:38,  6.46s/it][A
     27%|██▋       | 145/540 [54:03<2:50:03, 25.83s/it][A
     27%|██▋       | 146/540 [54:13<2:18:22, 21.07s/it][A
     27%|██▋       | 147/540 [54:20<1:50:17, 16.84s/it][A
     27%|██▋       | 148/540 [54:26<1:29:24, 13.69s/it][A
     28%|██▊       | 149/540 [54:33<1:15:32, 11.59s/it][A
     28%|██▊       | 150/540 [54:41<1:08:19, 10.51s/it][A
     28%|██▊       | 151/540 [54:50<1:06:02, 10.19s/it][A
     28%|██▊       | 152/540 [54:57<58:27,  9.04s/it]  [A
     28%|██▊       | 153/540 [55:03<53:18,  8.26s/it][A
     29%|██▊       | 154/540 [55:09<49:37,  7.71s/it][A
     29%|██▊       | 155/540 [55:16<47:15,  7.37s/it][A
     29%|██▉       | 156/540 [55:23<46:27,  7.26s/it][A
     29%|██▉       | 157/540 [55:30<46:01,  7.21s/it][A
     29%|██▉       | 158/540 [55:37<44:55,  7.06s/it][A
     29%|██▉       | 159/540 [55:43<43:34,  6.86s/it][A
     30%|██▉       | 160/540 [55:50<42:56,  6.78s/it][A
     30%|██▉       | 161/540 [55:57<43:41,  6.92s/it][A
     30%|███       | 162/540 [56:04<43:14,  6.86s/it][A
     30%|███       | 163/540 [56:10<42:40,  6.79s/it][A
     30%|███       | 164/540 [56:17<42:07,  6.72s/it][A
     31%|███       | 165/540 [1:00:01<7:29:08, 71.86s/it][A
     31%|███       | 166/540 [1:00:10<5:30:46, 53.07s/it][A
     31%|███       | 167/540 [1:00:17<4:03:29, 39.17s/it][A
     31%|███       | 168/540 [1:00:23<3:02:08, 29.38s/it][A
     31%|███▏      | 169/540 [1:00:30<2:18:57, 22.47s/it][A
     31%|███▏      | 170/540 [1:00:36<1:48:40, 17.62s/it][A
     32%|███▏      | 171/540 [1:00:42<1:27:36, 14.25s/it][A
     32%|███▏      | 172/540 [1:00:49<1:12:39, 11.85s/it][A
     32%|███▏      | 173/540 [1:00:55<1:02:57, 10.29s/it][A
     32%|███▏      | 174/540 [1:01:02<56:24,  9.25s/it]  [A
     32%|███▏      | 175/540 [1:01:08<50:47,  8.35s/it][A
     33%|███▎      | 176/540 [1:01:15<48:13,  7.95s/it][A
     33%|███▎      | 177/540 [1:01:22<45:05,  7.45s/it][A
     33%|███▎      | 178/540 [1:01:28<43:24,  7.19s/it][A
     33%|███▎      | 179/540 [1:01:36<43:47,  7.28s/it][A
     33%|███▎      | 180/540 [1:01:42<41:45,  6.96s/it][A
     34%|███▎      | 181/540 [1:01:48<40:55,  6.84s/it][A
     34%|███▎      | 182/540 [1:01:55<41:09,  6.90s/it][A
     34%|███▍      | 183/540 [1:02:02<40:14,  6.76s/it][A
     34%|███▍      | 184/540 [1:02:09<39:57,  6.74s/it][A
     34%|███▍      | 185/540 [1:02:59<1:57:31, 19.86s/it][A
     34%|███▍      | 186/540 [1:03:07<1:36:46, 16.40s/it][A
     35%|███▍      | 187/540 [1:03:14<1:20:00, 13.60s/it][A
     35%|███▍      | 188/540 [1:03:22<1:08:54, 11.75s/it][A
     35%|███▌      | 189/540 [1:03:28<59:19, 10.14s/it]  [A
     35%|███▌      | 190/540 [1:03:35<52:38,  9.02s/it][A
     35%|███▌      | 191/540 [1:03:42<48:45,  8.38s/it][A
     36%|███▌      | 192/540 [1:03:48<45:55,  7.92s/it][A
     36%|███▌      | 193/540 [1:03:55<43:55,  7.59s/it][A
     36%|███▌      | 194/540 [1:04:02<41:42,  7.23s/it][A
     36%|███▌      | 195/540 [1:04:08<40:36,  7.06s/it][A
     36%|███▋      | 196/540 [1:04:15<39:58,  6.97s/it][A
     36%|███▋      | 197/540 [1:04:22<39:30,  6.91s/it][A
     37%|███▋      | 198/540 [1:04:29<39:18,  6.90s/it][A
     37%|███▋      | 199/540 [1:04:35<38:15,  6.73s/it][A
     37%|███▋      | 200/540 [1:04:42<38:18,  6.76s/it][A
     37%|███▋      | 201/540 [1:04:49<38:09,  6.75s/it][A
     37%|███▋      | 202/540 [1:04:55<38:05,  6.76s/it][A
     38%|███▊      | 203/540 [1:05:02<37:51,  6.74s/it][A
     38%|███▊      | 204/540 [1:05:08<37:03,  6.62s/it][A
     38%|███▊      | 205/540 [1:05:15<36:47,  6.59s/it][A
     38%|███▊      | 206/540 [1:05:22<36:51,  6.62s/it][A
     38%|███▊      | 207/540 [1:05:28<36:58,  6.66s/it][A
     39%|███▊      | 208/540 [1:05:35<36:53,  6.67s/it][A
     39%|███▊      | 209/540 [1:05:42<37:28,  6.79s/it][A
     39%|███▉      | 210/540 [1:05:48<36:24,  6.62s/it][A
     39%|███▉      | 211/540 [1:05:55<36:29,  6.66s/it][A
     39%|███▉      | 212/540 [1:10:23<7:45:03, 85.07s/it][A
     39%|███▉      | 213/540 [1:10:31<5:37:12, 61.87s/it][A
     40%|███▉      | 214/540 [1:10:38<4:06:35, 45.38s/it][A
     40%|███▉      | 215/540 [1:10:45<3:03:28, 33.87s/it][A
     40%|████      | 216/540 [1:10:52<2:18:59, 25.74s/it][A
     40%|████      | 217/540 [1:10:59<1:48:13, 20.10s/it][A
     40%|████      | 218/540 [1:11:05<1:26:07, 16.05s/it][A
     41%|████      | 219/540 [1:11:12<1:10:57, 13.26s/it][A
     41%|████      | 220/540 [1:11:19<1:00:31, 11.35s/it][A
     41%|████      | 221/540 [1:11:26<53:01,  9.97s/it]  [A
     41%|████      | 222/540 [1:11:32<47:47,  9.02s/it][A
     41%|████▏     | 223/540 [1:11:39<43:39,  8.26s/it][A
     41%|████▏     | 224/540 [1:11:46<41:09,  7.82s/it][A
     42%|████▏     | 225/540 [1:11:52<39:29,  7.52s/it][A
     42%|████▏     | 226/540 [1:12:01<40:31,  7.74s/it][A
     42%|████▏     | 227/540 [1:12:07<38:37,  7.40s/it][A
     42%|████▏     | 228/540 [1:12:14<37:36,  7.23s/it][A
     42%|████▏     | 229/540 [1:12:21<36:41,  7.08s/it][A
     43%|████▎     | 230/540 [1:12:27<35:35,  6.89s/it][A
     43%|████▎     | 231/540 [1:12:34<35:00,  6.80s/it][A
     43%|████▎     | 232/540 [1:12:40<34:18,  6.68s/it][A
     43%|████▎     | 233/540 [1:12:47<33:30,  6.55s/it][A
     43%|████▎     | 234/540 [1:13:48<1:56:48, 22.90s/it][A
     44%|████▎     | 235/540 [1:13:54<1:31:53, 18.08s/it][A
     44%|████▎     | 236/540 [1:14:02<1:14:57, 14.80s/it][A
     44%|████▍     | 237/540 [1:14:08<1:02:17, 12.33s/it][A
     44%|████▍     | 238/540 [1:14:15<53:39, 10.66s/it]  [A
     44%|████▍     | 239/540 [1:14:21<47:21,  9.44s/it][A
     44%|████▍     | 240/540 [1:14:28<43:04,  8.62s/it][A
     45%|████▍     | 241/540 [1:14:35<40:34,  8.14s/it][A
     45%|████▍     | 242/540 [1:14:42<38:03,  7.66s/it][A
     45%|████▌     | 243/540 [1:14:48<36:20,  7.34s/it][A
     45%|████▌     | 244/540 [1:14:55<35:07,  7.12s/it][A
     45%|████▌     | 245/540 [1:15:02<34:13,  6.96s/it][A
     46%|████▌     | 246/540 [1:15:09<34:17,  7.00s/it][A
     46%|████▌     | 247/540 [1:15:15<33:50,  6.93s/it][A
     46%|████▌     | 248/540 [1:15:22<33:28,  6.88s/it][A
     46%|████▌     | 249/540 [1:15:29<32:51,  6.77s/it][A
     46%|████▋     | 250/540 [1:15:35<32:39,  6.76s/it][A
     46%|████▋     | 251/540 [1:15:43<33:35,  6.97s/it][A
     47%|████▋     | 252/540 [1:15:49<32:55,  6.86s/it][A
     47%|████▋     | 253/540 [1:15:56<32:20,  6.76s/it][A
     47%|████▋     | 254/540 [1:19:42<5:45:11, 72.42s/it][A
     47%|████▋     | 255/540 [1:19:50<4:12:04, 53.07s/it][A
     47%|████▋     | 256/540 [1:19:58<3:07:20, 39.58s/it][A
     48%|████▊     | 257/540 [1:20:05<2:20:53, 29.87s/it][A
     48%|████▊     | 258/540 [1:20:12<1:48:08, 23.01s/it][A
     48%|████▊     | 259/540 [1:20:19<1:25:30, 18.26s/it][A
     48%|████▊     | 260/540 [1:20:26<1:09:49, 14.96s/it][A
     48%|████▊     | 261/540 [1:20:33<58:16, 12.53s/it]  [A
     49%|████▊     | 262/540 [1:20:40<49:59, 10.79s/it][A
     49%|████▊     | 263/540 [1:20:46<43:53,  9.51s/it][A
     49%|████▉     | 264/540 [1:20:53<40:13,  8.74s/it][A
     49%|████▉     | 265/540 [1:21:01<37:59,  8.29s/it][A
     49%|████▉     | 266/540 [1:21:08<36:03,  7.90s/it][A
     49%|████▉     | 267/540 [1:21:15<35:07,  7.72s/it][A
     50%|████▉     | 268/540 [1:21:21<33:28,  7.39s/it][A
     50%|████▉     | 269/540 [1:21:28<32:26,  7.18s/it][A
     50%|█████     | 270/540 [1:21:35<32:11,  7.16s/it][A
     50%|█████     | 271/540 [1:21:41<30:36,  6.83s/it][A
     50%|█████     | 272/540 [1:21:47<29:23,  6.58s/it][A
     51%|█████     | 273/540 [1:21:53<28:26,  6.39s/it][A
     51%|█████     | 274/540 [1:22:02<30:51,  6.96s/it][A
     51%|█████     | 275/540 [1:22:07<29:14,  6.62s/it][A
     51%|█████     | 276/540 [1:22:13<28:16,  6.43s/it][A
     51%|█████▏    | 277/540 [1:23:10<1:33:50, 21.41s/it][A
     51%|█████▏    | 278/540 [1:23:17<1:14:41, 17.10s/it][A
     52%|█████▏    | 279/540 [1:23:23<1:00:14, 13.85s/it][A
     52%|█████▏    | 280/540 [1:23:30<50:31, 11.66s/it]  [A
     52%|█████▏    | 281/540 [1:23:36<43:16, 10.03s/it][A
     52%|█████▏    | 282/540 [1:23:42<38:12,  8.89s/it][A
     52%|█████▏    | 283/540 [1:23:49<35:23,  8.26s/it][A
     53%|█████▎    | 284/540 [1:23:55<32:36,  7.64s/it][A
     53%|█████▎    | 285/540 [1:24:01<30:45,  7.24s/it][A
     53%|█████▎    | 286/540 [1:24:07<29:10,  6.89s/it][A
     53%|█████▎    | 287/540 [1:24:14<28:09,  6.68s/it][A
     53%|█████▎    | 288/540 [1:24:20<27:55,  6.65s/it][A
     54%|█████▎    | 289/540 [1:24:26<27:17,  6.52s/it][A
     54%|█████▎    | 290/540 [1:24:33<26:53,  6.45s/it][A
     54%|█████▍    | 291/540 [1:24:39<26:21,  6.35s/it][A
     54%|█████▍    | 292/540 [1:24:45<25:56,  6.28s/it][A
     54%|█████▍    | 293/540 [1:24:51<26:02,  6.33s/it][A
     54%|█████▍    | 294/540 [1:24:58<25:55,  6.32s/it][A
     55%|█████▍    | 295/540 [1:25:04<25:45,  6.31s/it][A
     55%|█████▍    | 296/540 [1:25:10<25:27,  6.26s/it][A
     55%|█████▌    | 297/540 [1:25:16<25:11,  6.22s/it][A
     55%|█████▌    | 298/540 [1:25:23<25:19,  6.28s/it][A
     55%|█████▌    | 299/540 [1:25:29<25:27,  6.34s/it][A
     56%|█████▌    | 300/540 [1:25:35<25:03,  6.27s/it][A
     56%|█████▌    | 301/540 [1:25:41<24:27,  6.14s/it][A
     56%|█████▌    | 302/540 [1:26:34<1:19:28, 20.04s/it][A
     56%|█████▌    | 303/540 [1:26:41<1:04:19, 16.29s/it][A
     56%|█████▋    | 304/540 [1:26:48<52:32, 13.36s/it]  [A
     56%|█████▋    | 305/540 [1:26:54<43:52, 11.20s/it][A
     57%|█████▋    | 306/540 [1:27:00<37:56,  9.73s/it][A
     57%|█████▋    | 307/540 [1:27:06<33:50,  8.72s/it][A
     57%|█████▋    | 308/540 [1:27:13<31:05,  8.04s/it][A
     57%|█████▋    | 309/540 [1:27:19<28:57,  7.52s/it][A
     57%|█████▋    | 310/540 [1:27:25<27:12,  7.10s/it][A
     58%|█████▊    | 311/540 [1:27:32<26:08,  6.85s/it][A
     58%|█████▊    | 312/540 [1:27:38<25:31,  6.72s/it][A
     58%|█████▊    | 313/540 [1:27:45<25:16,  6.68s/it][A
     58%|█████▊    | 314/540 [1:27:51<24:53,  6.61s/it][A
     58%|█████▊    | 315/540 [1:27:57<24:18,  6.48s/it][A
     59%|█████▊    | 316/540 [1:28:03<23:53,  6.40s/it][A
     59%|█████▊    | 317/540 [1:28:10<24:04,  6.48s/it][A
     59%|█████▉    | 318/540 [1:28:17<24:30,  6.62s/it][A
     59%|█████▉    | 319/540 [1:28:24<25:13,  6.85s/it][A
     59%|█████▉    | 320/540 [1:28:31<24:54,  6.79s/it][A
     59%|█████▉    | 321/540 [1:28:37<24:12,  6.63s/it][A
     60%|█████▉    | 322/540 [1:28:44<23:45,  6.54s/it][A
     60%|█████▉    | 323/540 [1:28:51<24:18,  6.72s/it][A
     60%|██████    | 324/540 [1:28:57<23:48,  6.61s/it][A
     60%|██████    | 325/540 [1:29:03<23:10,  6.47s/it][A
     60%|██████    | 326/540 [2:29:09<64:34:38, 1086.35s/it][A
     61%|██████    | 327/540 [6:34:54<306:42:13, 5183.72s/it][A
     61%|██████    | 328/540 [6:35:02<213:50:04, 3631.15s/it][A
     61%|██████    | 329/540 [6:35:14<149:11:14, 2545.38s/it][A
     61%|██████    | 330/540 [7:13:36<144:12:49, 2472.24s/it][A
     61%|██████▏   | 331/540 [7:13:43<100:36:19, 1732.91s/it][A
     61%|██████▏   | 332/540 [7:13:51<70:13:01, 1215.30s/it] [A
     62%|██████▏   | 333/540 [7:13:58<49:02:38, 852.94s/it] [A
     62%|██████▏   | 334/540 [7:14:05<34:17:04, 599.15s/it][A
     62%|██████▏   | 335/540 [7:14:13<24:00:58, 421.75s/it][A
     62%|██████▏   | 336/540 [7:14:20<16:50:50, 297.31s/it][A
     62%|██████▏   | 337/540 [7:14:27<11:50:59, 210.15s/it][A
     63%|██████▎   | 338/540 [7:14:34<8:22:12, 149.17s/it] [A
     63%|██████▎   | 339/540 [7:14:40<5:56:19, 106.36s/it][A
     63%|██████▎   | 340/540 [7:14:47<4:15:09, 76.55s/it] [A
     63%|██████▎   | 341/540 [7:14:54<3:04:26, 55.61s/it][A
     63%|██████▎   | 342/540 [7:15:00<2:14:49, 40.85s/it][A
     64%|██████▎   | 343/540 [7:15:07<1:40:24, 30.58s/it][A
     64%|██████▎   | 344/540 [7:15:13<1:16:08, 23.31s/it][A
     64%|██████▍   | 345/540 [7:15:20<59:43, 18.38s/it]  [A
     64%|██████▍   | 346/540 [7:15:27<48:01, 14.85s/it][A
     64%|██████▍   | 347/540 [7:15:33<39:30, 12.28s/it][A
     64%|██████▍   | 348/540 [7:15:40<33:51, 10.58s/it][A
     65%|██████▍   | 349/540 [7:15:47<30:17,  9.51s/it][A
     65%|██████▍   | 350/540 [7:15:59<32:59, 10.42s/it][A
     65%|██████▌   | 351/540 [7:16:16<39:10, 12.44s/it][A
     65%|██████▌   | 352/540 [7:16:31<40:39, 12.98s/it][A
     65%|██████▌   | 353/540 [7:16:42<38:40, 12.41s/it][A
     66%|██████▌   | 354/540 [7:16:53<37:37, 12.14s/it][A
     66%|██████▌   | 355/540 [7:17:02<33:47, 10.96s/it][A
     66%|██████▌   | 356/540 [7:17:08<29:56,  9.76s/it][A
     66%|██████▌   | 357/540 [7:17:17<28:16,  9.27s/it][A
     66%|██████▋   | 358/540 [7:17:24<26:45,  8.82s/it][A
     66%|██████▋   | 359/540 [7:17:32<25:25,  8.43s/it][A
     67%|██████▋   | 360/540 [7:17:39<23:55,  7.98s/it][A
     67%|██████▋   | 361/540 [7:17:46<22:53,  7.67s/it][A
     67%|██████▋   | 362/540 [7:17:52<21:46,  7.34s/it][A
     67%|██████▋   | 363/540 [7:18:01<22:47,  7.72s/it][A
     67%|██████▋   | 364/540 [7:18:14<27:20,  9.32s/it][A
     68%|██████▊   | 365/540 [7:18:21<25:15,  8.66s/it][A
     68%|██████▊   | 366/540 [7:18:28<23:45,  8.19s/it][A
     68%|██████▊   | 367/540 [7:18:35<22:04,  7.66s/it][A
     68%|██████▊   | 368/540 [7:18:41<21:06,  7.36s/it][A
     68%|██████▊   | 369/540 [7:18:50<22:27,  7.88s/it][A
     69%|██████▊   | 370/540 [7:18:57<21:33,  7.61s/it][A
     69%|██████▊   | 371/540 [7:19:04<20:51,  7.40s/it][A
     69%|██████▉   | 372/540 [7:19:11<20:01,  7.15s/it][A
     69%|██████▉   | 373/540 [7:19:17<19:28,  6.99s/it][A
     69%|██████▉   | 374/540 [7:19:24<19:19,  6.99s/it][A
     69%|██████▉   | 375/540 [7:19:31<18:52,  6.87s/it][A
     70%|██████▉   | 376/540 [7:19:38<18:49,  6.88s/it][A
     70%|██████▉   | 377/540 [7:19:44<18:22,  6.76s/it][A
     70%|███████   | 378/540 [7:19:51<18:21,  6.80s/it][A
     70%|███████   | 379/540 [7:19:58<18:30,  6.90s/it][A
     70%|███████   | 380/540 [7:20:05<18:28,  6.93s/it][A
     71%|███████   | 381/540 [7:20:12<18:11,  6.86s/it][A
     71%|███████   | 382/540 [7:20:19<17:43,  6.73s/it][A
     71%|███████   | 383/540 [7:20:25<17:38,  6.74s/it][A
     71%|███████   | 384/540 [7:20:32<17:42,  6.81s/it][A
     71%|███████▏  | 385/540 [7:20:39<17:37,  6.82s/it][A
     71%|███████▏  | 386/540 [7:20:46<17:36,  6.86s/it][A
     72%|███████▏  | 387/540 [7:20:53<17:16,  6.78s/it][A
     72%|███████▏  | 388/540 [7:20:59<17:01,  6.72s/it][A
     72%|███████▏  | 389/540 [7:21:06<16:52,  6.70s/it][A
     72%|███████▏  | 390/540 [7:21:13<16:58,  6.79s/it][A
     72%|███████▏  | 391/540 [7:21:21<17:55,  7.22s/it][A
     73%|███████▎  | 392/540 [7:21:28<17:30,  7.10s/it][A
     73%|███████▎  | 393/540 [7:21:35<17:07,  6.99s/it][A
     73%|███████▎  | 394/540 [7:21:41<16:50,  6.92s/it][A
     73%|███████▎  | 395/540 [7:21:49<16:49,  6.96s/it][A
     73%|███████▎  | 396/540 [7:21:55<16:38,  6.93s/it][A
     74%|███████▎  | 397/540 [7:22:02<16:21,  6.86s/it][A
     74%|███████▎  | 398/540 [7:22:09<16:12,  6.85s/it][A
     74%|███████▍  | 399/540 [7:22:16<16:05,  6.85s/it][A
     74%|███████▍  | 400/540 [7:22:22<15:50,  6.79s/it][A
     74%|███████▍  | 401/540 [7:22:29<15:44,  6.80s/it][A
     74%|███████▍  | 402/540 [7:22:36<15:36,  6.79s/it][A
     75%|███████▍  | 403/540 [7:22:43<15:52,  6.95s/it][A
     75%|███████▍  | 404/540 [7:22:50<15:38,  6.90s/it][A
     75%|███████▌  | 405/540 [7:22:57<15:19,  6.81s/it][A
     75%|███████▌  | 406/540 [7:23:04<15:11,  6.80s/it][A
     75%|███████▌  | 407/540 [7:23:10<15:00,  6.77s/it][A
     76%|███████▌  | 408/540 [7:23:17<14:52,  6.76s/it][A
     76%|███████▌  | 409/540 [7:23:24<14:46,  6.77s/it][A
     76%|███████▌  | 410/540 [7:23:30<14:35,  6.73s/it][A
     76%|███████▌  | 411/540 [7:23:37<14:27,  6.72s/it][A
     76%|███████▋  | 412/540 [7:23:44<14:17,  6.70s/it][A
     76%|███████▋  | 413/540 [7:23:51<14:14,  6.73s/it][A
     77%|███████▋  | 414/540 [7:24:02<17:19,  8.25s/it][A
     77%|███████▋  | 415/540 [7:24:11<17:44,  8.51s/it][A
     77%|███████▋  | 416/540 [7:24:19<16:47,  8.12s/it][A
     77%|███████▋  | 417/540 [7:24:26<15:52,  7.74s/it][A
     77%|███████▋  | 418/540 [7:24:32<15:04,  7.41s/it][A
     78%|███████▊  | 419/540 [7:24:39<14:27,  7.17s/it][A
     78%|███████▊  | 420/540 [7:24:45<14:01,  7.01s/it][A
     78%|███████▊  | 421/540 [7:24:52<13:45,  6.94s/it][A
     78%|███████▊  | 422/540 [7:24:59<13:29,  6.86s/it][A
     78%|███████▊  | 423/540 [7:25:06<13:16,  6.81s/it][A
     79%|███████▊  | 424/540 [7:25:12<13:06,  6.78s/it][A
     79%|███████▊  | 425/540 [7:25:19<12:58,  6.77s/it][A
     79%|███████▉  | 426/540 [7:25:26<12:58,  6.83s/it][A
     79%|███████▉  | 427/540 [7:25:33<12:57,  6.88s/it][A
     79%|███████▉  | 428/540 [7:25:40<13:09,  7.05s/it][A
     79%|███████▉  | 429/540 [7:25:48<13:15,  7.17s/it][A
     80%|███████▉  | 430/540 [7:25:56<13:25,  7.33s/it][A
     80%|███████▉  | 431/540 [7:26:03<13:17,  7.32s/it][A
     80%|████████  | 432/540 [7:26:10<12:52,  7.15s/it][A
     80%|████████  | 433/540 [7:26:17<12:38,  7.08s/it][A
     80%|████████  | 434/540 [7:26:23<12:20,  6.99s/it][A
     81%|████████  | 435/540 [7:26:30<12:12,  6.98s/it][A
     81%|████████  | 436/540 [7:26:37<12:04,  6.97s/it][A
     81%|████████  | 437/540 [7:26:44<11:51,  6.91s/it][A
     81%|████████  | 438/540 [7:26:51<11:46,  6.92s/it][A
     81%|████████▏ | 439/540 [7:26:58<11:42,  6.96s/it][A
     81%|████████▏ | 440/540 [7:27:05<11:39,  6.99s/it][A
     82%|████████▏ | 441/540 [7:27:12<11:24,  6.92s/it][A
     82%|████████▏ | 442/540 [7:27:19<11:13,  6.87s/it][A
     82%|████████▏ | 443/540 [7:27:25<11:01,  6.82s/it][A
     82%|████████▏ | 444/540 [7:27:32<10:56,  6.84s/it][A
     82%|████████▏ | 445/540 [7:27:39<10:55,  6.90s/it][A
     83%|████████▎ | 446/540 [7:27:47<11:21,  7.25s/it][A
     83%|████████▎ | 447/540 [7:27:55<11:24,  7.37s/it][A
     83%|████████▎ | 448/540 [7:28:03<11:25,  7.45s/it][A
     83%|████████▎ | 449/540 [7:28:09<11:04,  7.31s/it][A
     83%|████████▎ | 450/540 [7:28:16<10:40,  7.12s/it][A
     84%|████████▎ | 451/540 [7:28:23<10:31,  7.09s/it][A
     84%|████████▎ | 452/540 [7:28:30<10:15,  6.99s/it][A
     84%|████████▍ | 453/540 [7:28:37<10:00,  6.90s/it][A
     84%|████████▍ | 454/540 [7:28:44<09:54,  6.91s/it][A
     84%|████████▍ | 455/540 [7:28:50<09:45,  6.89s/it][A
     84%|████████▍ | 456/540 [7:28:57<09:42,  6.94s/it][A
     85%|████████▍ | 457/540 [7:29:04<09:30,  6.88s/it][A
     85%|████████▍ | 458/540 [7:29:11<09:20,  6.83s/it][A
     85%|████████▌ | 459/540 [7:29:18<09:12,  6.82s/it][A
     85%|████████▌ | 460/540 [7:29:25<09:04,  6.81s/it][A
     85%|████████▌ | 461/540 [7:29:32<09:07,  6.93s/it][A
     86%|████████▌ | 462/540 [7:29:39<09:17,  7.15s/it][A
     86%|████████▌ | 463/540 [7:29:46<09:03,  7.06s/it][A
     86%|████████▌ | 464/540 [7:29:53<08:52,  7.01s/it][A
     86%|████████▌ | 465/540 [7:30:00<08:40,  6.94s/it][A
     86%|████████▋ | 466/540 [7:30:08<08:48,  7.15s/it][A
     86%|████████▋ | 467/540 [7:30:17<09:32,  7.84s/it][A
     87%|████████▋ | 468/540 [7:30:24<09:14,  7.70s/it][A
     87%|████████▋ | 469/540 [7:30:31<08:46,  7.42s/it][A
     87%|████████▋ | 470/540 [7:30:38<08:24,  7.21s/it][A
     87%|████████▋ | 471/540 [7:35:20<1:43:18, 89.84s/it][A
     87%|████████▋ | 472/540 [7:35:35<1:16:15, 67.28s/it][A
     88%|████████▊ | 473/540 [7:35:43<55:18, 49.53s/it]  [A
     88%|████████▊ | 474/540 [7:35:51<40:34, 36.88s/it][A
     88%|████████▊ | 475/540 [7:35:59<30:35, 28.24s/it][A
     88%|████████▊ | 476/540 [7:36:06<23:24, 21.94s/it][A
     88%|████████▊ | 477/540 [7:36:13<18:27, 17.57s/it][A
     89%|████████▊ | 478/540 [7:36:21<15:08, 14.66s/it][A
     89%|████████▊ | 479/540 [7:36:28<12:37, 12.42s/it][A
     89%|████████▉ | 480/540 [7:36:35<10:45, 10.76s/it][A
     89%|████████▉ | 481/540 [7:36:42<09:32,  9.71s/it][A
     89%|████████▉ | 482/540 [7:36:50<08:45,  9.06s/it][A
     89%|████████▉ | 483/540 [7:36:57<08:02,  8.46s/it][A
     90%|████████▉ | 484/540 [7:37:04<07:28,  8.01s/it][A
     90%|████████▉ | 485/540 [7:37:11<07:02,  7.68s/it][A
     90%|█████████ | 486/540 [7:37:20<07:12,  8.02s/it][A
     90%|█████████ | 487/540 [7:37:27<06:57,  7.89s/it][A
     90%|█████████ | 488/540 [7:37:34<06:35,  7.60s/it][A
     91%|█████████ | 489/540 [7:37:42<06:32,  7.69s/it][A
     91%|█████████ | 490/540 [7:37:49<06:11,  7.43s/it][A
     91%|█████████ | 491/540 [7:39:10<24:04, 29.47s/it][A
     91%|█████████ | 492/540 [7:39:23<19:34, 24.47s/it][A
     91%|█████████▏| 493/540 [7:39:32<15:40, 20.01s/it][A
     91%|█████████▏| 494/540 [7:39:40<12:33, 16.37s/it][A
     92%|█████████▏| 495/540 [7:39:48<10:15, 13.67s/it][A
     92%|█████████▏| 496/540 [7:39:55<08:36, 11.73s/it][A
     92%|█████████▏| 497/540 [7:40:02<07:22, 10.29s/it][A
     92%|█████████▏| 498/540 [7:40:09<06:32,  9.34s/it][A
     92%|█████████▏| 499/540 [7:40:16<05:57,  8.72s/it][A
     93%|█████████▎| 500/540 [7:40:23<05:28,  8.21s/it][A
     93%|█████████▎| 501/540 [7:40:30<05:06,  7.86s/it][A
     93%|█████████▎| 502/540 [7:40:38<04:56,  7.79s/it][A
     93%|█████████▎| 503/540 [7:40:45<04:40,  7.59s/it][A
     93%|█████████▎| 504/540 [7:40:52<04:26,  7.40s/it][A
     94%|█████████▎| 505/540 [7:40:59<04:14,  7.28s/it][A
     94%|█████████▎| 506/540 [7:41:06<04:05,  7.21s/it][A
     94%|█████████▍| 507/540 [7:41:13<03:56,  7.17s/it][A
     94%|█████████▍| 508/540 [7:41:20<03:49,  7.17s/it][A
     94%|█████████▍| 509/540 [7:41:27<03:40,  7.10s/it][A
     94%|█████████▍| 510/540 [7:41:34<03:31,  7.05s/it][A
     95%|█████████▍| 511/540 [7:41:43<03:42,  7.67s/it][A
     95%|█████████▍| 512/540 [7:41:50<03:25,  7.35s/it][A
     95%|█████████▌| 513/540 [7:43:27<15:29, 34.42s/it][A
     95%|█████████▌| 514/540 [7:43:36<11:35, 26.74s/it][A
     95%|█████████▌| 515/540 [7:43:43<08:42, 20.89s/it][A
     96%|█████████▌| 516/540 [7:43:51<06:46, 16.92s/it][A
     96%|█████████▌| 517/540 [7:43:58<05:20, 13.94s/it][A
     96%|█████████▌| 518/540 [7:44:05<04:23, 11.96s/it][A
     96%|█████████▌| 519/540 [7:44:13<03:41, 10.54s/it][A
     96%|█████████▋| 520/540 [7:44:20<03:09,  9.49s/it][A
     96%|█████████▋| 521/540 [7:44:27<02:48,  8.89s/it][A
     97%|█████████▋| 522/540 [7:44:34<02:29,  8.32s/it][A
     97%|█████████▋| 523/540 [7:44:41<02:14,  7.91s/it][A
     97%|█████████▋| 524/540 [7:44:48<02:02,  7.68s/it][A
     97%|█████████▋| 525/540 [7:44:56<01:54,  7.61s/it][A
     97%|█████████▋| 526/540 [7:45:03<01:45,  7.55s/it][A
     98%|█████████▊| 527/540 [7:45:10<01:35,  7.37s/it][A
     98%|█████████▊| 528/540 [7:45:17<01:27,  7.25s/it][A
     98%|█████████▊| 529/540 [7:45:24<01:19,  7.26s/it][A
     98%|█████████▊| 530/540 [7:45:32<01:13,  7.30s/it][A
     98%|█████████▊| 531/540 [7:45:39<01:06,  7.36s/it][A
     99%|█████████▊| 532/540 [7:45:46<00:56,  7.12s/it][A
     99%|█████████▊| 533/540 [7:45:53<00:49,  7.04s/it][A
     99%|█████████▉| 534/540 [7:48:27<05:08, 51.34s/it][A
     99%|█████████▉| 535/540 [7:48:36<03:12, 38.41s/it][A
     99%|█████████▉| 536/540 [7:48:44<01:57, 29.32s/it][A
     99%|█████████▉| 537/540 [7:48:51<01:08, 22.85s/it][A
    100%|█████████▉| 538/540 [7:48:59<00:36, 18.18s/it][A
    100%|█████████▉| 539/540 [7:49:06<00:14, 14.94s/it][A
    100%|██████████| 540/540 [7:49:13<00:00, 52.14s/it][A


This graph is a measure of the coherence score per number of topics. After 6 topics there is drop off in coherence scores, therefore the value of the parameter for number of topics is 6. The value of the alpha and beta parameter corresponds with the highest coherence score for 7 topics, which are .91 and .61 respectively.


```python
data=pd.read_csv('/Users/jhaelle/Desktop/PythonRB/LDAResults.csv')
plt.scatter(data.Topics, data.Coherence)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score")
plt.show()
```

    /Users/jhaelle/opt/anaconda3/lib/python3.8/site-packages/ipykernel/ipkernel.py:287: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
      and should_run_async(code)



    
![png](output_29_1.png)
    


This is the final Latent Dirichlet allocation model with all parameter values. 


```python
#Final Model----------------------------------------------
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=6, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=0.91,
                                           eta=0.61)
```

    /Users/jhaelle/opt/anaconda3/lib/python3.8/site-packages/ipykernel/ipkernel.py:287: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
      and should_run_async(code)


A visualization of the most relevant terms for each topic.


```python
#Visualize------------------------------------------------
pyLDAvis.enable_notebook()
LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
LDAvis_prepared
```

    /Users/jhaelle/opt/anaconda3/lib/python3.8/site-packages/ipykernel/ipkernel.py:287: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
      and should_run_async(code)






<link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.3.1/pyLDAvis/js/ldavis.v1.0.0.css">


<div id="ldavis_el74471405249460713282762657121"></div>
<script type="text/javascript">

var ldavis_el74471405249460713282762657121_data = {"mdsDat": {"x": [-0.10416464435579542, -0.09235192846227436, -0.023186158855970893, 0.07325667080079729, 0.07580300112024219, 0.07064305975300156], "y": [-0.014239290871599522, -0.03535923450152801, 0.08570182228068698, 0.06804534343003019, -0.0753932597232836, -0.02875538061430602], "topics": [1, 2, 3, 4, 5, 6], "cluster": [1, 1, 1, 1, 1, 1], "Freq": [31.96598031825035, 20.790188065816757, 16.508191052841344, 15.624482887018951, 7.898028348419849, 7.213129327652747]}, "tinfo": {"Term": ["child", "school", "violation", "court", "judge", "juvenile", "sexual", "offense", "judges", "woman", "whereas", "safety", "committee", "department", "medicaid", "afterschool", "public", "placement", "youth", "pharmacy", "security", "shall", "commission", "management", "marriage", "plan", "benefits", "house", "members", "dfcs", "violation", "sexual", "offense", "youths", "newborn", "contact", "placing", "rape", "molestation", "residential", "sodomy", "victim", "improper", "attempt", "commit", "obscene", "station", "aggravated", "casework", "ambulance", "adoption", "qualified", "viii", "incubators", "convictions", "payment", "homes", "elements", "consists", "indecent", "offenses", "agent", "children", "employee", "convicted", "foster", "individual", "providing", "services", "child", "facility", "placed", "home", "family", "treatment", "criminal", "shall", "care", "years", "provide", "parent", "department", "person", "age", "required", "court", "information", "act", "means", "laws", "placement", "official", "marriage", "petitioner", "emancipation", "change", "premarital", "petition", "evidence", "fact", "party", "father", "interests", "enter", "temporary", "dependent", "plan", "notice", "clergy", "recommendation", "intake", "sheet", "beneficiary", "registry", "license", "consider", "removal", "dfcs", "ordered", "living", "hearsay", "eighteenth", "protective", "legal", "order", "child", "rights", "placement", "caregiver", "abuse", "hearing", "attorney", "including", "court", "may", "limited", "said", "guardian", "permanency", "provided", "safe", "shall", "education", "follows", "age", "care", "program", "pursuant", "parent", "person", "juvenile", "services", "foster", "provide", "woman", "judges", "circuit", "penal", "postpartum", "associate", "prenatal", "pregnant", "gender", "female", "immediate", "appointment", "charge", "indemnification", "restraints", "women", "perinatal", "doula", "superior", "ethnicity", "color", "orientation", "expression", "institution", "religion", "origin", "labor", "full", "performing", "seminars", "judge", "pregnancy", "juvenile", "officer", "time", "provider", "court", "part", "property", "county", "shall", "duty", "compensation", "provided", "medical", "courts", "licensed", "health", "care", "provide", "term", "custodian", "follows", "revising", "act", "person", "laws", "services", "medicaid", "pharmacy", "cmo", "million", "cmos", "actuarial", "automatic", "gun", "filled", "affiliated", "sawed", "weapon", "prescription", "managed", "benefits", "machine", "semi", "shotgun", "actuary", "firearms", "appropriations", "budget", "forfeiture", "claims", "percentile", "pbm", "rifle", "ranked", "units", "organization", "commission", "drug", "behavioral", "system", "impact", "department", "management", "health", "members", "paid", "percent", "community", "general", "shall", "defined", "care", "provide", "assembly", "program", "official", "act", "laws", "afterschool", "homeland", "construction", "learning", "activity", "school", "schools", "plans", "capitol", "network", "play", "pm", "preparing", "unsafe", "suspicious", "critical", "statewide", "funding", "documents", "academic", "emergency", "improved", "security", "measures", "scholastic", "leaders", "architect", "affixed", "august", "displays", "every", "local", "help", "safety", "students", "public", "youth", "management", "whereas", "day", "board", "education", "agency", "programs", "shall", "agencies", "provide", "employee", "enforcement", "agent", "provided", "casa", "darden", "walker", "atlanta", "award", "abolishment", "whereas", "cares", "talents", "peggy", "violent", "gang", "citizens", "university", "task", "leadership", "outstanding", "vision", "douglas", "remarkable", "linnie", "adopts", "suggestions", "committee", "effort", "resolved", "duplication", "house", "resolution", "since", "representatives", "allowances", "chairperson", "legislation", "therefore", "victims", "trafficking", "organizations", "providers", "file", "members", "report", "judge", "mental", "clerk", "meetings", "shall", "children", "social", "health", "available", "authorized", "juvenile", "study", "court", "services", "purposes", "lc", "may", "violence"], "Freq": [400.0, 87.0, 156.0, 273.0, 86.0, 167.0, 119.0, 119.0, 52.0, 45.0, 36.0, 54.0, 41.0, 127.0, 37.0, 23.0, 62.0, 80.0, 44.0, 29.0, 29.0, 432.0, 44.0, 35.0, 35.0, 54.0, 31.0, 26.0, 50.0, 46.0, 151.46054859198657, 111.30197896312664, 110.90796270258723, 23.45238742501466, 21.87740250079358, 39.91514333416787, 20.187139010813343, 25.95722704607858, 18.490156495376812, 17.732511420376184, 17.72305173979956, 36.62542698144136, 23.2279285394268, 15.299394897564762, 15.978546918028629, 13.657536835256789, 12.824960980800416, 43.231770383244225, 11.982969649882262, 11.191719838009977, 21.981726133743553, 27.034704267320148, 10.306216282740992, 9.54050683965838, 9.472715024749949, 18.618490650359252, 11.173895009178626, 8.726573590157972, 8.724983202982607, 8.654393075412743, 16.384726036863295, 44.279969105114745, 87.29943673435322, 42.07198894399912, 25.43739242126675, 65.3489725193946, 48.36569778260848, 36.92719502005538, 96.64549351664873, 190.97321538241746, 32.10188537518793, 23.465285416958938, 29.5439239395897, 38.395225205534246, 36.21068860063026, 29.30552538248167, 132.21741343860222, 69.99755411311556, 39.77548523811321, 61.27516357756445, 36.32990517644661, 51.19454203397101, 42.96263266905726, 35.23973560849069, 30.00417141759176, 59.59957777563047, 31.401869012517682, 33.447370222810896, 31.33038134922468, 32.289207038552696, 31.860367545106346, 31.734160953390532, 33.18922621397174, 23.49763244321561, 22.03133851521563, 23.403683472987215, 18.316069238161248, 26.701749822989303, 25.876316370538508, 11.396116337539755, 10.07403332884858, 7.145519803190016, 11.492039036134178, 11.5793468291563, 9.386248992630303, 13.460423710733828, 40.87509696939256, 15.298628562230283, 6.835186954738266, 5.667347255391199, 5.6656353991548, 5.663139801508964, 5.660484205616864, 11.232129384995405, 15.1207530119184, 13.62699563494439, 7.156779054777726, 33.77792976121132, 6.057956723933529, 9.329103369001624, 4.921170121714474, 4.920147632921941, 9.245814281930597, 23.347519088603207, 36.102469282009295, 193.89293474585324, 11.573168146880636, 45.915646720090805, 15.385095114473218, 14.47171762597414, 20.939375484186872, 17.554331337545584, 35.95929335135601, 105.97479841368396, 47.237622229960266, 16.802395829486212, 24.036941531866244, 16.60359425854376, 20.067278793210317, 40.309738166272, 17.086357716989205, 91.78018400039052, 21.375193207202994, 31.238482596009042, 25.5718372079247, 40.595100038429806, 24.383095237686742, 21.047534298661628, 24.65695973586689, 26.70296730964447, 29.025414353378018, 25.428103783382088, 22.54215041622301, 22.313055427325725, 42.620584923691474, 49.48785103777186, 22.532044875733323, 19.789651690354656, 17.020321074627713, 17.773808841710117, 10.793399564950233, 17.74886640896685, 12.214599134924601, 8.026398927769442, 7.337211620807997, 12.210528744288704, 8.735699482973065, 6.634807206579314, 9.122317115854894, 5.953384013431407, 5.9513240047104095, 5.948075433983441, 17.980661309802375, 5.925164226413639, 5.923808918339188, 5.919128614256384, 5.916108326837453, 21.742950867813995, 5.915659204242294, 5.914739123257296, 7.392187700331797, 9.436988782868797, 7.341544537248137, 5.2729995093350635, 62.35735627454372, 10.122135475278, 92.01542738655264, 18.44552751292618, 34.079617057690065, 14.453172052003382, 94.23862095642262, 19.12589514390845, 9.481866303620357, 16.5984029020345, 115.94089967903354, 10.204293645304125, 10.268598075569571, 32.06256132990457, 17.84160699912464, 13.067284932173855, 17.13905221567349, 23.566910678156724, 30.522175044804474, 27.588363328029047, 15.679878213675352, 13.601724939834256, 18.314421431512592, 16.06549140000182, 16.883029508102467, 16.489341047888242, 15.767858600592522, 15.03593070314996, 34.75858211933779, 27.68623079274713, 22.239426335106685, 14.052610982773556, 14.052185432680783, 14.047236651469463, 17.536449621082884, 13.26023107360531, 12.686771792958004, 16.074469928851073, 11.322789805202964, 19.968018426678714, 9.963451292301295, 9.961109357684776, 26.341575167152392, 9.136245791434863, 8.600286586181381, 8.594814556842746, 8.593876092068617, 8.593356601558014, 11.344680740770045, 7.8947940482730745, 7.846340989862284, 9.983254871784178, 7.233649842218448, 7.233962663809823, 7.233120957666061, 7.231625426214197, 7.228969260900034, 9.843437711501727, 33.34594369216735, 16.87760601586093, 18.31152194767001, 16.158078779528722, 7.980317404666113, 55.866445393848466, 18.379792206924268, 34.81616383785127, 22.41461957793581, 14.770992146008231, 10.672347291547606, 14.291145431133364, 21.056648837517752, 51.601700729004726, 14.35288119492332, 27.967931802504307, 25.089356912703032, 14.950926561725268, 15.54725469768927, 16.044198568609318, 15.385597329665368, 15.142798925740227, 21.513291643227827, 10.894294418888041, 9.45454190811609, 6.908993053262201, 6.275546528372655, 59.19555246463681, 5.619816726465372, 7.8842337828261355, 4.879722834751833, 3.337543037342919, 3.337117562846736, 3.3369314177546556, 3.3367003983993064, 2.8287652884343415, 2.826943250887628, 3.375596650025269, 3.3422030752539613, 3.873496782545377, 5.428139041355051, 3.3668119787728794, 12.048962871979667, 3.339228552660109, 15.98446140313542, 3.337027102030304, 2.6615580851736995, 2.326167366638853, 2.324604365183287, 2.324658736645132, 2.3236620056546315, 2.32372848604466, 6.398084132993277, 10.501953749095101, 3.8471817073023677, 22.699108058487578, 4.391936268994794, 21.84305527160668, 15.900943038862314, 12.577504765983825, 12.532725474700259, 6.915196623371729, 10.547304771732914, 12.192925206866798, 11.13186494181491, 8.23065027404723, 25.472631941687563, 6.475069374366283, 11.769940876150091, 7.462776601640372, 5.627197949198265, 6.6193574631645555, 7.626889142745181, 7.709522187293111, 4.236206223385337, 4.238793565172999, 3.2710837179499386, 3.261120110189188, 3.2529248373495223, 21.51495866956347, 2.7766631287102888, 2.7669950746586047, 2.7664256512983636, 2.7376877693742245, 3.2735153682019367, 3.720595322251983, 3.1987545912862854, 2.2761304540726486, 2.2754233369447934, 2.275127566395148, 2.274679356918522, 2.2735988403137686, 2.2718959912475176, 2.2679416864503503, 2.266634571440626, 2.2651758848662378, 20.684552927433675, 2.2332732146927756, 5.242909861393671, 2.2102208927478935, 13.18253850054292, 9.682964877252685, 3.1015980856987193, 10.693350034609905, 3.253361771116044, 6.222741313818885, 3.2451762561024293, 3.2658290202009934, 3.9002533797414523, 5.751828628580426, 5.416779142788317, 4.574285839427261, 3.9357389068003936, 9.232402246664618, 7.156802056845869, 10.439023791131984, 5.451329219589802, 5.296606527632548, 4.228717850832948, 15.931287741528962, 7.645496970599162, 5.085253316403643, 7.023236500173745, 4.864388176339875, 4.740480265371647, 6.990908605625022, 4.669001370617935, 7.690662261101049, 6.717007729406923, 4.926654382257633, 4.775342592606775, 4.278134517725121, 4.267961658805714], "Total": [400.0, 87.0, 156.0, 273.0, 86.0, 167.0, 119.0, 119.0, 52.0, 45.0, 36.0, 54.0, 41.0, 127.0, 37.0, 23.0, 62.0, 80.0, 44.0, 29.0, 29.0, 432.0, 44.0, 35.0, 35.0, 54.0, 31.0, 26.0, 50.0, 46.0, 156.00381934621655, 119.19479729919132, 119.20969288486528, 25.49382715113123, 23.863575367768835, 43.68575676684335, 22.19888804266155, 28.723796337338033, 20.53064069713112, 19.73111391350635, 19.723129490503812, 40.824357732916226, 25.899328093532887, 17.26084939902799, 18.063446951835182, 15.611366835454948, 14.78622189239812, 50.1252940660634, 13.955398346320957, 13.13780084699541, 25.981606326250994, 32.16904318721372, 12.2927368007825, 11.485520500919593, 11.465628483492743, 22.62599821118195, 13.63405317871034, 10.662329445394683, 10.661762451499815, 10.63997538142513, 20.185475679016985, 56.18322872717938, 116.40017706867559, 54.97146920768582, 32.753691112651396, 91.21767381023572, 66.37167551878125, 51.86389346699854, 159.1438087996977, 400.57081695592166, 45.81006183314107, 31.206812179629523, 42.446598088467816, 62.59430702711625, 58.4081285266039, 43.1565673553158, 432.94411753024747, 173.47148113745342, 71.95637648286218, 149.515229109665, 64.62935683595693, 127.36288218098682, 95.83851007817957, 65.22078706112845, 46.781635473581595, 273.35097612358805, 59.5597171823952, 85.24841637430345, 64.64515217064891, 86.1580758190944, 80.70695729812832, 82.78248027543852, 35.27536581970062, 25.561144315698183, 24.07197207577949, 25.573965652806418, 20.335649636688686, 30.284183628225488, 30.199148579060697, 13.636521112601594, 12.111650345485533, 9.13330285020838, 14.86869565710125, 14.99461660524415, 12.197026957484557, 17.54520791166361, 54.00692393736079, 20.248657463026202, 9.166505393886256, 7.640188972783124, 7.6401407357112445, 7.639697414357102, 7.63944209587527, 15.294589038799256, 20.632806792683354, 18.608511558020556, 9.818608369189956, 46.758763664621405, 8.421168542216334, 13.025730873621086, 6.893337653276483, 6.89329106263058, 13.022962036109176, 34.69649517574316, 57.52896844578113, 400.57081695592166, 16.916046532472777, 80.70695729812832, 23.94209999558255, 22.3307695108716, 34.83394004395333, 28.512645371653875, 69.91567077736052, 273.35097612358805, 100.49236874170195, 27.367179955335658, 45.64055759424168, 27.113411044926988, 35.83118986398099, 110.17013037064322, 28.304504555670015, 432.94411753024747, 44.586759936568534, 95.25222327391762, 65.22078706112845, 173.47148113745342, 61.44746234898447, 44.81000921854183, 64.62935683595693, 95.83851007817957, 167.84608818449797, 159.1438087996977, 91.21767381023572, 149.515229109665, 45.485903093069375, 52.99373705035463, 24.61721900018323, 21.84379291942347, 19.06622313511708, 20.50382647657073, 12.821404484932678, 21.249335099614257, 14.731478880051212, 10.045673372497816, 9.350910357618467, 15.699545773565957, 11.242310099678823, 8.655831279738736, 12.199494881280708, 7.96446558553409, 7.963876351304952, 7.964421482406505, 24.12716396709437, 7.966330026039973, 7.965521989306475, 7.965859081492865, 7.966165807977346, 29.278827935344257, 7.967146877754292, 7.966714884408925, 10.036283617077176, 12.92592926502444, 10.097682059437055, 7.269702077160398, 86.97136856738837, 14.36609448144023, 167.84608818449797, 30.567566772713793, 65.43067124579828, 23.402978083897235, 273.35097612358805, 36.883044737240404, 14.799688683806673, 31.301739481582104, 432.94411753024747, 16.364952539678356, 17.03903984690175, 110.17013037064322, 43.35222812295147, 25.88277063832639, 47.23267804382035, 96.23718234020578, 173.47148113745342, 149.515229109665, 44.26869694323784, 31.734660784322166, 95.25222327391762, 61.57282610652986, 85.24841637430345, 95.83851007817957, 86.1580758190944, 159.1438087996977, 37.208811705441555, 29.79413801504913, 24.317668186832545, 16.103474931224657, 16.103903869564608, 16.10337733935704, 20.33084435499247, 15.393669823093578, 14.733889684173818, 18.967297250024192, 13.363655871538862, 23.607229931011826, 11.995096302385889, 11.996356792572636, 31.8928232497257, 11.28376401775014, 10.625541739122442, 10.624995195503164, 10.626596555649193, 10.626223191650444, 14.055618570094307, 9.937795783891614, 9.94428702728663, 12.690364907019452, 9.257128585921286, 9.25789859074866, 9.257386539330307, 9.256844480842128, 9.257264068603362, 12.663835880960795, 44.85026182850949, 22.42327005359042, 25.2504771733315, 23.595803055165238, 10.45882370868396, 127.36288218098682, 35.08905255026558, 96.23718234020578, 50.78380094292513, 26.025575196810934, 15.853339030043356, 28.57324495490494, 62.63107351128662, 432.94411753024747, 29.695047186351527, 173.47148113745342, 149.515229109665, 40.675745587347315, 61.44746234898447, 82.78248027543852, 85.24841637430345, 86.1580758190944, 23.72282003099389, 13.077542108892917, 12.220811979245147, 9.680076109971937, 9.246638461739215, 87.39626998373072, 8.521881388899699, 12.210237481490035, 7.977197179454836, 5.456940262236726, 5.456727344271406, 5.4569294993181385, 5.457228757588673, 4.952072749678399, 4.952957942818223, 5.956827110338077, 5.955664177688988, 6.965435705664296, 9.835811182796876, 6.13980222205783, 22.049454869558307, 6.201944087584025, 29.715372255814117, 6.204392292999025, 5.057817602257124, 4.442864313546948, 4.444202699138701, 4.444460469691528, 4.444544952416235, 4.44473831983122, 12.587067963709705, 22.302817297599344, 7.609688845395949, 54.17705342800719, 9.213543691296813, 62.21272503382913, 44.424702663145105, 35.08905255026558, 36.02137745084827, 16.34734133438823, 35.316112935436635, 44.586759936568534, 42.78181220439605, 35.31621123491905, 432.94411753024747, 27.2215823642639, 149.515229109665, 54.97146920768582, 25.947220992321018, 56.18322872717938, 110.17013037064322, 9.881078186864821, 6.402996303510271, 6.413456617176951, 5.404040647790434, 5.40852754785841, 5.402856862756325, 36.02137745084827, 4.9053942786811415, 4.907868631719463, 4.9114519081323404, 4.924878920260634, 6.086454444987844, 6.921197491590458, 6.123493516590607, 4.406556840929948, 4.409035131302402, 4.409630215299949, 4.410315821183736, 4.411728432046913, 4.409004967031363, 4.408602028471607, 4.406583928069379, 4.406348364799996, 41.00498767575351, 4.434786808538188, 10.4430646369842, 4.407699912065272, 26.290921052471894, 19.36025015594803, 6.237149860241992, 22.429754177470983, 6.776731084127038, 14.801909045254076, 6.7690064790447515, 6.925709612743928, 8.827411120619184, 15.08449420953665, 17.736941653470787, 13.564841232023934, 11.06398480183707, 50.78380094292513, 32.944391565817234, 86.97136856738837, 22.975599397310166, 22.052855348061993, 13.118595053881645, 432.94411753024747, 116.40017706867559, 30.065892247149137, 96.23718234020578, 29.96343053719613, 28.558170527261765, 167.84608818449797, 26.81285613606151, 273.35097612358805, 159.1438087996977, 54.00053037915158, 77.90205258938923, 100.49236874170195, 18.824809471473458], "Category": ["Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6"], "logprob": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -3.9982, -4.3063, -4.3098, -5.8636, -5.9331, -5.3318, -6.0135, -5.7621, -6.1013, -6.1431, -6.1437, -5.4178, -5.8732, -6.2907, -6.2473, -6.4042, -6.4671, -5.2519, -6.535, -6.6033, -5.9283, -5.7214, -6.6858, -6.763, -6.7701, -6.0944, -6.6049, -6.8522, -6.8523, -6.8605, -6.2222, -5.228, -4.5492, -5.2791, -5.7823, -4.8388, -5.1397, -5.4096, -4.4475, -3.7664, -5.5496, -5.863, -5.6326, -5.3706, -5.4292, -5.6407, -4.1341, -4.7701, -5.3353, -4.9031, -5.4259, -5.0829, -5.2582, -5.4563, -5.6172, -4.9309, -5.5717, -5.5085, -5.5739, -5.5438, -5.5572, -5.5611, -5.0861, -5.4314, -5.4959, -5.4354, -5.6806, -5.3036, -5.335, -6.1551, -6.2784, -6.6218, -6.1467, -6.1391, -6.3491, -5.9886, -4.8778, -5.8606, -6.6662, -6.8536, -6.8539, -6.8544, -6.8548, -6.1696, -5.8723, -5.9763, -6.6203, -5.0685, -6.787, -6.3552, -6.9948, -6.995, -6.3642, -5.4378, -5.002, -3.321, -6.1396, -4.7615, -5.8549, -5.9161, -5.5467, -5.723, -5.0059, -3.9251, -4.7331, -5.7668, -5.4087, -5.7787, -5.5892, -4.8917, -5.7501, -4.0689, -5.5261, -5.1467, -5.3468, -4.8847, -5.3944, -5.5415, -5.3833, -5.3036, -5.2202, -5.3525, -5.4729, -5.4832, -4.6054, -4.456, -5.2428, -5.3725, -5.5233, -5.48, -5.9788, -5.4814, -5.8551, -6.275, -6.3647, -5.8554, -6.1903, -6.4654, -6.147, -6.5737, -6.5741, -6.5746, -5.4684, -6.5785, -6.5787, -6.5795, -6.58, -5.2784, -6.5801, -6.5803, -6.3573, -6.1131, -6.3642, -6.6951, -4.2248, -6.043, -3.8358, -5.4429, -4.829, -5.6868, -3.8119, -5.4067, -6.1083, -5.5484, -3.6046, -6.0349, -6.0286, -4.89, -5.4762, -5.7876, -5.5163, -5.1979, -4.9393, -5.0403, -5.6053, -5.7475, -5.45, -5.581, -5.5314, -5.555, -5.5997, -5.6473, -4.7543, -4.9818, -5.2008, -5.6599, -5.6599, -5.6603, -5.4384, -5.7179, -5.7621, -5.5255, -5.8759, -5.3086, -6.0038, -6.004, -5.0315, -6.0904, -6.1509, -6.1515, -6.1516, -6.1517, -5.8739, -6.2365, -6.2426, -6.0018, -6.3239, -6.3239, -6.324, -6.3242, -6.3246, -6.0159, -4.7958, -5.4767, -5.3952, -5.5203, -6.2257, -4.2797, -5.3914, -4.7526, -5.193, -5.61, -5.935, -5.643, -5.2555, -4.3591, -5.6387, -4.9716, -5.0802, -5.5979, -5.5588, -5.5273, -5.5693, -5.5852, -4.5518, -5.2322, -5.374, -5.6876, -5.7838, -3.5396, -5.8942, -5.5556, -6.0354, -6.4152, -6.4154, -6.4154, -6.4155, -6.5806, -6.5813, -6.4039, -6.4138, -6.2663, -5.9289, -6.4065, -5.1315, -6.4147, -4.8488, -6.4154, -6.6416, -6.7762, -6.7769, -6.7769, -6.7773, -6.7773, -5.7645, -5.2689, -6.2731, -4.4981, -6.1407, -4.5366, -4.8541, -5.0886, -5.0921, -5.6867, -5.2646, -5.1196, -5.2107, -5.5126, -4.3829, -5.7525, -5.1549, -5.6105, -5.8929, -5.7305, -5.5888, -5.4873, -6.0861, -6.0855, -6.3446, -6.3477, -6.3502, -4.461, -6.5085, -6.512, -6.5122, -6.5226, -6.3439, -6.2159, -6.367, -6.7073, -6.7076, -6.7077, -6.7079, -6.7084, -6.7091, -6.7109, -6.7115, -6.7121, -4.5004, -6.7263, -5.8729, -6.7367, -4.9509, -5.2594, -6.3978, -5.1601, -6.3501, -5.7015, -6.3526, -6.3462, -6.1687, -5.7802, -5.8403, -6.0093, -6.1597, -5.307, -5.5617, -5.1842, -5.8339, -5.8627, -6.0879, -4.7615, -5.4956, -5.9034, -5.5805, -5.9478, -5.9736, -5.5851, -5.9888, -5.4897, -5.6251, -5.9351, -5.9663, -6.0762, -6.0786], "loglift": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.1109, 1.072, 1.0683, 1.057, 1.0536, 1.0502, 1.0455, 1.0392, 1.0358, 1.0337, 1.0336, 1.032, 1.0316, 1.0199, 1.0179, 1.0068, 0.9982, 0.9925, 0.9881, 0.9802, 0.9733, 0.9666, 0.9642, 0.955, 0.9496, 0.9456, 0.9415, 0.9402, 0.94, 0.9339, 0.9319, 0.9024, 0.8528, 0.8731, 0.8877, 0.807, 0.824, 0.8008, 0.6417, 0.3997, 0.7849, 0.8554, 0.7781, 0.6518, 0.6624, 0.7534, -0.0457, 0.2329, 0.5477, 0.2485, 0.5645, 0.2291, 0.3382, 0.5249, 0.6963, -0.3826, 0.5004, 0.2049, 0.4162, 0.159, 0.211, 0.1817, 1.5097, 1.4865, 1.4821, 1.482, 1.4661, 1.4448, 1.4162, 1.3912, 1.3865, 1.3252, 1.3131, 1.3122, 1.3087, 1.3057, 1.2921, 1.2904, 1.2772, 1.272, 1.2717, 1.2713, 1.2709, 1.262, 1.2599, 1.2591, 1.2545, 1.2455, 1.2413, 1.2369, 1.2337, 1.2335, 1.2281, 1.1745, 1.1048, 0.8451, 1.1911, 1.0067, 1.1284, 1.1369, 1.0617, 1.0856, 0.9058, 0.6231, 0.8158, 1.0829, 0.9295, 1.0803, 0.991, 0.5653, 1.0659, 0.0195, 0.8355, 0.4558, 0.6344, 0.1183, 0.6464, 0.815, 0.6071, 0.2928, -0.1842, -0.2633, 0.1728, -0.3315, 1.7362, 1.7329, 1.7128, 1.7026, 1.6878, 1.6584, 1.6291, 1.6213, 1.614, 1.5769, 1.5588, 1.55, 1.549, 1.5354, 1.5106, 1.5103, 1.51, 1.5094, 1.5073, 1.5053, 1.5052, 1.5043, 1.5038, 1.5037, 1.5036, 1.5035, 1.4955, 1.4867, 1.4826, 1.4802, 1.4686, 1.4512, 1.2002, 1.2962, 1.149, 1.3194, 0.7364, 1.1446, 1.3561, 1.1669, 0.4838, 1.329, 1.2949, 0.567, 0.9135, 1.1178, 0.7876, 0.3943, 0.0638, 0.1113, 0.7634, 0.9541, 0.1525, 0.4578, 0.1821, 0.0414, 0.1031, -0.5581, 1.7882, 1.783, 1.767, 1.7201, 1.72, 1.7197, 1.7085, 1.7071, 1.7067, 1.6908, 1.6906, 1.6889, 1.6708, 1.6704, 1.6651, 1.6452, 1.6449, 1.6443, 1.644, 1.644, 1.6421, 1.6262, 1.6194, 1.6164, 1.6097, 1.6096, 1.6096, 1.6094, 1.609, 1.6044, 1.5599, 1.5722, 1.535, 1.4777, 1.5859, 1.0323, 1.2097, 0.8396, 1.0385, 1.2899, 1.4606, 1.1635, 0.7663, -0.2707, 1.1293, 0.0314, 0.0714, 0.8555, 0.482, 0.2155, 0.1442, 0.1177, 2.4408, 2.3559, 2.2819, 2.2013, 2.151, 2.149, 2.1222, 2.1011, 2.0471, 2.0469, 2.0468, 2.0467, 2.0466, 1.9786, 1.9778, 1.9706, 1.9608, 1.9518, 1.9441, 1.9377, 1.9342, 1.9194, 1.9185, 1.9184, 1.8965, 1.8915, 1.8905, 1.8905, 1.89, 1.89, 1.8619, 1.7854, 1.8565, 1.6686, 1.7977, 1.4919, 1.5111, 1.5126, 1.4828, 1.6782, 1.3301, 1.242, 1.1923, 1.0821, -0.2944, 1.1025, -0.0033, 0.5417, 1.0101, 0.3999, -0.1318, 2.3811, 2.2162, 2.2151, 2.1272, 2.1234, 2.1219, 2.1139, 2.0602, 2.0562, 2.0553, 2.0421, 2.0091, 2.0086, 1.9799, 1.9687, 1.9678, 1.9675, 1.9672, 1.9664, 1.9662, 1.9646, 1.9645, 1.9639, 1.945, 1.9433, 1.9402, 1.939, 1.9389, 1.9364, 1.9307, 1.8885, 1.8955, 1.7627, 1.8941, 1.8775, 1.8124, 1.6651, 1.4431, 1.5422, 1.5957, 0.9244, 1.1025, 0.5092, 1.1907, 1.2029, 1.4971, -0.6731, -0.0936, 0.8522, 0.0117, 0.8112, 0.8335, -0.5492, 0.8813, -0.9415, -0.5359, 0.2349, -0.1627, -0.5273, 1.1452]}, "token.table": {"Topic": [1, 6, 1, 2, 3, 4, 6, 1, 3, 5, 1, 2, 3, 4, 5, 6, 1, 4, 5, 1, 4, 1, 4, 1, 2, 4, 1, 6, 1, 4, 1, 5, 1, 5, 1, 2, 3, 1, 2, 4, 5, 6, 1, 2, 3, 4, 5, 1, 2, 3, 5, 6, 1, 2, 3, 4, 1, 3, 4, 6, 1, 1, 2, 3, 1, 3, 4, 1, 5, 1, 2, 3, 4, 5, 1, 2, 3, 1, 6, 1, 1, 2, 4, 6, 1, 5, 1, 3, 4, 5, 6, 1, 4, 1, 2, 3, 4, 5, 6, 1, 6, 1, 4, 5, 6, 1, 2, 1, 3, 4, 1, 2, 3, 4, 5, 6, 1, 4, 1, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 1, 6, 1, 6, 1, 1, 4, 6, 1, 2, 1, 3, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 3, 1, 5, 6, 1, 3, 4, 1, 2, 1, 2, 3, 5, 6, 1, 4, 1, 4, 1, 3, 1, 2, 3, 4, 6, 1, 3, 1, 2, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 3, 4, 6, 1, 2, 4, 1, 1, 3, 5, 1, 3, 5, 1, 2, 3, 4, 1, 1, 2, 3, 5, 6, 1, 2, 3, 4, 5, 6, 1, 3, 4, 1, 2, 3, 4, 6, 1, 5, 6, 1, 2, 3, 1, 6, 1, 2, 3, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 1, 2, 1, 5, 1, 2, 5, 1, 6, 1, 3, 1, 2, 4, 6, 1, 6, 1, 2, 3, 4, 6, 1, 2, 3, 4, 5, 6, 1, 2, 6, 1, 2, 1, 1, 2, 1, 3, 5, 1, 3, 4, 5, 1, 2, 3, 4, 5, 6, 1, 2, 3, 1, 3, 1, 2, 3, 4, 5, 6, 1, 2, 4, 1, 3, 1, 3, 4, 5, 1, 2, 1, 2, 3, 4, 5, 6, 1, 2, 1, 3, 1, 2, 3, 6, 1, 4, 1, 4, 1, 2, 3, 4, 5, 6, 1, 4, 1, 2, 3, 4, 6, 1, 2, 3, 1, 5, 6, 1, 4, 6, 1, 3, 6, 1, 2, 3, 4, 5, 6, 1, 2, 1, 4, 1, 2, 3, 4, 5, 6, 1, 2, 4, 1, 2, 1, 5, 1, 2, 4, 1, 5, 1, 6, 1, 3, 4, 5, 6, 1, 3, 1, 4, 6, 1, 5, 1, 2, 5, 1, 2, 3, 4, 5, 6, 1, 1, 1, 3, 1, 2, 3, 4, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 5, 1, 2, 1, 2, 3, 6, 1, 2, 3, 4, 6, 1, 2, 3, 4, 6, 1, 2, 3, 4, 5, 6, 1, 3, 4, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 5, 1, 6, 1, 4, 5, 1, 2, 3, 4, 5, 6, 1, 4, 6, 1, 2, 1, 2, 3, 4, 5, 1, 2, 3, 5, 1, 6, 1, 2, 1, 2, 3, 4, 5, 6, 1, 4, 1, 4, 1, 2, 3, 4, 5, 6, 1, 2, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 1, 2, 5, 1, 4, 6, 1, 2, 3, 4, 6, 1, 3, 4, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 4, 1, 1, 5, 1, 1, 2, 3, 1, 1, 2, 3, 4, 5, 6, 1, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 1, 2, 1, 3, 4, 1, 4, 5, 6, 1, 3, 1, 3, 1, 6, 1, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5, 1, 2, 1, 3, 4, 1, 4, 1, 6, 1, 3, 1, 4, 5, 1, 4, 1, 2, 3, 1, 3, 1, 2, 1, 2, 3, 4, 5, 6, 1, 2, 6, 1, 2, 1, 4, 1, 2, 3, 4, 1, 2, 3, 1, 1, 2, 4, 5, 6, 1, 2, 3, 5, 1, 5, 1, 5, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 3, 1, 5, 1, 4, 1, 2, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 6, 1, 2, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 1, 2, 3, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 1, 2, 3, 4, 6, 1, 4, 1, 2, 1, 2, 1, 2, 1, 3, 1, 6, 1, 2, 4, 1, 2, 3, 4, 5, 6, 1, 3, 4, 5, 6, 1, 2, 3, 4, 5, 1, 2, 1, 5, 6, 1, 5, 6, 1, 2, 3, 1, 2, 3, 4, 5, 1, 4, 1, 2, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 4, 1, 5, 1, 2, 3, 4, 5, 6, 1, 4, 5, 6, 1, 2, 4, 5, 1, 4, 1, 3, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 1, 4, 1, 2, 6, 1, 2, 5, 6, 1, 1, 5, 6, 1, 1, 3, 4, 5, 1, 4, 5, 6, 1, 6, 1, 2, 3, 4, 1, 5, 1, 3, 4, 5, 6, 1, 6, 1, 6, 1, 2, 1, 2, 3, 4, 1, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 6, 1, 2, 3, 4, 5, 6, 1, 4, 1, 2, 4, 6, 1, 5, 1, 2, 3, 1, 2, 6, 1, 1, 2, 4, 6, 1, 2, 4, 5, 6, 1, 6, 1, 6, 1, 2, 6, 1, 3, 4, 1, 2, 5, 6, 1, 2, 3, 1, 3, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1], "Freq": [0.18508726501590853, 0.5552617950477255, 0.17912504081207878, 0.6269376428422757, 0.044781260203019695, 0.044781260203019695, 0.044781260203019695, 0.16287169583531597, 0.16287169583531597, 0.4886150875059479, 0.38710396513532463, 0.18768677097470285, 0.19941719416062179, 0.17595634778878394, 0.023460846371837856, 0.011730423185918928, 0.1081474099087798, 0.1081474099087798, 0.6488844594526788, 0.062098774618910293, 0.869382844664744, 0.09410350668374542, 0.8469315601537089, 0.8467528806243168, 0.03848876730110531, 0.07697753460221061, 0.2269331564593896, 0.4538663129187792, 0.05272232447344202, 0.8435571915750724, 0.2249991887247916, 0.4499983774495832, 0.04215350446083134, 0.9273770981382895, 0.536638724816309, 0.39864590986354387, 0.045997604984255064, 0.477562245502165, 0.14694222938528154, 0.036735557346320384, 0.2204133440779223, 0.07347111469264077, 0.4441139592036182, 0.16362093233817512, 0.046748837810907176, 0.07012325671636076, 0.2571186079599895, 0.7831518586028576, 0.017798905877337675, 0.0711956235093507, 0.12459234114136372, 0.017798905877337675, 0.857850328884405, 0.019950007648474537, 0.07980003059389815, 0.03990001529694907, 0.14756377191095485, 0.14756377191095485, 0.14756377191095485, 0.4426913157328646, 0.8372786380390048, 0.0636961103475838, 0.1273922206951676, 0.7643533241710055, 0.07114592609447074, 0.07114592609447074, 0.7826051870391781, 0.2250122390218166, 0.4500244780436332, 0.24584675352848656, 0.14750805211709195, 0.19667740282278925, 0.3687701302927298, 0.024584675352848657, 0.04877138426540421, 0.04877138426540421, 0.8778849167772759, 0.1850467206254033, 0.55514016187621, 0.8690186475321832, 0.2455051051474557, 0.6312988418077432, 0.03507215787820796, 0.10521647363462386, 0.2249949118989919, 0.4499898237979838, 0.49022748101582814, 0.10504874593196317, 0.17508124321993862, 0.07003249728797545, 0.17508124321993862, 0.04918634870934117, 0.885354276768141, 0.26699212528648636, 0.10012204698243239, 0.16687007830405398, 0.20024409396486478, 0.10012204698243239, 0.16687007830405398, 0.18489320635816406, 0.5546796190744921, 0.1188096359291172, 0.7128578155747032, 0.0792064239527448, 0.0396032119763724, 0.13089961118233037, 0.7853976670939823, 0.0313550165242458, 0.1254200660969832, 0.8152304296303908, 0.39641961802518255, 0.05663137400359751, 0.14157843500899378, 0.08494706100539627, 0.3114725570197863, 0.028315687001798755, 0.10062593574532105, 0.8050074859625684, 0.12535731253772772, 0.6267865626886386, 0.12535731253772772, 0.40352454213804845, 0.23635008896657123, 0.17870372580399288, 0.16140981685521938, 0.01152927263251567, 0.01152927263251567, 0.29237201420474973, 0.626511459010178, 0.20385721171201326, 0.6115716351360397, 0.1012035307370937, 0.8096282458967496, 0.8598822980329719, 0.13511770636377868, 0.3377942659094467, 0.40535311909133603, 0.039102265701614514, 0.8993521111371338, 0.08894969015563524, 0.8005472114007172, 0.08894969015563524, 0.47681955827804956, 0.4843088707117362, 0.02746081225685102, 0.002496437477895547, 0.002496437477895547, 0.007489312433686642, 0.7474215434283265, 0.05154631333988458, 0.034364208893256386, 0.06013736556319868, 0.034364208893256386, 0.06872841778651277, 0.040621972774120295, 0.9343053738047667, 0.14448366792235612, 0.14448366792235612, 0.5779346716894245, 0.07879994053180202, 0.07879994053180202, 0.7879994053180202, 0.10909282840404651, 0.7636497988283255, 0.5441472231419938, 0.045345601928499485, 0.09069120385699897, 0.045345601928499485, 0.2267280096424974, 0.04112236388444007, 0.9046920054576816, 0.06209674424907235, 0.8693544194870129, 0.12554105071111177, 0.7532463042666706, 0.08918565548835576, 0.06688924161626682, 0.04459282774417788, 0.735781657778935, 0.04459282774417788, 0.8857667112297444, 0.05536041945185902, 0.26826004892337435, 0.024387277174852215, 0.1463236630491133, 0.04877455434970443, 0.5121328206718965, 0.13999110028675102, 0.034997775071687755, 0.06999555014337551, 0.4899688510036286, 0.13999110028675102, 0.13999110028675102, 0.1760662588359107, 0.5868875294530357, 0.11737750589060714, 0.05868875294530357, 0.0537388493906158, 0.7523438914686211, 0.16121654817184738, 0.8441381095237166, 0.08182762337709805, 0.08182762337709805, 0.7364486103938824, 0.9156302410757191, 0.022890756026892976, 0.022890756026892976, 0.763272753413234, 0.030530910136529355, 0.12212364054611742, 0.030530910136529355, 0.7849547901327389, 0.1916826380696955, 0.09584131903484774, 0.5431008078641372, 0.09584131903484774, 0.06389421268989849, 0.21949802722808887, 0.387779848102957, 0.34388024265733924, 0.014633201815205924, 0.007316600907602962, 0.029266403630411848, 0.3477216610911462, 0.5022646215761001, 0.11590722036371541, 0.6719718869491582, 0.09268577751022872, 0.09268577751022872, 0.1158572218877859, 0.02317144437755718, 0.16787460530195672, 0.5036238159058701, 0.16787460530195672, 0.157556434397753, 0.3781354425546072, 0.4411580163137084, 0.1561768822905265, 0.624707529162106, 0.18351607999333855, 0.12234405332889237, 0.12234405332889237, 0.42820418665112325, 0.12234405332889237, 0.16837824734281298, 0.23572954627993817, 0.06735129893712519, 0.47145909255987634, 0.03367564946856259, 0.03367564946856259, 0.40043063667110906, 0.0942189733343786, 0.02355474333359465, 0.43968854222710013, 0.03140632444479287, 0.015703162222396434, 0.1709868594948753, 0.7409430578111262, 0.2352500181334524, 0.7271364196852165, 0.22498512354220504, 0.4499702470844101, 0.20333859229608423, 0.10166929614804211, 0.5083464807402106, 0.2266685303510464, 0.4533370607020928, 0.12555839770773194, 0.7533503862463917, 0.04459652841044385, 0.0891930568208877, 0.7581409829775454, 0.0891930568208877, 0.22687569933304283, 0.45375139866608566, 0.06110619615763667, 0.18331858847291, 0.6110619615763667, 0.06110619615763667, 0.06110619615763667, 0.06728454824409662, 0.47099183770867636, 0.06728454824409662, 0.06728454824409662, 0.26913819297638647, 0.022428182748032207, 0.22548998253416017, 0.22548998253416017, 0.45097996506832033, 0.14506858783624108, 0.7253429391812054, 0.8440932205379676, 0.041542088735063404, 0.9139259521713948, 0.13605778545309205, 0.2721155709061841, 0.5442311418123682, 0.7640326992411507, 0.07276501897534769, 0.018191254743836923, 0.12733878320685846, 0.23123863637557476, 0.03853977272926246, 0.34685795456336216, 0.07707954545852493, 0.23123863637557476, 0.03853977272926246, 0.06669060145560937, 0.8002872174673125, 0.13338120291121874, 0.12552831689513816, 0.753169901370829, 0.1588932391376834, 0.0794466195688417, 0.0794466195688417, 0.0794466195688417, 0.47667971741305026, 0.0794466195688417, 0.06622703268484688, 0.8609514249030095, 0.03311351634242344, 0.12553090459134011, 0.7531854275480406, 0.6985364943744685, 0.06548779634760642, 0.15280485814441497, 0.06548779634760642, 0.0733324864709001, 0.8066573511799011, 0.6070839634591395, 0.2556143004038482, 0.031951787550481024, 0.015975893775240512, 0.031951787550481024, 0.04792768132572154, 0.10948941652331003, 0.7664259156631702, 0.09954534284757001, 0.7963627427805601, 0.09038334902936232, 0.3615333961174493, 0.09038334902936232, 0.3615333961174493, 0.06787074027533509, 0.882319623579356, 0.09410681311359526, 0.8469613180223572, 0.32545172106747616, 0.32545172106747616, 0.1889719670714378, 0.11548286876587864, 0.020996885230159752, 0.010498442615079876, 0.10056025105229259, 0.8044820084183407, 0.7125812058661182, 0.2521441189987803, 0.010962787782555664, 0.02192557556511133, 0.010962787782555664, 0.07736387686306198, 0.15472775372612396, 0.6962748917675579, 0.1435660369654693, 0.5742641478618772, 0.1435660369654693, 0.16429926635259606, 0.16429926635259606, 0.4928977990577882, 0.06788184731094178, 0.8145821677313014, 0.06788184731094178, 0.28739727727573205, 0.19159818485048805, 0.14369863863786603, 0.3352968234883541, 0.03193303080841468, 0.01596651540420734, 0.3319390535217785, 0.6269959899855816, 0.06496176749872862, 0.8445029774834721, 0.12469192996090726, 0.166255906614543, 0.24938385992181453, 0.36368479571931284, 0.031172982490226816, 0.07273695914386256, 0.3444916074626771, 0.6028603130596849, 0.028707633955223088, 0.1450676073476088, 0.725338036738044, 0.2628228355499778, 0.5256456710999556, 0.7067704209763422, 0.25914915435799213, 0.02355901403254474, 0.07646696846190888, 0.8411366530809976, 0.8068033662342295, 0.07334576056674814, 0.038035943967279884, 0.038035943967279884, 0.34232349570551895, 0.07607188793455977, 0.4944672715746385, 0.10694145936125567, 0.7485902155287897, 0.0956130467300735, 0.764904373840588, 0.0956130467300735, 0.8880539262230183, 0.03861104027056601, 0.16123976383501246, 0.16123976383501246, 0.4837192915050374, 0.20024123124816473, 0.5149060232095665, 0.11442356071323699, 0.10012061562408237, 0.057211780356618495, 0.014302945089154624, 0.8706614558042316, 0.8458666187998765, 0.11552905407719356, 0.8087033785403549, 0.7232000642565879, 0.09040000803207349, 0.10546667603741908, 0.015066668005345582, 0.04520000401603674, 0.5204860175051848, 0.268637944518805, 0.08394935766212658, 0.06715948612970125, 0.03357974306485063, 0.016789871532425314, 0.1366174905919426, 0.0683087452959713, 0.7513961982556843, 0.03415437264798565, 0.13088764128726574, 0.7853258477235945, 0.06725539503005447, 0.7398093453305993, 0.06725539503005447, 0.06725539503005447, 0.04599214736859825, 0.10348233157934608, 0.7128782842132729, 0.011498036842149563, 0.11498036842149564, 0.018870154392957802, 0.018870154392957802, 0.9246375652549322, 0.018870154392957802, 0.018870154392957802, 0.17873517532933936, 0.17277733615169472, 0.5481212043433074, 0.03574703506586787, 0.029789195888223225, 0.04170487424351252, 0.09963847557062419, 0.6974693289943693, 0.09963847557062419, 0.37141033728736245, 0.23213146080460154, 0.18570516864368122, 0.17409859560345115, 0.023213146080460153, 0.011606573040230076, 0.29524254156985785, 0.2567326448433546, 0.17971285139034823, 0.14120295466384505, 0.06418316121083865, 0.06418316121083865, 0.22508002257706874, 0.4501600451541375, 0.2268069929632441, 0.4536139859264882, 0.10330497287824518, 0.10330497287824518, 0.7231348101477162, 0.17292812918449152, 0.6628911618738842, 0.05764270972816384, 0.02882135486408192, 0.02882135486408192, 0.02882135486408192, 0.14773216765204233, 0.29546433530408467, 0.443196502956127, 0.19386601348966487, 0.7269975505862433, 0.2964049590203509, 0.1905460450845113, 0.3599203073818547, 0.14820247951017546, 0.021171782787167925, 0.07308023710386238, 0.6211820153828304, 0.18270059275965597, 0.07308023710386238, 0.22682927457316535, 0.4536585491463307, 0.15354224798627444, 0.690940115938235, 0.08967476948373147, 0.17934953896746295, 0.08967476948373147, 0.08967476948373147, 0.4932112321605231, 0.044837384741865736, 0.08862290973357213, 0.7976061876021493, 0.08335864106835628, 0.8335864106835627, 0.0284989171071942, 0.0569978342143884, 0.0284989171071942, 0.5129805079294957, 0.3704859223935246, 0.0284989171071942, 0.02834839488585882, 0.935497031233341, 0.2288731003954885, 0.4676972051559982, 0.10946104801523363, 0.1293630567452761, 0.019902008730042477, 0.03980401746008495, 0.47954098581386045, 0.09281438435106976, 0.21656689681916277, 0.1856287687021395, 0.030938128117023254, 0.16117613986600912, 0.16117613986600912, 0.4835284195980274, 0.02687535436273436, 0.9406374026957026, 0.02687535436273436, 0.4382704378218809, 0.06920059544556015, 0.41520357267336083, 0.04613373029704009, 0.023066865148520046, 0.30491069993173125, 0.07622767498293281, 0.22868302494879844, 0.30491069993173125, 0.27567845927354495, 0.019691318519538925, 0.059073955558616775, 0.43320900742985635, 0.03938263703907785, 0.17722186667585033, 0.30467105031521025, 0.04352443575931575, 0.13057330727794725, 0.21762217879657875, 0.0870488715186315, 0.21762217879657875, 0.0620983982817894, 0.8693775759450517, 0.8767383476013614, 0.18325287651034566, 0.5497586295310369, 0.9219071183153108, 0.0493859902477973, 0.7407898537169595, 0.1481579707433919, 0.896782462904185, 0.9311323375961187, 0.025165738853949154, 0.025165738853949154, 0.00838857961798305, 0.00838857961798305, 0.00838857961798305, 0.7926491430980825, 0.04954057144363016, 0.09908114288726032, 0.03271441287543542, 0.1962864772526125, 0.5888594317578375, 0.13085765150174167, 0.38655522150976634, 0.20535746142706338, 0.1691179094105228, 0.19327761075488317, 0.024159701344360397, 0.024159701344360397, 0.22597311148125324, 0.6257716933327013, 0.0869127351850974, 0.03476509407403896, 0.11874836550140036, 0.7124901930084021, 0.07896501576614959, 0.07896501576614959, 0.7896501576614958, 0.05637950552790586, 0.5074155497511528, 0.11275901105581172, 0.2818975276395293, 0.12553573817584934, 0.7532144290550961, 0.1255222528368659, 0.7531335170211954, 0.22677638513323245, 0.4535527702664649, 0.23054245505149162, 0.15369497003432775, 0.5763561376287291, 0.5570224084292787, 0.38682111696477695, 0.015472844678591077, 0.030945689357182154, 0.10845091636269508, 0.10845091636269508, 0.5151418527228017, 0.24401456181606393, 0.02711272909067377, 0.08256513121456958, 0.8256513121456958, 0.8397419562514615, 0.0441969450658664, 0.0883938901317328, 0.10801587317010489, 0.7561111121907342, 0.20360578067438845, 0.6108173420231654, 0.04577959531518912, 0.9155919063037824, 0.1261563886453093, 0.693860137549201, 0.1261563886453093, 0.10802485789393171, 0.756174005257522, 0.099032628885896, 0.099032628885896, 0.693228402201272, 0.1255669922394188, 0.753401953436513, 0.39072104647223527, 0.5581729235317646, 0.44867141574846126, 0.2817239122141501, 0.16694750353431118, 0.08347375176715559, 0.010434218970894448, 0.010434218970894448, 0.0330205368015263, 0.8915544936412101, 0.0330205368015263, 0.03912187919481592, 0.8998032214807662, 0.033563649315677346, 0.9397821808389656, 0.7370185672156998, 0.09613285659335215, 0.09613285659335215, 0.032044285531117385, 0.39649617667772136, 0.5699632539742245, 0.024781011042357585, 0.9009460276372513, 0.0925807217941015, 0.7591619187116323, 0.0370322887176406, 0.0925807217941015, 0.0185161443588203, 0.08189848899466028, 0.16379697798932055, 0.08189848899466028, 0.6551879119572822, 0.1832600269188495, 0.5497800807565485, 0.18325323794726564, 0.5497597138417969, 0.05244877251846236, 0.8916291328138601, 0.06960834075621003, 0.2088250222686301, 0.6960834075621003, 0.04706029601924595, 0.0941205920384919, 0.847085328346427, 0.04917472605329726, 0.8851450689593506, 0.07799457549094324, 0.8579403304003757, 0.1832431888821643, 0.5497295666464929, 0.08336740071032982, 0.8336740071032982, 0.27665910600913457, 0.39057756142466055, 0.2603850409497737, 0.032548130118721715, 0.032548130118721715, 0.42473412281464346, 0.1698936491258574, 0.056631216375285794, 0.0849468245629287, 0.22652486550114317, 0.028315608187642897, 0.0675689888730002, 0.0675689888730002, 0.6081208998570018, 0.2027069666190006, 0.0675689888730002, 0.15357489290489654, 0.6910870180720344, 0.4079851956435709, 0.14714220170751738, 0.18727189308229483, 0.1672070473949061, 0.08025938274955492, 0.006688281895796244, 0.19969114973346977, 0.3630748176972177, 0.2904598541577742, 0.06353809309701311, 0.07261496353944355, 0.009076870442430444, 0.21364802300269314, 0.08545920920107726, 0.5982144644075408, 0.04272960460053863, 0.22115997885162028, 0.14743998590108018, 0.07371999295054009, 0.22115997885162028, 0.36859996475270046, 0.7134057535333985, 0.05784370974595123, 0.019281236581983746, 0.13496865607388622, 0.03856247316396749, 0.019281236581983746, 0.16073881982443858, 0.14466493784199472, 0.12859105585955086, 0.1928865837893263, 0.3536254036137649, 0.032147763964887714, 0.3518483960545596, 0.18518336634450502, 0.18518336634450502, 0.12962835644115353, 0.07407334653780201, 0.09259168317225251, 0.2677973115665986, 0.46864529524154747, 0.1115822131527494, 0.1115822131527494, 0.022316442630549882, 0.8393162284270779, 0.03108578623803992, 0.09325735871411976, 0.03108578623803992, 0.03108578623803992, 0.1080281733229493, 0.756197213260645, 0.9051728293381132, 0.03481433958992743, 0.13088681491548576, 0.7853208894929146, 0.19614780053191436, 0.7192086019503526, 0.12551544678963808, 0.7530926807378285, 0.22680854466655598, 0.45361708933311196, 0.1018474270893545, 0.7129319896254815, 0.1018474270893545, 0.3338959828116385, 0.060708360511207, 0.0303541802556035, 0.2124792617892245, 0.0910625407668105, 0.2124792617892245, 0.044583636186455645, 0.044583636186455645, 0.3120854533051895, 0.08916727237291129, 0.4904199980510121, 0.641277281059178, 0.2137590936863927, 0.06412772810591781, 0.02137590936863927, 0.04275181873727854, 0.9122647651270531, 0.05068137584039184, 0.2582611257460356, 0.15495667544762137, 0.5165222514920712, 0.09575733127787907, 0.2872719938336372, 0.4787866563893954, 0.08197060695803329, 0.16394121391606659, 0.7377354626222996, 0.27609582140322014, 0.3248186134155531, 0.2598548907324425, 0.1136865146954436, 0.016240930670777656, 0.10802184782405569, 0.7561529347683899, 0.23646187023199697, 0.7093856106959909, 0.10599019651093076, 0.600611113561941, 0.035330065503643586, 0.035330065503643586, 0.17665032751821794, 0.035330065503643586, 0.29532798459151144, 0.16612199133272518, 0.07383199614787786, 0.03691599807393893, 0.4245339778502977, 0.15337235934390003, 0.5258480891790858, 0.1752826963930286, 0.0876413481965143, 0.04382067409825715, 0.07482982273808336, 0.823128050118917, 0.19771373320258437, 0.5931411996077531, 0.14874776695183925, 0.10297922327435025, 0.0228842718387445, 0.034326407758116746, 0.6750860192429627, 0.0228842718387445, 0.11734497986589729, 0.11734497986589729, 0.7040698791953838, 0.11734497986589729, 0.26922092481727933, 0.033652615602159916, 0.13461046240863966, 0.5384418496345587, 0.09411284850710958, 0.8470156365639863, 0.13755721890471306, 0.6877860945235652, 0.6095116155105134, 0.15709062255425604, 0.09425437353255361, 0.06911987392387266, 0.025134499608680964, 0.043985374315191686, 0.9312486997345906, 0.03355851170214741, 0.008389627925536852, 0.008389627925536852, 0.008389627925536852, 0.008389627925536852, 0.30488923317171035, 0.21249855645301025, 0.2679329624842303, 0.12010787973431014, 0.05774417294918757, 0.036956270687480045, 0.1308952365208501, 0.7853714191251007, 0.09411768961770749, 0.8470592065593673, 0.16032964132774605, 0.16032964132774605, 0.48098892398323817, 0.4323836423391914, 0.2660822414395024, 0.0997808405398134, 0.166301400899689, 0.9126340730392986, 0.16790738533347527, 0.5037221560004258, 0.16790738533347527, 0.879196869531868, 0.10853587213621269, 0.21707174427242537, 0.21707174427242537, 0.43414348854485074, 0.2983643353548051, 0.410250961112857, 0.07459108383870128, 0.1864777095967532, 0.2269452883000525, 0.453890576600105, 0.04144705947884474, 0.16578823791537897, 0.7460470706192054, 0.04144705947884474, 0.20189955407354054, 0.6056986622206216, 0.04238041814733214, 0.08476083629466429, 0.6780866903573143, 0.12714125444199642, 0.08476083629466429, 0.20375443497754173, 0.6112633049326252, 0.2269345514192806, 0.4538691028385612, 0.08198719273850275, 0.7378847346465246, 0.451786507871338, 0.0225893253935669, 0.3614292062970704, 0.1581252777549683, 0.14438953636749513, 0.28877907273499026, 0.4331686091024854, 0.2903836937362936, 0.07641676150955094, 0.5196339782649465, 0.01528335230191019, 0.04585005690573057, 0.04585005690573057, 0.33146620168669116, 0.19887972101201468, 0.39775944202402935, 0.6163525678382695, 0.23969266527043817, 0.06848361864869662, 0.03424180932434831, 0.017120904662174155, 0.03424180932434831, 0.10802327691953476, 0.7561629384367433, 0.16330547216073033, 0.16330547216073033, 0.16330547216073033, 0.48991641648219103, 0.2019356440320758, 0.6058069320962275, 0.9063216681096078, 0.04899036043835718, 0.02449518021917859, 0.1132834968640111, 0.3398504905920333, 0.4531339874560444, 0.8134884982946552, 0.9679250202515128, 0.0064100994718643235, 0.012820198943728647, 0.0064100994718643235, 0.05312138757714226, 0.5312138757714225, 0.05312138757714226, 0.10624277515428451, 0.21248555030856903, 0.20305067722295966, 0.609152031668879, 0.22674113159805376, 0.4534822631961075, 0.15592215862530864, 0.15592215862530864, 0.6236886345012346, 0.042359904271798615, 0.08471980854359723, 0.8471980854359723, 0.02776129262032013, 0.02776129262032013, 0.3608968040641617, 0.6107484376470429, 0.021984833365930656, 0.021984833365930656, 0.9453478347350182, 0.1255577024296905, 0.753346214578143, 0.5558923608323549, 0.2501515623745597, 0.13897309020808873, 0.027794618041617744, 0.013897309020808872, 0.013897309020808872, 0.4051799769260553, 0.11254999359057091, 0.022509998718114183, 0.045019997436228365, 0.3601599794898269, 0.06752999615434255, 0.9021791770867728], "Term": ["abolishment", "abolishment", "abuse", "abuse", "abuse", "abuse", "abuse", "academic", "academic", "academic", "act", "act", "act", "act", "act", "act", "activity", "activity", "activity", "actuarial", "actuarial", "actuary", "actuary", "adoption", "adoption", "adoption", "adopts", "adopts", "affiliated", "affiliated", "affixed", "affixed", "afterschool", "afterschool", "age", "age", "age", "agencies", "agencies", "agencies", "agencies", "agencies", "agency", "agency", "agency", "agency", "agency", "agent", "agent", "agent", "agent", "agent", "aggravated", "aggravated", "aggravated", "aggravated", "allowances", "allowances", "allowances", "allowances", "ambulance", "appointment", "appointment", "appointment", "appropriations", "appropriations", "appropriations", "architect", "architect", "assembly", "assembly", "assembly", "assembly", "assembly", "associate", "associate", "associate", "atlanta", "atlanta", "attempt", "attorney", "attorney", "attorney", "attorney", "august", "august", "authorized", "authorized", "authorized", "authorized", "authorized", "automatic", "automatic", "available", "available", "available", "available", "available", "available", "award", "award", "behavioral", "behavioral", "behavioral", "behavioral", "beneficiary", "beneficiary", "benefits", "benefits", "benefits", "board", "board", "board", "board", "board", "board", "budget", "budget", "capitol", "capitol", "capitol", "care", "care", "care", "care", "care", "care", "caregiver", "caregiver", "cares", "cares", "casa", "casa", "casework", "chairperson", "chairperson", "chairperson", "change", "change", "charge", "charge", "charge", "child", "child", "child", "child", "child", "child", "children", "children", "children", "children", "children", "children", "circuit", "circuit", "citizens", "citizens", "citizens", "claims", "claims", "claims", "clergy", "clergy", "clerk", "clerk", "clerk", "clerk", "clerk", "cmo", "cmo", "cmos", "cmos", "color", "color", "commission", "commission", "commission", "commission", "commission", "commit", "commit", "committee", "committee", "committee", "committee", "committee", "community", "community", "community", "community", "community", "community", "compensation", "compensation", "compensation", "compensation", "consider", "consider", "consider", "consists", "construction", "construction", "construction", "contact", "contact", "contact", "convicted", "convicted", "convicted", "convicted", "convictions", "county", "county", "county", "county", "county", "court", "court", "court", "court", "court", "court", "courts", "courts", "courts", "criminal", "criminal", "criminal", "criminal", "criminal", "critical", "critical", "critical", "custodian", "custodian", "custodian", "darden", "darden", "day", "day", "day", "day", "day", "defined", "defined", "defined", "defined", "defined", "defined", "department", "department", "department", "department", "department", "department", "dependent", "dependent", "dfcs", "dfcs", "displays", "displays", "documents", "documents", "documents", "douglas", "douglas", "doula", "doula", "drug", "drug", "drug", "drug", "duplication", "duplication", "duty", "duty", "duty", "duty", "duty", "education", "education", "education", "education", "education", "education", "effort", "effort", "effort", "eighteenth", "eighteenth", "elements", "emancipation", "emancipation", "emergency", "emergency", "emergency", "employee", "employee", "employee", "employee", "enforcement", "enforcement", "enforcement", "enforcement", "enforcement", "enforcement", "enter", "enter", "enter", "ethnicity", "ethnicity", "every", "every", "every", "every", "every", "every", "evidence", "evidence", "evidence", "expression", "expression", "facility", "facility", "facility", "facility", "fact", "fact", "family", "family", "family", "family", "family", "family", "father", "father", "female", "female", "file", "file", "file", "file", "filled", "filled", "firearms", "firearms", "follows", "follows", "follows", "follows", "follows", "follows", "forfeiture", "forfeiture", "foster", "foster", "foster", "foster", "foster", "full", "full", "full", "funding", "funding", "funding", "gang", "gang", "gang", "gender", "gender", "gender", "general", "general", "general", "general", "general", "general", "guardian", "guardian", "gun", "gun", "health", "health", "health", "health", "health", "health", "hearing", "hearing", "hearing", "hearsay", "hearsay", "help", "help", "home", "home", "home", "homeland", "homeland", "homes", "homes", "house", "house", "house", "house", "house", "immediate", "immediate", "impact", "impact", "impact", "improper", "improper", "improved", "improved", "improved", "including", "including", "including", "including", "including", "including", "incubators", "indecent", "indemnification", "indemnification", "individual", "individual", "individual", "individual", "individual", "information", "information", "information", "information", "information", "information", "institution", "institution", "institution", "institution", "intake", "intake", "interests", "interests", "interests", "interests", "judge", "judge", "judge", "judge", "judge", "judges", "judges", "judges", "judges", "judges", "juvenile", "juvenile", "juvenile", "juvenile", "juvenile", "juvenile", "labor", "labor", "labor", "laws", "laws", "laws", "laws", "laws", "laws", "lc", "lc", "lc", "lc", "lc", "lc", "leaders", "leaders", "leadership", "leadership", "learning", "learning", "learning", "legal", "legal", "legal", "legal", "legal", "legal", "legislation", "legislation", "legislation", "license", "license", "licensed", "licensed", "licensed", "licensed", "licensed", "limited", "limited", "limited", "limited", "linnie", "linnie", "living", "living", "local", "local", "local", "local", "local", "local", "machine", "machine", "managed", "managed", "management", "management", "management", "management", "management", "management", "marriage", "marriage", "may", "may", "may", "may", "may", "may", "means", "means", "means", "means", "means", "measures", "measures", "measures", "medicaid", "medicaid", "medicaid", "medical", "medical", "medical", "medical", "medical", "meetings", "meetings", "meetings", "meetings", "members", "members", "members", "members", "members", "members", "mental", "mental", "mental", "mental", "mental", "mental", "million", "million", "molestation", "network", "network", "newborn", "notice", "notice", "notice", "obscene", "offense", "offense", "offense", "offense", "offense", "offense", "offenses", "offenses", "offenses", "officer", "officer", "officer", "officer", "official", "official", "official", "official", "official", "official", "order", "order", "order", "order", "ordered", "ordered", "organization", "organization", "organization", "organizations", "organizations", "organizations", "organizations", "orientation", "orientation", "origin", "origin", "outstanding", "outstanding", "paid", "paid", "paid", "parent", "parent", "parent", "parent", "part", "part", "part", "part", "part", "party", "party", "payment", "payment", "payment", "pbm", "pbm", "peggy", "peggy", "penal", "penal", "percent", "percent", "percent", "percentile", "percentile", "performing", "performing", "performing", "perinatal", "perinatal", "permanency", "permanency", "person", "person", "person", "person", "person", "person", "petition", "petition", "petition", "petitioner", "petitioner", "pharmacy", "pharmacy", "placed", "placed", "placed", "placed", "placement", "placement", "placement", "placing", "plan", "plan", "plan", "plan", "plan", "plans", "plans", "plans", "plans", "play", "play", "pm", "pm", "postpartum", "postpartum", "pregnancy", "pregnancy", "pregnancy", "pregnant", "pregnant", "pregnant", "premarital", "premarital", "prenatal", "prenatal", "preparing", "preparing", "prescription", "prescription", "program", "program", "program", "program", "program", "programs", "programs", "programs", "programs", "programs", "programs", "property", "property", "property", "property", "property", "protective", "protective", "provide", "provide", "provide", "provide", "provide", "provide", "provided", "provided", "provided", "provided", "provided", "provided", "provider", "provider", "provider", "provider", "providers", "providers", "providers", "providers", "providers", "providing", "providing", "providing", "providing", "providing", "providing", "public", "public", "public", "public", "public", "public", "purposes", "purposes", "purposes", "purposes", "purposes", "purposes", "pursuant", "pursuant", "pursuant", "pursuant", "pursuant", "qualified", "qualified", "qualified", "qualified", "qualified", "ranked", "ranked", "rape", "rape", "recommendation", "recommendation", "registry", "registry", "religion", "religion", "remarkable", "remarkable", "removal", "removal", "removal", "report", "report", "report", "report", "report", "report", "representatives", "representatives", "representatives", "representatives", "representatives", "required", "required", "required", "required", "required", "residential", "residential", "resolution", "resolution", "resolution", "resolved", "resolved", "resolved", "restraints", "restraints", "restraints", "revising", "revising", "revising", "revising", "revising", "rifle", "rifle", "rights", "rights", "safe", "safe", "safe", "safe", "safe", "safe", "safety", "safety", "safety", "safety", "safety", "said", "said", "said", "said", "said", "sawed", "sawed", "scholastic", "scholastic", "school", "school", "school", "school", "school", "school", "schools", "schools", "schools", "schools", "security", "security", "security", "security", "semi", "semi", "seminars", "seminars", "services", "services", "services", "services", "services", "services", "sexual", "sexual", "sexual", "sexual", "sexual", "sexual", "shall", "shall", "shall", "shall", "shall", "shall", "sheet", "sheet", "shotgun", "shotgun", "since", "since", "since", "social", "social", "social", "social", "sodomy", "statewide", "statewide", "statewide", "station", "students", "students", "students", "students", "study", "study", "study", "study", "suggestions", "suggestions", "superior", "superior", "superior", "superior", "suspicious", "suspicious", "system", "system", "system", "system", "system", "talents", "talents", "task", "task", "temporary", "temporary", "term", "term", "term", "term", "therefore", "therefore", "therefore", "time", "time", "time", "time", "time", "time", "trafficking", "trafficking", "trafficking", "treatment", "treatment", "treatment", "treatment", "treatment", "treatment", "units", "units", "university", "university", "university", "university", "unsafe", "unsafe", "victim", "victim", "victim", "victims", "victims", "victims", "viii", "violation", "violation", "violation", "violation", "violence", "violence", "violence", "violence", "violence", "violent", "violent", "vision", "vision", "walker", "walker", "walker", "weapon", "weapon", "weapon", "whereas", "whereas", "whereas", "whereas", "woman", "woman", "woman", "women", "women", "years", "years", "years", "years", "years", "years", "youth", "youth", "youth", "youth", "youth", "youth", "youths"]}, "R": 30, "lambda.step": 0.01, "plot.opts": {"xlab": "PC1", "ylab": "PC2"}, "topic.order": [5, 1, 3, 4, 6, 2]};

function LDAvis_load_lib(url, callback){
  var s = document.createElement('script');
  s.src = url;
  s.async = true;
  s.onreadystatechange = s.onload = callback;
  s.onerror = function(){console.warn("failed to load library " + url);};
  document.getElementsByTagName("head")[0].appendChild(s);
}

if(typeof(LDAvis) !== "undefined"){
   // already loaded: just create the visualization
   !function(LDAvis){
       new LDAvis("#" + "ldavis_el74471405249460713282762657121", ldavis_el74471405249460713282762657121_data);
   }(LDAvis);
}else if(typeof define === "function" && define.amd){
   // require.js is available: use it to load d3/LDAvis
   require.config({paths: {d3: "https://d3js.org/d3.v5"}});
   require(["d3"], function(d3){
      window.d3 = d3;
      LDAvis_load_lib("https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.3.1/pyLDAvis/js/ldavis.v3.0.0.js", function(){
        new LDAvis("#" + "ldavis_el74471405249460713282762657121", ldavis_el74471405249460713282762657121_data);
      });
    });
}else{
    // require.js not available: dynamically load d3 & LDAvis
    LDAvis_load_lib("https://d3js.org/d3.v5.js", function(){
         LDAvis_load_lib("https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.3.1/pyLDAvis/js/ldavis.v3.0.0.js", function(){
                 new LDAvis("#" + "ldavis_el74471405249460713282762657121", ldavis_el74471405249460713282762657121_data);
            })
         });
}
</script>


