#!/usr/bin/env python
# coding: utf-8

# In[1]:


from setting import *


# In[2]:


from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# In[3]:


# # similar questions search

# In[4]:


data = pd.read_json(
    os.path.join(
    data_path,
    'question_embedding.json',
    ),
    lines = True,
    orient = 'records',
)


# In[5]:


def embedding_text(sentence):
    try:
        return model.encode([sentence])[0]
    except:
        return None


# In[6]:


def similarity_embedding(
    query_question_embedding,
    question_embedding,
    ):
    try:
        return np.dot(
            query_question_embedding,
            np.array(question_embedding),
        )
    except:
        return None   


# # prompt engineering

# In[18]:


def answer_search(
    query_question,
    ):
    
    try:

        query_question_embedding = embedding_text(query_question)
        data['score'] = data['question_embedding'].apply(
            lambda x: similarity_embedding(query_question_embedding, x))

        data[[
            #'question',
            'answer',
            'reference text',
            'score',
        ]].to_sql('sorted_questions', conn, if_exists = 'replace')


        similar_qa_pairs = pd.read_sql(f"""
        select 
        answer,
        `reference text`,
        score
        from sorted_questions
        order by score desc
        limit 1
        """, conn)        

        return similar_qa_pairs.to_dict(
            orient = 'records',
            )[0]
    
    except:
        
        return None

'''
# # test

# In[19]:


query_question = "How can urban planning and design reduce visual pollution?"

response = answer_search(query_question)

print(response)


# In[21]:


query_question = "an abandoned/empty building has risk of collapse, is it a kind of visual pollution?"

response = answer_search(query_question)

print(response)


# In[22]:


query_question = "Is an empty house a kind of visual pollution?"

response = answer_search(query_question)

print(response)


# In[24]:


query_question = "What are the most common sources of visual pollution?"

response = answer_search(query_question)

print(response)


# In[30]:


query_question = "What is visual pollution?"

response = answer_search(query_question)

print(response)


{'answer': 'visual pollution is a set of visual elements as defined in the national plan for visual pollution visible in the infrastructure or urban fabric that have worn out their original state or violate the regulations and instructions governing this in a way that affects the appearance of the city due to negligence in inspection, bypassing the various parties (e.g., residents, private companies, and others), and usually producing visual distortion in addition to the irregular diligence of the owners of private property.', 'reference text': 'VP_Sample Processed Data for Tonomus page 1', 'score': 1.000000017179252}


# # end

'''