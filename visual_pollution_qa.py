#!/usr/bin/env python
# coding: utf-8

# In[1]:


from setting import *


# In[2]:


from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# In[3]:


from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

pipe = pipeline("text2text-generation", model="google/flan-t5-base")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")


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

# In[7]:


def question_to_prompt(
    query_question,
    ):

    query_question_embedding = embedding_text(query_question)
    data['similarity'] = data['question_embedding'].apply(
        lambda x: similarity_embedding(query_question_embedding, x))

    data[[
        'question',
        'answer',
        'similarity'
    ]].to_sql('sorted_questions', conn, if_exists = 'replace')


    similar_qa_pairs = pd.read_sql(f"""
    select * 
    from sorted_questions
    order by similarity desc
    limit 2
    """, conn)

    examples = []

    for r in similar_qa_pairs.to_dict(
        orient = 'records',
        ):
        question = re.sub(r'\n+', r' ', r['question'])
        answer = re.sub(r'\n+', r' ', r['answer'])
        example = f"Q: {question}\nA: {answer}".strip()
        examples.append(example)

    examples = "\n\n".join(examples)

    prompt = f"""
Learn from the following Q&A pairs and answer the question:

{examples}

question: {query_question}
    """
    
    return prompt


# # LLM model calling

# In[8]:


def visual_pollution_qa(
    query_question,
    ):
    
    try:
        prompt = question_to_prompt(query_question)
        sequences = pipe(
        prompt,
        max_new_tokens = 100,
        do_sample=True,
        top_k=1,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        )

        return sequences[0]['generated_text']
    except:
        return None


# # test

# In[37]:


query_question = "How can urban planning and design reduce visual pollution?"


# In[34]:


query_question = "What are the 5 main categories of visual pollution?"


# In[38]:


print(visual_pollution_qa(query_question))


# In[39]:


print(question_to_prompt(query_question))


# # end
