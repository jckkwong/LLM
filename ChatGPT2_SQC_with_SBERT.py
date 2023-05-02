

import openai
import pandas as pd
import numpy as np
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, PromptHelper, ServiceContext, LLMPredictor

from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import os
import re
import time

#Excel Loader ---------------------------------------------------------------------------------
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings import SelfHostedEmbeddings
from langchain import PromptTemplate

#Make Chroma (Langchang default Vector DB) persistant -------------------------------------------

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader

#Although FAISS can work locally, limited API, so probably use Chroma
#FAISS Vector DB (from Facebook)-------------------------Pip install faiss-cpu or gpu-------------
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

#Add SBERT-----------------------------------------------------------------------------------------
# from langchain.embeddings import SentenceTransformerEmbeddings
# for langchain, you need to update the embedding folder with newst files from github
# (https://github.com/hwchase17/langchain/tree/0cf934ce7d8150dddf4a2514d6e7729a16d55b0f/langchain/embeddings)


#Add Langchain Document Spliter---------------------------------------------------------------------
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import TokenTextSplitter

#Add Unstructured Loader-----------------------------------------------------------------------------
# https://langchain-fanyi.readthedocs.io/en/latest/modules/indexes/document_loaders/examples/directory_loader.html

from langchain.document_loaders import DirectoryLoader

#Open-source framework Sentence Transformer
from sentence_transformers import SentenceTransformer

SEP = '=================='

# setting openai key

OPENAI_API = open('./OPENAI_API.txt', "r").read()
openai.api_key = OPENAI_API
os.environ["OPENAI_API_KEY"] = OPENAI_API

COMPLETIONS_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"

GUIDE = "Suppose you are a person from a company filling out a survey using the given context, "\
        "Answer the question as truthfully as possible and keep the answer short."\
        "Do not mention context or original answer."\
        "Replace answers containing 'Signifi' with 'ACME Co.'."\
        "Input: Signifi  Output: ACME." \
        "Input: Signifi Corp Output: ACEM Co." \
        "Input: Signifi associates Output: ACEM associates" \
        "And if you are unsure of the answer, say only 'N/A' without giving any reason.\n"


GUIDE_For_CONTROL = \
        "Suppose you are giving advice on how to accomplish what is being asked using the given context, "\
        "Answer the question as truthfully as possible with recommended steps and keep the answer short."\
        "Do not mention context or original answer."\
        'Do not answer the question, give only action steps \n'\
        'describe in no more than 3 steps needed to acheive this \n'\
        "And if you are unsure of the answer, say only 'N/A' without giving any reason.\n"

GUIDE_For_CONTROL_NIST = \
        "Suppose you are giving advice on how to accomplish what is being asked using the given context, "\
        "Answer the question as truthfully as possible with recommended steps and keep the answer short."\
        'describe in no more than 5 steps needed to acheive this \n'\
        "And if you are unsure of the answer, say only 'I can't seem to find this info' without giving any reason.\n"


###############################################################################################
# GLOBAL FILES and DATA
###############################################################################################

SURVEY_FILE = './Data/Survey/SurveyTestSmall.csv'
SURVEY_WITH_ANSWERS_FILE = './Data/SurveyWithAnswers/SurveyWithAnswers.csv'
POLICY_INPUT_DIR = './Data/Other_Documents/Policies/'
CONTROL_INPUT_DIR = './Data/Other_Documents/Controls/'
FRAMEWORK_INPUT_DIR = './Data/Other_Documents/Frameworks/'

CONTROL_FILE_NAME = 'NIST800-53v4_OPPOS_Edited.csv'

SURVEY_EMBEDDINGS_DIR = './Data/KnowledgeBase/Chroma/Survey/'
POLICY_EMBEDDINGS_DIR = './Data/KnowledgeBase/Chroma/Policy/'
CONTROL_EMBEDDINGS_DIR = './Data/KnowledgeBase/Chroma/Control/'
GDPR_EMBEDDINGS_DIR = './Data/KnowledgeBase/Chroma/GDPR/'
HIPAA_EMBEDDINGS_DIR = './Data/KnowledgeBase/Chroma/HIPAA/'
FRAMEWORK_EMBEDDINGS_DIR = './Data/KnowledgeBase/Chroma/framework/'

SURVEY_RESPONSE_DIR = './Data/Output/'
SURVEY_RESPONSE_NAME = 'SurveyResponse'

# ----------------------------------------------------------------------------------------------
SURVEY_WITH_ANSWERS = pd.read_csv(SURVEY_WITH_ANSWERS_FILE)
SURVEY = pd.read_csv(SURVEY_FILE)


# -----------------------------------------------------------------------------------------------
# take only the questions column
QUESTIONS = SURVEY[['question']].copy()

# -----------------------------------------------------------------------------------------------


###############################################################################################

def gpt_response(question: str, context: str='') -> str:
    complete_prompt= context + 'Question:\n' + question
    return openai.ChatCompletion.create(
        temperature=0,
        messages=[
            {"role": "system", "content": GUIDE},
            {"role": "user", "content": complete_prompt}
            ],
        max_tokens=1000,
        model=COMPLETIONS_MODEL
        )["choices"][0]["message"]['content'].strip(" \n")


def gpt_response_ctrl(question: str, context: str='') -> str:
    complete_prompt= context + 'Question:\n' + question

    return openai.ChatCompletion.create(
        temperature=0,
        messages=[
            {"role": "system", "content": GUIDE_For_CONTROL},
            {"role": "user", "content": complete_prompt}
            ],
        max_tokens=1000,
        model=COMPLETIONS_MODEL
        )["choices"][0]["message"]['content'].strip(" \n")

def gpt_response_ctrl_NIST(question: str, context: str='') -> str:
    complete_prompt= context + 'Question:\n' + question

    return openai.ChatCompletion.create(
        temperature=0,
        messages=[
            {"role": "system", "content": GUIDE_For_CONTROL_NIST},
            {"role": "user", "content": complete_prompt}
            ],
        max_tokens=1000,
        model=COMPLETIONS_MODEL
        )["choices"][0]["message"]['content'].strip(" \n")



def answerQuestions(questionContext: pd.DataFrame) -> pd.DataFrame:

    answerList = []
    # Answer question using filled out surveys and policy first
    for i, row in questionContext.iterrows():
        answer = gpt_response (row.question, row.questionAnswer + '\n' + row.policy)
        answer = re.sub(re.escape('signifi '), 'ACME ', answer, flags=re.IGNORECASE)
        # Try again with NIST control document if no answer
        if answer == "N/A":
            answer = gpt_response_ctrl (row.question, row.control)
            answer = re.sub(re.escape('signifi '), 'ACME ', answer, flags=re.IGNORECASE)
            answer = f'Not sure about this one \n -How about this ...\n\n{answer}'

        answerList.append (answer)

    questionContext['answer'] = pd.Series(answerList,index=questionContext.index)

    return questionContext

def answerQuestionsNIST(questionContext: pd.DataFrame) -> pd.DataFrame:

    answerList = []
    # Answer question using filled out surveys and policy first
    for i, row in questionContext.iterrows():
        answer = gpt_response_ctrl_NIST (row.question, row.control)
        answer = re.sub(re.escape('signifi '), 'ACME ', answer, flags=re.IGNORECASE)

        answerList.append (answer)

    questionContext['answer'] = pd.Series(answerList,index=questionContext.index)

    return questionContext


def questionMatchKB (question_df: pd.DataFrame,
                target_col: str,
                surveyAnswerIndex,
                policyIndex,
                controlIndex) -> pd.DataFrame:


    # Extract from returned documents
    def extractToString(doc,prefix) -> str:
        resultString = ''
        for i, row in enumerate(doc):
            resultString += f'\n {prefix}_{i + 1}. {row.page_content}'
        return resultString + '\n'

    # Iterate through questions and get results
    query_df = pd.DataFrame(columns=['question','questionAnswer','policy','control'])
    for i, row in question_df.iterrows():
        surveyQuestion = question_df.question[i]
        surveyAnswerSrchResult = extractToString(surveyAnswerIndex.similarity_search(surveyQuestion,k=5),'s')
        policySrchResult = extractToString(policyIndex.similarity_search(surveyQuestion,k=5),'p')
        controlSrchResult = extractToString(controlIndex.similarity_search(surveyQuestion,k=5),'c')

        query_df.loc[len(query_df)] = [surveyQuestion, surveyAnswerSrchResult, policySrchResult, controlSrchResult]

    return query_df



def retriveSurveyEmbedding(surveyWithAnswers, surveyAnswersIndexDst, computeEmbedding = False):

    # Define Embedding Model
    embedding = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    if computeEmbedding:

        # combine the question and answer into one string with a line break
        surveyWithAnswers['combined'] = surveyWithAnswers['question'] + '\n' + surveyWithAnswers['answer']
        Chroma.from_texts(texts=surveyWithAnswers.combined.tolist(),
                              embedding=embedding,
                              persist_directory=surveyAnswersIndexDst)


    assert os.path.exists(surveyAnswersIndexDst)
    return Chroma(persist_directory=surveyAnswersIndexDst, embedding_function=embedding)



def retrievePolicyEmbedding(docDir, docIndexDst, computeEmbedding = False):

    # Define Embedding Model
    embedding = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    if computeEmbedding:
        # Instead of just PDF, it's also possible to use unstructured-IO for different formats in the directory
        # https://github.com/Unstructured-IO/unstructured#coffee-getting-started

        loader = DirectoryLoader(docDir, glob="**/*.docx") #load dir
        # loader = UnstructuredWordDocumentLoader(f'{docDir}CO001 - Code of Ethics.docx') #One file at a time
        documents = loader.load()

        text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=30)
        pdf_chunks = text_splitter.split_documents(documents)


        Chroma.from_documents(documents=pdf_chunks, embedding=embedding, persist_directory=docIndexDst)


    assert os.path.exists(docIndexDst)
    return Chroma(persist_directory=docIndexDst, embedding_function = embedding)


def retrieveControlEmbedding(controlInputDir, controlFileName, controlIndexDst, computeEmbedding = False):

    # Define Embedding Model
    embedding = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    if computeEmbedding:
        # To make Chroma persistant https://github.com/hwchase17/chroma-langchain/blob/master/persistent-qa.ipynb
        loader = CSVLoader(file_path=f'{controlInputDir}{controlFileName}')  # Excel Source 2
        documents = loader.load()


        Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=controlIndexDst)


    assert os.path.exists(controlIndexDst)
    return Chroma(persist_directory=controlIndexDst, embedding_function=embedding)

# def retrieveGDPREmbedding(GDPR_InputDir, GDPR_FileName, GDPR_IndexDst, GDPR_Embedding = False):
#
#     # Define Embedding Model
#     embedding = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
#     if GDPR_Embedding:
#         # To make Chroma persistant https://github.com/hwchase17/chroma-langchain/blob/master/persistent-qa.ipynb
#         loader = CSVLoader(file_path=f'{GDPR_InputDir}{GDPR_FileName}')  # Excel Source 2
#         documents = loader.load()
#
#
#         Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=GDPR_IndexDst)
#
#
#     assert os.path.exists(GDPR_IndexDst)
#     return Chroma(persist_directory=GDPR_IndexDst, embedding_function=embedding)
#
# def retrieveHIPAAEmbedding(HIPAA_InputDir, HIPAA_FileName, HIPAA_IndexDst, computeEmbedding = False):
#
#     # Define Embedding Model
#     embedding = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
#     if computeEmbedding:
#         # To make Chroma persistant https://github.com/hwchase17/chroma-langchain/blob/master/persistent-qa.ipynb
#         loader = CSVLoader(file_path=f'{HIPAA_InputDir}{HIPAA_FileName}')  # Excel Source 2
#         documents = loader.load()
#
#
#         Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=HIPAA_IndexDst)
#
#
#     assert os.path.exists(HIPAA_IndexDst)
#     return Chroma(persist_directory=HIPAA_IndexDst, embedding_function=embedding)

def surveyFill (suffix):

    # Get Vector Index
    surveyKB = retriveSurveyEmbedding(SURVEY_WITH_ANSWERS, SURVEY_EMBEDDINGS_DIR,computeEmbedding=False)
    policyKB = retrievePolicyEmbedding(POLICY_INPUT_DIR,POLICY_EMBEDDINGS_DIR,computeEmbedding=False)
    controlKB = retrieveControlEmbedding(CONTROL_INPUT_DIR, CONTROL_FILE_NAME, CONTROL_EMBEDDINGS_DIR,
                                         computeEmbedding=False)

    # Fill survey
    questionMatch_df = questionMatchKB (QUESTIONS, 'question', surveyKB, policyKB, controlKB )
    answers_df = answerQuestions (questionMatch_df)
    answers_df.to_csv(f'{SURVEY_RESPONSE_DIR}{SURVEY_RESPONSE_NAME}_{suffix}.csv', index=False)

def askQuestion (question:str) -> str:

    # Get Vector Index
    surveyKB = retriveSurveyEmbedding(SURVEY_WITH_ANSWERS, SURVEY_EMBEDDINGS_DIR,computeEmbedding=False)
    policyKB = retrievePolicyEmbedding(POLICY_INPUT_DIR,POLICY_EMBEDDINGS_DIR,computeEmbedding=False)
    controlKB = retrieveControlEmbedding(CONTROL_INPUT_DIR, CONTROL_FILE_NAME, CONTROL_EMBEDDINGS_DIR,
                                         computeEmbedding=False)

    question = pd.DataFrame({'question':[question]})

    questionMatch_df = questionMatchKB (question, 'question', surveyKB, policyKB, controlKB )

    answers_df = answerQuestions (questionMatch_df)

    return answers_df.answer[0]


def askQuestionNIST (question:str) -> str:

    # Get Vector Index
    surveyKB = retriveSurveyEmbedding(SURVEY_WITH_ANSWERS, SURVEY_EMBEDDINGS_DIR,computeEmbedding=False)
    policyKB = retrievePolicyEmbedding(POLICY_INPUT_DIR,POLICY_EMBEDDINGS_DIR,computeEmbedding=False)
    controlKB = retrieveControlEmbedding(CONTROL_INPUT_DIR, CONTROL_FILE_NAME, CONTROL_EMBEDDINGS_DIR,
                                         computeEmbedding=False)

    question = pd.DataFrame({'question':[question]})

    questionMatch_df = questionMatchKB (question, 'question', surveyKB, policyKB, controlKB )

    answers_df = answerQuestionsNIST (questionMatch_df)

    return answers_df.answer[0]

if __name__ == '__main__':

    # askQuestion('Configure security policy filters')
    # Handle Excel Files
        # https://shabeelkandi.medium.com/chat-with-an-excel-dataset-with-openai-and-langchain-5520ce2ac5d3
    # Vector DBs
        # chroma, faise-cpu, pinecone




    ###############################################################################################
    # COMPUTE KNOWLEDGE BASE
    ###############################################################################################

    # retrieve or compute and retrieve
    # surveyKB = retriveSurveyEmbedding(surveyWithAnswers, surveyEmbeddingsDir,computeEmbedding=False)
    # policyKB = retrievePolicyEmbedding(policyInputDir,policyEmbeddingsDir,computeEmbedding=False)
    # controlKB = retrieveControlEmbedding(controlInputDir, controlFileName, controlEmbeddingsDir, computeEmbedding=True)
    # frameworkKB = retrieveFrameworkEmbedding(computeEmbedding=True)

    ###############################################################################################
    # COMPLETE SURVEYS
    ###############################################################################################

    # surveyFill('_AnswrPol01')
    print (askQuestion('Do you have a security policy?'))





