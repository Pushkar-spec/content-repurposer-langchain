import sys
sys.stdout.reconfigure(encoding='utf-8')

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnablePassthrough, RunnableLambda 
from dotenv import load_dotenv

from prompts import *

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",  # or "Qwen/Qwen2.5-7B-Instruct"
    task="text-generation",
    max_new_tokens=1024,
    temperature=0.5,
)

model = ChatHuggingFace(llm = llm)

classifier_llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text-generation',
    max_new_tokens=100,
    temperature=0.1,
)
classifier_model = ChatHuggingFace(llm=classifier_llm)

parser = StrOutputParser()

classifier_chain = (classify_prompt | classifier_model | category_parser | RunnableLambda(lambda x : x.category))

article_branch = RunnableBranch(
    (lambda x : x['category'] == 'technical', technical_article_prompt | model | parser),
    (lambda x : x['category'] == 'news', news_article_prompt | model | parser),
    casual_article_prompt | model | parser
)

summary_chain = summary_prompt | model | parser

parallel_chain = RunnableParallel({
    'tweet' : tweet_prompt | model | parser,
    'linkedin' : linkedin_prompt | model | parser,
    'instagram' : instagram_prompt | model | parser,
    'summary': RunnableLambda(lambda x: x['summary']),    # carried through for the UI
    'category': RunnableLambda(lambda x: x['category']),
})

def clean(result):
    result['tweet'] = result['tweet'].strip()
    result['linkedin'] = result['linkedin'].strip()
    result['instagram'] = result['instagram'].strip()
    result['tweet_length'] = len(result['tweet'])
    result['tweet_ok'] = result['tweet_length'] <= 280
    return result

pipeline = (
    RunnablePassthrough.assign(category = classifier_chain) | RunnablePassthrough.assign(article = article_branch) | RunnablePassthrough.assign(summary = summary_chain) | parallel_chain |RunnableLambda(clean)
)

if __name__ == '__main__':
    result = pipeline.invoke({'topic': 'black holes'})
    for key, value in result.items():
        print(f'\n--- {key.upper()} ---\n{value}')

    pipeline.get_graph().print_ascii()