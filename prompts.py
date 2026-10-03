from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from schema import category

category_parser = PydanticOutputParser(pydantic_object = category)

classify_prompt = PromptTemplate(
    template = 'Classify the topic below as exactly one of: technical, news, casual.\n''Topic: {topic}\n\n{format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': category_parser.get_format_instructions()}
)

technical_article_prompt = PromptTemplate(
    template = 'Write a precise, detailed and well structured technical article on {topic}.\n''Use correct terminology. Keep it under 400 words',
    input_variables = ['topic']
)

news_article_prompt = PromptTemplate(
    template='Write a factual news-style article on {topic}. ''Cover who, what, when, where and why. Keep it under 400 words.',
    input_variables=['topic']
)

casual_article_prompt = PromptTemplate(
    template='Write a friendly, easy-to-read and engaging article on {topic}, ''as if explaining it to a friend. Keep it under 400 words.',
    input_variables=['topic']
)

summary_prompt = PromptTemplate(
    template='Here is an article:\n\n{article}\n\nWrite a 5 line summary of the article above.',
    input_variables=['article']
)

tweet_prompt = PromptTemplate(
    template='Summary:\n{summary}\n\n'
             'Write ONE tweet based on this summary. Max 280 characters, strong hook, '
             '1-2 hashtags. Return only the tweet.',
    input_variables=['summary']
)

linkedin_prompt = PromptTemplate(
    template='Summary:\n{summary}\n\n'
            'Write a LinkedIn post based on this summary. Professional tone, strong first line, '
            'short paragraphs, end with a question and 3 hashtags. Return only the post.',
    input_variables=['summary']
)

instagram_prompt = PromptTemplate(
    template='Summary:\n{summary}\n\n'
            'Write an Instagram caption based on this summary. Casual tone, 2-3 emojis, '
            '5 hashtags. Return only the caption.',
    input_variables=['summary']
)