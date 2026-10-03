import streamlit as st
from chains import pipeline

st.set_page_config(page_title='AI Content Repurposing Pipeline', page_icon='✍️')
st.title('AI Content Repurposing Pipeline')
st.caption('One topic in, content for three platforms out. Built with LangChain.')

topic = st.text_input('Enter a topic', placeholder='e.g. black holes, the latest iPhone, why cats purr')

if st.button('Generate') and topic.strip():
    with st.spinner('Classifying, writing, summarizing and generating...'):
        result = pipeline.invoke({'topic': topic})

    st.info(f"Detected category: **{result['category']}**")

    with st.expander('See the summary used for all platforms'):
        st.write(result['summary'])

    st.subheader('Tweet')
    st.write(result['tweet'])
    if result['tweet_ok']:
        st.caption(f"{result['tweet_length']}/280 characters")
    else:
        st.warning(f"Too long for a tweet: {result['tweet_length']}/280 characters")

    st.subheader('LinkedIn')
    st.write(result['linkedin'])

    st.subheader('Instagram')
    st.write(result['instagram'])