"""
===============================================================================
Multiclass Classification Project: Sentiment Analysis using an Artificial
Intelligence (AI) model and Gradio application
===============================================================================

This file is organised as follows:
Prediction using Gradio application
"""
# Standard libraries
import platform
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Other libraries
import langchain_community
import langchain_core
import gradio as gr


from nltk import word_tokenize
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Display versions of platforms and packages
print('\nPython: {}'.format(platform.python_version()))
print('LangChain Community: {}'.format(langchain_community.__version__))
print('LangChain Core: {}'.format(langchain_core.__version__))
print('Gradio: {}'.format(gr.__version__))



"""
===============================================================================
Prediction using Gradio application
===============================================================================
"""
def get_prediction(text: str):
    """This function predicts the sentiment of the text using an AI model.

    Args:
        text (str): the user input

    Returns:
        response (str): the response of the model with the predicted sentiment
    """

    # Encode the text
    text = text.encode(encoding='utf-8').decode(encoding='utf-8')

    # Instantiate the model
    model = Ollama(
        model='zephyr:7b',
        temperature=0.8,
        top_k=50,
        top_p=0.95
    )
    input = (f"You are a nice helpful assistant and your role is to provide "
             f"concisely the Sentiment Analysis of the text: '{text}' by "
             f"selecting one of the following word: Very Negative, Negative, "
             f"Neutral, Correct, Positive, and Very Positive.")
    prompt = PromptTemplate.from_template('{input}')
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({'input': input})
    return response


# Instantiate the app
app = gr.Interface(
    fn=get_prediction,
    inputs='text',
    outputs='text',
    title='Sentiment Analysis'
)



if __name__ == '__main__':
    app.launch()
