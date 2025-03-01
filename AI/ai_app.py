"""
===============================================================================
Multiclass Classification Project: Sentiment Analysis application with an
Artificial Intelligence (AI) model and Gradio
===============================================================================

This file is organised as follows:
Prediction application with Gradio
"""
# Standard libraries
import platform
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Other libraries
import nltk
import transformers
import langchain_community
import langchain_core
import gradio as gr


from nltk import word_tokenize
from transformers import AutoTokenizer
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Display versions of platforms and packages
print('\nPython: {}'.format(platform.python_version()))
print('NLTK: {}'.format(nltk.__version__))
print('Transformers: {}'.format(transformers.__version__))
print('LangChain Community: {}'.format(langchain_community.__version__))
print('LangChain Core: {}'.format(langchain_core.__version__))
print('Gradio: {}'.format(gr.__version__))



"""
===============================================================================
Prediction application with Gradio
===============================================================================
"""
def get_response(text: str):
    """This function predicts the sentiment of a text using an AI model.

    Args:
        text (str): the user input

    Returns:
        response (str): the response of the model with the text's sentiment
    """

    # Encode the text
    text = text.encode(encoding='utf-8').decode(encoding='utf-8')
    text_length = len(word_tokenize(text))


    # Check wether the user input is valid
    if text:

        # Instantiate the tokenizer
        tokenizer_path = 'HuggingFaceH4/zephyr-7b-beta'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


        # Create the template for the user prompt
        input = (f"You are a nice helpful assistant and your role is to "
                 f"provide concisely the Sentiment Analysis of the text: "
                 f"'{text}' by selecting one of the following word: "
                 f"Very Negative, Negative, Neutral, Correct, Positive, and "
                 f"Very Positive.")
        input_length = len(tokenizer.tokenize(input))


        # Check whether the number of tokens at the model input is less
        # than the maximum value selected
        if input_length <= 8192:

            # Select the context window size
            num_ctx = next(
                pow(2, i) for i in range(11, 14) if input_length <= pow(2, i))

            # Instantiate the model
            model = Ollama(
                model='zephyr:7b-beta',
                num_ctx=num_ctx,
                temperature=0.6,
                top_k=50,
                top_p=0.95
            )
            prompt = PromptTemplate.from_template('{input}')
            chain = prompt | model | StrOutputParser()
            response = chain.invoke({'input': input})

        else:

            response = (f'The number of text tokens is: {text_length}.'
                        f'\nThe text is too long and the maximum number of '
                        f'tokens is exceeded.')

    else:

        response = 'Invalid input data. Please complete the field correctly.'
    return response



# Instantiate the app
app = gr.Interface(
    fn=get_response,
    inputs='text',
    outputs='text',
    title='Sentiment Analysis Application'
)



if __name__ == '__main__':
    app.launch()
