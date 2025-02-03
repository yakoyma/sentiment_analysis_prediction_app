"""
===============================================================================
Multiclass Classification Project: Sentiment Analysis using a Natural
Language Processing (NLP) Classifier and Gradio application
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
import transformers
import nltk
import gradio as gr


from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          pipeline)
from nltk import word_tokenize


# Display versions of platforms and packages
print('\nPython: {}'.format(platform.python_version()))
print('Transformers: {}'.format(transformers.__version__))
print('NLTK: {}'.format(nltk.__version__))
print('Gradio: {}'.format(gr.__version__))



"""
===============================================================================
Prediction using Gradio application
===============================================================================
"""
def get_prediction(text: str):
    """This function predicts the sentiment of a text using a NLP
    Classification model.

    Args:
        text (str): the user input

    Returns:
        response (str): the predicted sentiment and the score of prediction
    """

    # Encode the text
    text = text.encode(encoding='utf-8').decode(encoding='utf-8')

    # Instantiate the model
    model_path = 'tabularisai/multilingual-sentiment-analysis'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    pipe = pipeline(
        task='text-classification',
        model=model,
        tokenizer=tokenizer
    )

    if len(tokenizer.tokenize(text)) <= 512:
        score = round(pipe(text)[0]['score'], 2)
        response = (f'The number of words in the text is '
                    f'{len(word_tokenize(text))}.'
                    + '\nThe predicted sentiment is: ' + pipe(text)[0]['label']
                    + f'.\nThe score of prediction is {score}.')
    else:
        response = (f'The number of words in the text is '
                    f'{len(word_tokenize(text))}. The text is too long.')
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
