"""
===============================================================================
Multiclass Classification Project: Sentiment Analysis application with a
Natural Language Processing (NLP) Classifier and Gradio
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
import gradio as gr


from nltk import word_tokenize
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          pipeline)


# Display versions of platforms and packages
print('\nPython: {}'.format(platform.python_version()))
print('NLTK: {}'.format(nltk.__version__))
print('Transformers: {}'.format(transformers.__version__))
print('Gradio: {}'.format(gr.__version__))



"""
===============================================================================
Prediction application with Gradio
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
    text_length = len(word_tokenize(text))


    # Instantiate the model
    model_path = 'tabularisai/multilingual-sentiment-analysis'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    pipe = pipeline(
        task='text-classification', model=model, tokenizer=tokenizer)
    input_length = len(tokenizer.tokenize(text))

    if input_length <= 512:
        score = round(pipe(text)[0]['score'], 2)
        response = (f'The number of text tokens is: {text_length}.'
                    + '\nThe predicted sentiment is: ' + pipe(text)[0]['label']
                    + f'.\nThe score of prediction is {score}.')
    else:
        response = (f'The number of text tokens is: {text_length}.'
                    f'\nThe text is too long and the maximum number of '
                    f'tokens is exceeded.')
    return response



# Instantiate the app
app = gr.Interface(
    fn=get_prediction,
    inputs='text',
    outputs='text',
    title='Sentiment Analysis Application'
)



if __name__ == '__main__':
    app.launch()
