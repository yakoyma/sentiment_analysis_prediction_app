"""
===============================================================================
Multiclass Classification Project: Sentiment Analysis application of Recipe
Reviews and User Feedback with a Machine Learning (ML) model and Gradio
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
import numpy as np
import pandas as pd
import pycaret
import nltk
import gradio as gr


from nltk import word_tokenize
from pycaret.classification import *


# Display versions of platforms and packages
print('\nPython: {}'.format(platform.python_version()))
print('NLTK: {}'.format(nltk.__version__))
print('PyCaret: {}'.format(pycaret.__version__))
print('Gradio: {}'.format(gr.__version__))



"""
===============================================================================
Prediction application with Gradio
===============================================================================
"""
def get_prediction(text: str):
    """This function predicts the sentiment of a text using a ML model.

    Args:
        text (str): the user input

    Returns:
        response (str): the predicted sentiment
    """

    # Check wether the user input is valid
    if text:

        # Encode the text
        text = text.encode(encoding='utf-8').decode(encoding='utf-8')
        text = text.strip()
        text_length = len(word_tokenize(text))

        # Create dataset with inputs data
        X = pd.DataFrame(data={'text': [text]})


        # Load the model
        model = load_model(model_name='models/pycaret/model')

        # Make prediction
        prediction = model.predict(X)


        # Display the sentiment
        sentiments = {
            'Neutral': 0,
            'Very dissatisfied': 1,
            'Dissatisfied': 2,
            'Correct': 3,
            'Satisfied': 4,
            'Very satisfied': 5
        }
        sentiment = next(
            key for key, value in sentiments.items() if prediction[0] == value)
        response = (f'The number of text tokens is: {text_length}.'
                    f'\nThe predicted sentiment is: {sentiment}.')

    else:

        response = 'Invalid input data. Please complete the field correctly.'
    return response



# Instantiate the app
app = gr.Interface(
    fn=get_prediction,
    inputs='text',
    outputs='text',
    title='Recipe Reviews and User Feedback Sentiment Analysis Application'
)



if __name__ == '__main__':
    app.launch()
