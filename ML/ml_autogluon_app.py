"""
===============================================================================
Multiclass Classification Project: Sentiment Analysis of cooking Recipe Reviews
 and User Feedback using Machine Learning (ML) model and Gradio application
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
import numpy as np
import pandas as pd
import nltk
import gradio as gr


from autogluon.tabular import TabularDataset, TabularPredictor
from nltk import word_tokenize


# Display versions of platforms and packages
print('\nPython: {}'.format(platform.python_version()))
print('NLTK: {}'.format(nltk.__version__))
print('Gradio: {}'.format(gr.__version__))



"""
===============================================================================
Prediction using Gradio application
===============================================================================
"""
def get_prediction(text: str):
    """This function predicts the sentiment of text using Machine Learning
    model.

    Args:
        text (str): the user input

    Returns:
        response (str): the predicted sentiment
    """

    # Encode the text
    text = text.encode(encoding='utf-8').decode(encoding='utf-8')

    # Create dataset with inputs data
    X = TabularDataset(data=pd.DataFrame(data={'text': [text]}))

    # Load the model
    model = TabularPredictor.load('models/autogluon')

    # Make prediction
    prediction = np.array(model.predict(X)).flatten()
    sentiments = {
        'Neutral': 0,
        'Very dissatisfied': 1,
        'Dissatisfied': 2,
        'Correct': 3,
        'Satisfied': 4,
        'Very satisfied': 5
    }
    for key, value in sentiments.items():
        if prediction[0] == value:
            sentiment = key
    response = (f"The number of words in the text is "
                f"{len(word_tokenize(text))}."
                f"\nThe predicted sentiment is: {sentiment}.")
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
