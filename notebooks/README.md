# Can you Judge a Book by its Blurb?
### _Predicting genres based on natural language processing_

The genre generator app is a machine learning-powered app that will predict the genre of a book based on it's synopsis. Whether it's a tale of romance, an epic sci-fi story or a chilling thriller book, this app aims to utilize natural language processing (NLP) to classify the story behind the summary. 

### Features
- Predicting the genre from the synopsis using a fine-tuned DistilBERT model.
- Supports the following genres currently:
            - Fantasy
            - Historical Fiction
            - Horror
            - Mystery
            - Romance
            - Science Fiction
            - Thriller
            - Western
- Simplistic interactive application interface.
- Transparent model evaluation and performance statistics.

## Dataset
Data was collected by querying the Google Books API, retrieving each book's synopsis along with it's associated genre metadata.
- Source: _Google Books API_
- Fields collected: 
        - title
        - description
        - genre
        - published date
        - author
- API call for 500 books per genre, consistently stopped early for all genres. Unfortunately was never able to get more books pulled under each genre. 

## Model Details
- Model: `DistillBertForSequenceClassification`
- Tokneizer: `DistilbertTokenizerFast`
- Dataset: 1131 book blurbs labed by genre
- Preprocessing:
        - Lowercasing
        - Punctuation removal
        - Lemmatization using NLTK's `WordNetLemmatizer`
- Training: 5 epochs
- Frameowrk: PyTorch and HuggingFace Transformers

## Performance Metrics
- **Accuracy**:  _65.37%_
- **Macro F1**:   _0.64_

## Future Improvements
- **Model Fine-tuning**: Further fine-tune the model with more diverse datasets to improve accuracy.
- **Token Attribution**: Implement token attribution methods to understand which words influence predictions.
- **User Feedback Loop**: Implement a feedback mechanism to learn from user guesses and improve predictions.
- **Multi-label Classification**: Explore multi-label classification to allow for books that fit multiple genres.
- **Less Preprocessing**: Try using raw text to see if the model performs better without heavy preprocessing.

## Acknowledgements
_A huge shout out to Michael and NSS for providing me with the skillset to create this fun app._
_Special shout out to Lexa for guidance along the way and assisting my mental fortitude._ 
- HuggingFace Transformers
- NLTK
- Google Books API

