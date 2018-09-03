import sys
import nltk
from multiprocessing import pool

class SentimeAnanlysisML:
    positiveReviews = None
    negativeReviews = None
    trainingNegativeReviews = None
    trainingPositiveReviews = None
    vocabulary = None
    trainedNBClassifier = None
    isModelCreated = False

    def __init__(self, **kwargs):
        return super().__init__(**kwargs)

    def getSentiment(self,sentiment):
        if not self.isModelCreated:
            self.isModelCreated = self.createModel()
        return self.naiveBayesSentimentCalculator(sentiment)

    def createModel(self):
        self.trainingNegativeReviews,self.trainingPositiveReviews = self.loadRawData()
        self.vocabulary = self.getVocabulary()
        trainingData = self.getTrainingData()
        self.trainedNBClassifier = self.getTrainedNaiveBayesClassifier(self.extract_features,trainingData)
        return true

    def loadRawData(self):
        positiveReviewsFileName = r"C:\Projects\Sentiment Analysis\Cornell\rt-polaritydata\rt-polarity.pos"
        negativeReviewFileName = r"C:\Projects\Sentiment Analysis\Cornell\rt-polaritydata\rt-polarity.neg"

        testTrainingSplitIndex = 2500

        with open(positiveReviewsFileName,'r') as f:
             positiveReviews = f.readlines()

        with open(negativeReviewFileName,'r') as f:
            negativeReviews = f.readlines()

        #testNegativeReviews = negativeReviews[testTrainingSplitIndex + 1:]
        #testPositiveReviews = positiveReviews[testTrainingSplitIndex + 1:]

        trainingNegativeReviews = negativeReviews[:testTrainingSplitIndex]
        trainingPositiveReviews = positiveReviews[:testTrainingSplitIndex]
        return trainingNegativeReviews,trainingPositiveReviews

    def getVocabulary(self):
        positiveWordList = [word for line in self.trainingPositiveReviews for word in line.split()]
        negativeWordList = [word for line in self.trainingNegativeReviews for word in line.split()]
        allWordList = [item for sublist in [positiveWordList,negativeWordList] for item in sublist]
        allWordSet = list(set(allWordList))
        vocabulary = allWordSet
        return vocabulary

    def getTrainingData(self):
      negTaggedTrainingReviewList = [{'review':oneReview.split(),'label':'negative'} for oneReview in self.trainingNegativeReviews] 
      posTaggedTrainingReviewList = [{'review':oneReview.split(),'label':'positive'} for oneReview in self.trainingPositiveReviews] 
      fullTaggedTrainingData = [item for sublist in [negTaggedTrainingReviewList,posTaggedTrainingReviewList] for item in sublist]
      trainingData = [(review['review'],review['label']) for review in fullTaggedTrainingData]
      return trainingData

    def naiveBayesSentimentCalculator(self,review):
         problemInstance = review.split()
         problemFeatures = self.extract_features(problemInstance)
         return self.trainedNBClassifier.classify(problemFeatures)

    def getTrainedNaiveBayesClassifier(self,extract_features, trainingData):
        trainingFeatures = nltk.classify.apply_features(extract_features,trainingData)
        #apply parallel processing here
        trainedNBClassifier = pool.Pool.apply_async(nltk.NaiveBayesClassifier.train(trainingFeatures))
        return trainedNBClassifier

    def extract_features(self,review):
        review_words = set(review)
        features = {}
        for word in self.vocabulary:
            features[word] = (word in review_words)
        return features