import sys
import nltk

positiveReviewsFileName = r"C:\Projects\Sentiment Analysis\Cornell\rt-polaritydata\rt-polarity.pos"
negativeReviewFileName = r"C:\Projects\Sentiment Analysis\Cornell\rt-polaritydata\rt-polarity.neg"

testTrainingSplitIndex = 2500

with open(positiveReviewsFileName,'r') as f:
     positiveReviews = f.readlines()

with open(negativeReviewFileName,'r') as f:
    negativeReviews = f.readlines()

testNegativeReviews = negativeReviews[testTrainingSplitIndex + 1:]
testPositiveReviews = positiveReviews[testTrainingSplitIndex + 1:]

trainingNegativeReviews = negativeReviews[:testTrainingSplitIndex]
trainingPositiveReviews = positiveReviews[:testTrainingSplitIndex]


def main():
    naiveBayesSentimentCalculator('What an awesome movie')
    naiveBayesSentimentCalculator('What a terrible movie')
    runDiagnostics(getTestReviewSentiments(naiveBayesSentimentCalculator))

def getVocabulary():
    positiveWordList = [word for line in trainingPositiveReviews for word in line.split()]
    negativeWordList = [word for line in trainingNegativeReviews for word in line.split()]
    allWordList = [item for sublist in [positiveWordList,negativeWordList] for item in sublist]
    allWordSet = list(set(allWordList))
    vocabulary = allWordSet
    return vocabulary

def getTrainingData():
  negTaggedTrainingReviewList = [{'review':oneReview.split(),'label':'negative'} for oneReview in trainingNegativeReviews] 
  posTaggedTrainingReviewList = [{'review':oneReview.split(),'label':'positive'} for oneReview in trainingPositiveReviews] 
  fullTaggedTrainingData = [item for sublist in [negTaggedTrainingReviewList,posTaggedTrainingReviewList] for item in sublist]
  trainingData = [(review['review'],review['label']) for review in fullTaggedTrainingData]
  return trainingData

def naiveBayesSentimentCalculator(review):
     problemInstance = review.split()
     problemFeatures = extract_features(problemInstance)
     return trainedNBClassifier.classify(problemFeatures)

def getTrainedNaiveBayesClassifier(extract_features, trainingData):
    trainingFeatures = nltk.classify.apply_features(extract_features,trainingData)
    trainedNBClassifier = nltk.NaiveBayesClassifier.train(trainingFeatures)
    return trainedNBClassifier

def extract_features(review):
    review_words = set(review)
    features = {}
    for word in vocabulary:
        features[word] = (word in review_words)
    return features

vocabulary = getVocabulary()
trainingData = getTrainingData()
trainedNBClassifier = getTrainedNaiveBayesClassifier(extract_features,trainingData)

def getTestReviewSentiments(naiveBayesSentimentCalculator):
    testNegResults = [naiveBayesSentimentCalculator(review) for review in testNegativeReviews]
    testPosResults = [naiveBayesSentimentCalculator(review) for review in testPositiveReviews]
    labelToNum = {'positive':1,'negative':-1}
    numericNegResults = [labelToNum[x] for x in testNegResults]
    numericPosResults = [labelToNum[y] for x in testPosResults]
    return {'results-on-positive':numericPosResults,'results-on-negative':numericNegResults}

def runDiagnostics(reviewResults):
    positiveReviewsResult = reviewResult['result-on-positive']
    negativeReviewResult = reviewResults['result-on-negative']
    numTruePositive = sum(x > 0 for x in positiveReviewsResult)
    numTrueNegative = sum(x > 0 for x in negativeReviewResult)
    pctTruePositive = float(numTruePositive) / len(positiveReviewsResult)
    pctTrueNegative = float(numTrueNegative) / len(negativeReviewResult)
    totalAccurate = numTruePositive + numTrueNegative
    total = len(positiveReviewsResult) + len(negativeReviewResult)
    print('Accuracy on positive reviews =' + '%.2f' % (pctTruePositive * 100) + '%')
    print('Accuracy on negative reviews =' + '%.2f' % (pctTrueNegative * 100) + '%')
    print('Overall accuracy =' + '%.2f' % (totalAccurate * 100 / total) + '%')

naiveBayesSentimentCalculator("What an awesome movie")
naiveBayesSentimentCalculator("What a terrible movie")

runDiagnostics(getTestReviewSentiments(naiveBayesSentimentCalculator))

if __name__ == "__main__":
    sys.exit(int(main() or 0))