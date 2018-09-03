import SentimentAnalysisML as saML

def getSentiment(sentiment):
    obj = saML.SentimeAnanlysisML()
    return obj.getSentiment(sentiment)

if __name__ == "__api__":
    sys.exit(int(main() or 0))

