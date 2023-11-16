from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

# initializing of embedding & recognizer
embeddingFile = "embeddings.pickle"
# New & Empty at initial
recognizerFile = "recognizer.pickle"
labelEncFile = "le.pickle"

print("Loading face embeddings...")
data = pickle.loads(open(embeddingFile, "rb").read())

print("Encoding labels...")
labelEnc = LabelEncoder()
labels = labelEnc.fit_transform(data["names"])

print("Training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

f = open(recognizerFile, "wb")
f.write(pickle.dumps(recognizer))
f.close()

f = open(labelEncFile, "wb")
f.write(pickle.dumps(labelEnc))
f.close()
print("Process Completed")
