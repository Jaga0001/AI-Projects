import os

os.chdir("C:/Users/jagaj/Desktop/Jaga/Artificial Intelligence/Day - 12 Image Classification Using CNN/Dataset/batmanJoker")
i=1
for file in os.listdir():
    src = file
    dst = "1"+"_"+str(i)+".jpg"
    os.rename(src, dst)
    i+=1