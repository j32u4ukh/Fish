#%%
import numpy as np
#%%
x = np.random.rand(10, 1, 28, 28)
print(x.shape)
#print(x)
#%%
from Fish.util import im2col
#%%
x1 = np.random.rand(1, 3, 7, 7)
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape) # (9, 75)
#%%
x2 = np.random.rand(10, 3, 7, 7) # 10个数据
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape) # (90, 75)
#%%

#%%

#%%
import numpy as np
#%%
array = np.array([[1, 2, 3],
                  [4, 5, 6]])
#%%
print("shape", array.shape)
print(array)
#%%

#%%

#%%
import pandas as pd
import numpy as np
#%%
left = pd.DataFrame({'A':['A0', 'A1', 'A2'],
                     'B':['B0', 'B1', 'B2']}, index=['K0', 'K1', 'K2'])
right = pd.DataFrame({'C':['C0', 'C2', 'C3'],
                      'D':['D0', 'D2', 'D3']}, index=['K0', 'K2', 'K3'])
#%%
result = pd.merge(left, right, on=['key1', 'key2'], how='right')
result
#%%
df1 = pd.DataFrame({'col1':[0, 1], 'col2':[4, 7]})
df2 = pd.DataFrame({'col1':[1, 2, 2], 'col2':[2, 2, 2]})
print(df1)
print(df2)
#%%
df3 = pd.concat([df1, df2], ignore_index=True)
print(df3)
#%%
result = pd.merge(df1, df2, on='col1', how='outer', indicator="indicator_column")
result
#%%
print(left)
print(right)
#%%
result = pd.merge(left, right, left_index=True, right_index=True, how='outer')
result
#%%
boys = pd.DataFrame({'k':['K0', 'K1', 'K2'], 'age':[1, 2, 3]})
girls = pd.DataFrame({'k':['K0', 'K0', 'K3'], 'age':[4, 5, 6]})
print(boys)
print(girls)
#%%
result = pd.merge(boys, girls, on='k', suffixes=['_boy', '_girl'], how='inner')
result
#%%
import matplotlib.pyplot as plt
#%%
data = pd.Series(np.random.randn(1000), index=np.arange(1000))
data = data.cumsum()
#%%
data = pd.DataFrame(np.random.randn(1000, 4), index=np.arange(1000), columns=list("ABCD"))
data = data.cumsum()
print(data.head())
#%%
data.plot()
plt.show()
#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%
import requests
import json
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import patches
from io import BytesIO
#%%
def annotate_image2(path=None, url=None, draw=False):
    key = "b03fd45a503940e58b140f4bdd79c4fb"
    assert key
    api = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0/detect'

    face_attributes = 'age,gender,headPose,smile,facialHair,glasses,emotion,'
    face_attributes += 'hair,makeup,occlusion,accessories,blur,exposure,noise'
    params = {
        'returnFaceId': 'true',
        'returnFaceLandmarks': 'false',
        'returnFaceAttributes': face_attributes
    }

    if path is None:
        headers = {'Ocp-Apim-Subscription-Key': key}
        response = requests.post(api, params=params, headers=headers, json={"url": url})
        img_source = requests.get(url).content
    else:
        headers = {'Ocp-Apim-Subscription-Key': key, "Content-Type": "application/octet-stream"}
        image_data = open(path, "rb").read()
        response = requests.post(api, params=params, headers=headers, data=image_data)
        img_source = image_data

    faces = response.json()
    print("Found ", len(faces), "faces")
    for face in faces:
        print(face["faceAttributes"]["gender"], face["faceAttributes"]["age"])
        emotion = face['faceAttributes']['emotion']
        js = json.dumps(emotion, sort_keys=True, indent=4, separators=(',', ':'))
        print(js)

    if draw:
        img = Image.open(BytesIO(img_source))
        plt.figure(figsize=(8, 8))
        ax = plt.imshow(img, alpha=0.6)
        for face in faces:
            fr = face["faceRectangle"]
            fa = face["faceAttributes"]
            (left, top) = (fr["left"], fr["top"])
            p = patches.Rectangle((left, top), fr["width"], fr["height"], fill=False, linewidth=2, color='b')
            ax.axes.add_patch(p)
            plt.text(left, top, "%s, %d" % (fa["gender"].capitalize(), fa["age"]),
                     fontsize=20, weight="bold", va="bottom")
        plt.axis("off")
#%%
image_url = 'https://raw.githubusercontent.com/Microsoft/Cognitive-Face-Windows/master/Data/detection1.jpg'
url1 = "https://how-old.net/Images/faces2/main001.jpg"
url2 = "https://how-old.net/Images/faces2/main002.jpg"
url3 = "https://how-old.net/Images/faces2/main004.jpg"
path1 = "/home/j32u4ukh/Documents/happy.jpeg"
#%%
annotate_image2(path=None, url=url3, draw=True)
#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%
line = ""
key_in = input(">>> ")
while key_in != "q":
    line += key_in + "\t"
    A = input("A >>> ")
    B = input("B >>> ")
    key_in = input(">>> ")
print(line, A, B)
#%%

#%%
line = "%s\t%s" % ("a", "bd")
line
#%%
key_in = input(">>>")
while key_in != "q":
    with open("C:\\Users\\etlab\\Downloads\\Questionnaire.txt", "a+") as f:
        line = ""
        _id = input("_id >>>")
        line += _id + "\t"
        school = input("school >>>")
        line += school + "\t"
        gender = input("gender >>>")
        line += gender + "\t"
        age = input("age >>>")
        line += age + "\t"
        group = input("group >>>")
        line += group + "\t"
        A = input("A >>> ")
        while len(A) != 16:
            A = input("A >>> ")
        B = input("B >>> ")
        while len(B) != 12:
            B = input("B >>> ")
        C = input("C >>> ")
        while len(C) != 10:
            C = input("C >>> ")
        D = input("D >>> ")
        while len(D) != 12:
            D = input("D >>> ")
        for i in A:
            line += i + "\t"
        for i in B:
            line += i + "\t"
        for i in C:
            line += i + "\t"
        for i in D:
            line += i + "\t"
        f.write(line + "\n")
        key_in = input(">>>")
#%%
e = "4444444434343444"
for i in e:
    print(i)
#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%
