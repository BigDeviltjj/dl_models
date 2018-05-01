import skimage
import skimage.io
import skimage.transform
import numpy as np



def load_image(path):
    img = skimage.io.imread(path)
    xx,yy = img.shape[:2]

    img = img /255.0
    min_size = min([xx,yy])
    xx = int((xx - min_size)/2)
    yy = int((yy - min_size)/2)
    img = img[xx:xx+min_size,yy:yy+min_size,:]
    img = skimage.transform.resize(img,(224,224))
    return img

def print_prob(prob,path):
    with open(path) as f:
        synset = [l for l in f.readlines()]
    
    pred = np.argsort(prob)[::-1]
    print((synset[pred[0]],prob[pred[0]]))
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return synset[pred[0]]
