import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as processimage

# load trained model

model = load_model('./model_name.h5')



class MainPredictImg(object):
    def __init__(self): # 初始化
        pass  # 初始化语句不执行
    
    def pred(self,filename):
        # np array
        pred_img = processimage.imread(filename) # 读取照片
        pred_img = np.array(pred_img)  # 转成numpy
        pred_img = pred_img.reshape(-1, 28, 28, 1) # reshape into network needed shape
#       pred_img = pred_img/255.0
        prediction = model.predict(pred_img) #shape
        Final_prediction = [result.argmax() for result in prediction][0]
        a = 0
        for i in prediction[0]:
            print('Percent:{:.30%}'.format(i))
            a = a+1
        print(a)
        return Final_prediction

def main():
    Predict = MainPredictImg()
    res = Predict.pred('../other/7.jpg')
    print('Your number is:-->', res)

if __name__ == '__main__':
    main()
