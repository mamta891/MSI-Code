import cv2
from tensorflow.keras.models import load_model

image_green = r"/home/rajupoddar/Desktop/ML rice leaf/Green.png"
image_white = r"/home/rajupoddar/Desktop/ML rice leaf/Green.png"
image_yellow = r"/home/rajupoddar/Desktop/ML rice leaf/Green.png"
image_blue = r"/home/rajupoddar/Desktop/ML rice leaf/Green.png"
image_orange = r"/home/rajupoddar/Desktop/ML rice leaf/Green.png"
image_violet = r"/home/rajupoddar/Desktop/ML rice leaf/Green.png"
image_red = r"/home/rajupoddar/Desktop/ML rice leaf/Green.png"


def resizer(image):
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    test_input = image.reshape((1, 224, 224, 3))
    return test_input


green_model = load_model('model_green.h5')
red_model = load_model('model_red.h5')
white_model = load_model('model_white.h5')
yellow_model = load_model('model_yellow.h5')
blue_model = load_model('model_blue.h5')
orange_model = load_model('model_orange.h5')
violet_model = load_model('model_violet.h5')

prediction_green = green_model.predict(resizer(image_green))
prediction_red = red_model.predict(resizer(image_red))
prediction_white = white_model.predict(resizer(image_white))
prediction_yellow = yellow_model.predict(resizer(image_yellow))
prediction_blue = blue_model.predict(resizer(image_blue))
prediction_orange = orange_model.predict(resizer(image_orange))
prediction_violet = violet_model.predict(resizer(image_violet))

weight = [0.556, 0.5000, 0.5714, 0.8571, 0.7000, 0.6000, 0.8571]

disease_green = (prediction_green[0][0])* weight[4]
healthy_green = prediction_green[0][1] * weight[4]

disease_red = prediction_red[0][0] * weight[5]
healthy_red = prediction_red[0][1] * weight[5]

disease_white = prediction_white[0][0] * weight[1]
healthy_white = prediction_white[0][1] * weight[1]

disease_yellow = prediction_yellow[0][0] * weight[2]
healthy_yellow = prediction_yellow[0][1] * weight[2]

disease_blue = prediction_blue[0][0] * weight[3]
healthy_blue = prediction_blue[0][1] * weight[3]

disease_orange = prediction_orange[0][0] * weight[0]
healthy_orange = prediction_orange[0][1] * weight[0]

disease_violet = prediction_violet[0][0] * weight[6]
healthy_violet = prediction_violet[0][1] * weight[6]

# if disease_green > healthy_green:
#     print("Unhealthy")
# else :
#     print("Healthy")

disease = (disease_violet + disease_orange + disease_blue + disease_yellow + disease_green + disease_red + disease_white)/7

healthy = (healthy_violet + healthy_orange + healthy_blue + healthy_yellow + healthy_green + healthy_red +healthy_white)/ 7

if disease > healthy:
    print("Unhealthy")
else :
    print("Healthy")

# print(disease_green)
# print(healthy_green)
