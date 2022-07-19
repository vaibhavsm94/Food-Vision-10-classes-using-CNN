import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras.models

from flask import Flask, render_template, request

app = Flask(__name__)

class_names = ['chicken_curry',
 'chicken_wings',
 'fried_rice',
 'grilled_salmon',
 'hamburger',
 'ice_cream',
 'pizza',
 'ramen',
 'steak',
 'sushi']

model = tf.keras.models.load_model(
       ('FoodVision10.h5'),
       custom_objects={'KerasLayer':hub.KerasLayer}
)

def load_and_prep(image_path):
  img = tf.io.read_file(image_path)
  img = tf.image.decode_image(img)

  img = tf.image.resize(img, size=[224,224])
  img = img/ 255.
  return img

def pred_and_plot(model, image_path):
  img = load_and_prep(image_path)

  pred = model.predict(tf.expand_dims(img, axis=0))

  pred_label = class_names[tf.argmax(pred[0])]

  print(f"The predicted class is {pred_label}")
  return pred_label
	
# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

    #Save image to static folder
		img_path = "static/" + img.filename	
		img.save(img_path)
    
    #Call model.predict and get the predicted label
		p = pred_and_plot(model, img_path)
	return render_template("index.html", prediction = p, img_path = img_path)
  

if __name__ =='__main__':
	app.run()
