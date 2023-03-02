from flask import Flask, jsonify, request, render_template
import requests
import base64
import os
import subprocess
import shutil
import numpy as np
from PIL import Image
import time

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        print(type(f))
        print(f)
        f.save('user_image.jpg')
        print('File saved')
    
    with open('user_image.jpg', 'rb') as myImage:
        imagestring = base64.b64encode(myImage.read())

    adrr = 'https://back-ground-image-remove.onrender.com/'
    adrr='http://127.0.0.1:5000/'
    # adrr = adrr + 'serv'
    r = requests.post(adrr + 'serv', imagestring)
    print(r.json())

    time.sleep(5)
    return render_template('user_image_show.html')

@app.route('/serv', methods=['POST'])
def serv():
    client_data = request.data

    with open('server_image.jpg', 'wb+') as serv_img:
         serv_img.write(base64.b64decode(client_data))

    input_folder = 'MODNet/demo/image_matting/colab/input'
    if os.path.exists(input_folder):
        shutil.rmtree(input_folder)
    os.makedirs(input_folder)

    output_folder = 'MODNet/demo/image_matting/colab/output'
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    shutil.move('server_image.jpg', os.path.join(input_folder, 'server_image.jpg'))

    subprocess.run([
        'python', 
        '-m', 
        'MODNet.demo.image_matting.colab.inference',
        '--input-path', 
        'MODNet/demo/image_matting/colab/input',    
        '--output-path', 
        'MODNet/demo/image_matting/colab/output',
        '--ckpt-path',
        'MODNet/pretrained/modnet_photographic_portrait_matting.ckpt'
    ])

    def combined_display(image, matte):
        # calculate display resolution
        w, h = image.width, image.height
        rw, rh = 800, int(h * 800 / (3 * w))

        # obtain predicted foreground
        image = np.asarray(image)
        if len(image.shape) == 2:
            image = image[:, :, None]
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] == 4:
            image = image[:, :, 0:3]
        matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
        foreground = image * matte + np.full(image.shape, 255) * (1 - matte)

        # combine image, foreground, and alpha into one line
        combined = np.concatenate((image, foreground, matte * 255), axis=1)
        combined = Image.fromarray(np.uint8(combined)).resize((rw, rh))

        foreground_img = Image.fromarray(np.uint8(foreground))
        foreground_img.save('static/foreground.jpg')

        return combined

    image_names = os.listdir(input_folder)
    print(image_names)
    for image_name in image_names:
        matte_name = image_name.split('.')[0] + '.png'
        print(matte_name)
        image = Image.open(os.path.join(input_folder, image_name))
        matte = Image.open(os.path.join(output_folder, matte_name))
        combined_display(image, matte)

    return {
            'status':'ok',
            'message':' I am server'
               }

port = int(os.environ.get('PORT', 5000))
if __name__ == "__main__":
    app.run(port=port)