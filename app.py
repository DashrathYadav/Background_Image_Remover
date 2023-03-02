from flask import Flask ,jsonify,request,render_template
import requests
from PIL import Image
import base64
# image=Image.open('sandy_pic.jpg ')
import os
from git import Repo
import subprocess
import shutil
import numpy as np
from PIL import Image
import time
# import files
# subprocess.run(["git", "clone", "https://github.com/ZHKKKe/MODNet"])

# git_url=' https://github.com/ZHKKKe/'



app=Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')





@app.route('/upload',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['file']
        print(type(f))
        print(f)
        f.save('user_imgage')
    with open("user_imgage",'rb') as myImage:
        imagestring= base64.b64encode(myImage.read())
    print("i hitted")
    # adrr='http://127.0.0.1:5000/'
    adrr= 'https://back-ground-image-remove.onrender.com/'
    url=adrr+'serv'
    r=requests.post('https://back-ground-image-remove.onrender.com/serv',imagestring)
    print(r.json())
    time.sleep(5)
    return render_template('user_image_show.html')

@app.route('/serv',methods=['POST'])
def  serv():
        
        client_data=request.data
        print("client data is ")

        with open("server_pic.jpg",'wb+') as serv_img:
             serv_img.write(base64.b64decode(client_data))
             print("writting is done")
        
        # if not os.path.exists('MODNet'):
        #     subprocess.run(["git", "clone", "https://github.com/ZHKKKe/MODNet"])

        print("Modnet is made")

        # pretrained_ckpt = 'pretrained/modnet_photographic_portrait_matting.ckpt'

        # if os.path.exists('MODNet.demo/image_matting/colab/input'):
        #     print("You have this path")
        
        input_folder = 'MODNet/demo/image_matting/colab/input'
        if os.path.exists(input_folder):
            shutil.rmtree(input_folder)
        os.makedirs(input_folder)
        print(os.getcwd())

        output_folder = 'MODNet/demo/image_matting/colab/output'
        if os.path.exists(output_folder):
         shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        user_img='server_pic.jpg'
        shutil.move('server_pic.jpg', os.path.join(input_folder, 'server_pic.jpg'))

        

        # if os.path.exists('MODNet.demo/image_matting/colab/input'):
        #     print("You have this path")
        # else :
        #     print("no path exist")
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
            print("foreground type is")
            print("foreground type is",type(foreground))
            foreground_img = Image.fromarray(np.uint8(foreground))
            foreground_img.save("foreground.jpg")
            print("The path of os  is",os.getcwd())
            shutil.move('foreground.jpg', os.path.join('static','foreground.jpg'))
            return combined

        # visualize all images
        image_names = os.listdir(input_folder)
        print(image_names)
        for image_name in image_names:
            matte_name = image_name.split('.')[0] + '.png'
            print(matte_name)
            image = Image.open(os.path.join(input_folder, image_name))
            matte = Image.open(os.path.join(output_folder, matte_name))
            (combined_display(image, matte))
            # print(image_name, '\n')
            
            
        zip_filename = 'matte.zip'
        if os.path.exists(zip_filename):
            os.remove(zip_filename)
        os.system(f"zip -r -j {zip_filename} {output_folder}/*")
        print("server working")
        return {
            'status':'ok',
            'message':' I am server'
               }


# with open("sandy_pic.jpg",'rb') as myImage:
#     imagestring= base64.b64encode(myImage.read())
# # print(imagestring) 
# # image.show()
port = int(os.environ.get('PORT', 5000))
if __name__ == "__main__":
    app.run(port=port)
