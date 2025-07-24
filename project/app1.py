from flask import Flask, request, render_template, send_from_directory, url_for
import os
from PIL import Image
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Dummy image generator (simulate G1, G2, final)
def generate_fake_output(input_path):
    img = Image.open(input_path).convert("L").resize((128, 128))
    img_arr = np.array(img)

    # Just make variations for testing
    g1 = Image.fromarray((img_arr * 0.8).astype(np.uint8))
    g2 = Image.fromarray((img_arr * 1.2).clip(0, 255).astype(np.uint8))
    final = Image.fromarray(((img_arr * 0.5) + (img_arr * 1.5)).clip(0, 255).astype(np.uint8))

    g1.save(os.path.join(UPLOAD_FOLDER, 'g1_output.png'))
    g2.save(os.path.join(UPLOAD_FOLDER, 'g2_output.png'))
    final.save(os.path.join(UPLOAD_FOLDER, 'final_output.png'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)

            generate_fake_output(input_path)

            return render_template('index.html',
                                   input_image=filename,
                                   g1='g1_output.png',
                                   g2='g2_output.png',
                                   final='final_output.png')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
