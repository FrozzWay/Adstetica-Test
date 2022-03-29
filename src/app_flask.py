from flask import Flask, render_template, send_from_directory
from src.neuronet import Neuronet
from blueprints.api import bp as api
import os, re


app = Flask(__name__,
            template_folder=os.path.join(os.getcwd(), 'templates'),
            )

app.config['NET_FILES_FOLDER'] = os.path.join(os.getcwd(), 'net_files')
STATIC_FOLDER = os.path.join(os.getcwd(), 'static')

app.register_blueprint(api, url_prefix="/api")

@app.get('/')
def index_page():
    return render_template('index.html')


@app.get('/<path:page>')
def fallback(page):
    if check_for_static_files(page):
        return send_from_directory(STATIC_FOLDER, page)

    return render_template('index.html')


def check_for_static_files(page):
    regex = re.match(r"^.*(\..+)$", page)
    if regex:
        return True
