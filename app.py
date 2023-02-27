import sys
sys.path.insert(0, 'C:\\Users\\Utente\\Desktop\\Dev\\Progetti\\OrderAi\\lib\\')
sys.path.insert(1, 'C:\\Users\\Utente\\Desktop\\Dev\\Progetti\\OrderAi\\Models\\')
sys.path.insert(2, 'C:\\Users\\Utente\\Desktop\\Dev\\Progetti\\OrderAi\\Models\\components')
from flask import Flask
from datetime import datetime
from graphics import Graphics
from preprocessing import *
from termcolor import colored, cprint

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


def main():
    print(__name__)


if __name__ == '__main__':
    main()
