import os

DATABASE = os.path.join(os.path.dirname(__file__), 'instance', 'app.db')
SECRET_KEY = os.urandom(24)
