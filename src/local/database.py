"""
Contains the code to create and manage the vectorized database of faces.
"""

import os
from deepface import DeepFace
import numpy as np


class VectorizedDatabase:
    """
    database of embeddings for the DeepFace package
    """

    def __init__(self):
        """
        initialize db
        """

        # load the images from the directory
        self.embeddings = {}
        for folder in os.listdir('./db'):
            if os.path.isfile(os.path.join('./db', folder)):
                continue

            for filename in os.listdir(os.path.join('./db', folder)):
                img_path = os.path.join('./db', folder, filename)
                img_embedding = DeepFace.represent(img_path=img_path, model_name='Facenet512')
                self.embeddings[filename.split('.')[0]] = img_embedding

        print(len(self.embeddings))

