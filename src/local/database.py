"""
contains the code to create and manage the vectorized database of faces
"""

import os
from deepface import DeepFace
import numpy as np


class VectorizedDatabase:
    """
    database of embeddings for the DeepFace package
    """

    def __init__(self, db_path: str = './db') -> None:
        """
        initialize db

        Args:
            db_path: path to the database

        Returns:
            None
        """
        self.db_path = db_path

        # load the images from the directory
        self.embeddings = {}
        self.filenames = []

        try:
            for folder in os.listdir(db_path):
                if os.path.isfile(os.path.join(db_path, folder)):
                    continue
                for filename in os.listdir(os.path.join(db_path, folder)):
                    self.filenames.append(filename)
                    img_path = os.path.join(db_path, folder, filename)
                    img_embedding = DeepFace.represent(img_path=img_path, model_name='Facenet512')
                    self.embeddings[filename.split('.')[0]] = img_embedding
        except FileNotFoundError:
            print(f"Error: Folder not found: {self.db_path}")

    def __str__(self) -> str:
        """
        string representation of the database

        Returns:
            string representation of the database
        """
        return 'vae victis'


