# -*- coding: utf-8 -*-
import os

# Generate environement of the program

def create_folders() :
    """
    Create folders needed for the program
    """
    path = os.getcwd()
    path = os.path.join(path, "Env_prog")
    print("ui")
        
    
    """
    Database Folder, will contain :
        database
        img_dataset folder
            img + attribute selection
    """
    create_path = os.path.join(path, "Database", "img_dataset")
    os.makedirs(create_path)
    
    """
    Result Folder, will contain :
        Output of the program
    """
    create_path = os.path.join(path, "Result")
    os.makedirs(create_path)
    
    """
    Interface Folder, will contain :
        logo.png
    """
    create_path = os.path.join(path, "Interface")
    os.makedirs(create_path)
    
    """
    Auto-encoder Folder, will contain :
        gen_img
            .img
    """
    create_path = os.path.join(path, "Auto_encoder", "gen_img")
    os.makedirs(create_path)
    
create_folders()
  
