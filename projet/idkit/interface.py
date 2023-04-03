from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
from tkinter.messagebox import *
from tkinter import ttk
import os
import utils
import database
import algen
import logging


def f1(env_path):
    """
    Creates the window 1 from the execution of main.py 

    Parameter 
    ---------
    env_path : <string> : the relative path to file Setup_dev 

    """

    def aide():
        """
        Event linked to the "Aide" button : displays a pannel "Aide" when the user clicks on the button 
        """
        showinfo('Aide', """Bienvenue dans Idkit, le logiciel de génération de\
 portrait robot ! \n\nAfin de lancer le programme, appuyer sur Commencer.\
 \n\nUne nouvelle fenêtre apparaitra afin de vous permettre de sélectionner\
 les attributs que présente le suspect. \n\nVeillez à  séléctionner\
 judicieusement ces attributs car ils permettront de choisir les images\
 ressemblant le plus à la personne souhaitée. \n\nUne fois les attributs\
 sélectionnés, plusieurs images vous seront présentées et vous devrez\
 sélectionner les 3 images les plus pertinentes à chaque fois. \nSi une\
 des images présentée vous semble convenable, vous pouvez arrêter la\
 recherche en séléctionnant une unique photo et en sélectionnnant\
 'Image finale'. \n\nAinsi la fenêtre d'export apparaitra et en utilisant\
 le menu déroulant en haut à gauche, vous pourrez:\n\n\
    - Exporter la photo à l'endroit de votre choix\n\n
    - Recommencer une nouvelle recherche\n\n
    - Quitter le programme""")

    def openf2():
        """
        Event linked to the "Commencer" button : destroys the current window and opens the window 2 
        """
        f1_acc.destroy()
        f2(env_path)

    # creation of the window

    f1_acc = Tk()

    w, h = f1_acc.winfo_screenwidth(), f1_acc.winfo_screenheight()
    f1_acc.geometry("%dx%d" % (w, h))
    f1_acc.configure(bg='white')

    logo_path = utils.get_path(env_path, "Interface")
    logo_path = os.path.join(logo_path, "idkit.png")

    framelogo = Frame(f1_acc, width=400, height=400)
    framelogo.pack()
    framelogo.place(anchor='center', relx=0.5, rely=0.45)
    logo = ImageTk.PhotoImage(Image.open(logo_path))
    label_imagee = Label(framelogo, image=logo)
    label_imagee.pack()

    canvas = Canvas(f1_acc, width=1800, height=100, bg='ivory')
    canvas.pack(side=TOP, padx=5, pady=5)
    txtitre = canvas.create_text(500, 60, text="IdKit",
                                 font="Arial 50 italic", fill="green")
    txt = canvas.create_text(750, 75,
                             text="Le logiciel de constitution de portaits-robots",
                             font="Arial 12 italic", fill="green")

    boutS = Button(f1_acc, text="Commencer", font='Arial 13',
                   borderwidth=4, bg='#BDECB6', padx=5, pady=5,
                   command=openf2)
    boutS.place(anchor=tk.W, relheight=0.1, relwidth=0.15, relx=0.3, rely=0.8)

    boutH = Button(text='Aide', command=aide, font='Arial 13',
                   borderwidth=4, bg="#D2B48C")
    boutH.place(anchor=tk.E, relheight=0.1, relwidth=0.15, relx=0.7, rely=0.8)

    f1_acc.mainloop()



def f2(env_path):
    """
    Creates the window 1 from the execution of openf2 

    Parameter 
    ---------
    env_path : <string> : the relative path to file Setup_dev 
    """

    def recup_RB_genre():
        """
        Event : get the value linked to the user's input for the widget
        Button 'Genre'
        Returns
        -------
        rbg : <int> : 1 if woman, 2 if man, 0 if unknown
        """
        rbg = vari.get()
        return rbg

    def recup_RB_age():
        """
        Event : get the value linked to the user's input for the widget
        Button 'Age'
        Returns
        -------
        rba : <int> : 3 if young, 4 if old, 0 if unknown
        """
        rba = valeur.get()
        return rba

    def recup_valCBX(menucombo):
        """
        Event : get the value linked to the user's input for the widget
        Button 'Age'
        Parameters
        ----------
        The assiociated combobox
        Returns
        ----------
        cb : <string> : the value chosen by the user
        """
        cb = menucombo.get()
        return cb

    def recup_valCkB():
        """
        Event : get the value linked to the user's input for the
        checkButtons 'Accessoires'
        Returns
        -------
        liste_acc : <list of int> : for each value, 1 if the checkbutton
         was selected, 0 if not

        """
        lunet = vlun.get()
        mous = vmous.get()
        brd = vbrd.get()
        auc = vauc.get()
        nsp = vnsp.get()
        liste_acc = [lunet, mous, brd, auc, nsp]
        return liste_acc

    def verif_reponses():
        """
        Verify that all the questions were answered by the user
        Returns
        -------
        stop : <boolean> : True if one of the questions was left
        unaswered answered, False if all were answered
        """
        g = recup_RB_genre()
        a = recup_RB_age()
        c = recup_valCBX(menucombo)
        acc = recup_valCkB()
        stop = FALSE
        if g is None or a is None or c == 'Veuillez choisir un élément' or \
                acc == [0, 0, 0, 0, 0]:
            stop = TRUE

        testliste = liste_db()
        return stop

    def liste_db():
        """
        Returns a 2D array that corresponds to the selection of the
        parameters by the user
        Returns
        -------
        liste_acc : <2D array> : containing the values associated to the
         inputs of the user : 1 if present, 2 if not, 0 if unkown
        """
        g = recup_RB_genre()
        a = recup_RB_age()
        c = recup_valCBX(menucombo)
        acc = recup_valCkB()

        liste_acc = [['Genre', 'Age', "Cheveux", "Lunettes",
                      "Moustache", "Barbe"], [0, 0, 0, 0, 0, 0]]

        # test for the genre
        if (g == '1'):  
            liste_acc[1][0] = 1
        elif (g == '2'):
            liste_acc[1][0] = 2

        # test for the age
        if (a == '3'):  
            liste_acc[1][1] = 1
        elif (a == '4'):
            liste_acc[1][1] = 2

        # test for the hair
        list_cheveux = ["Noirs", "Blonds", "Bruns", "Gris", "Chauve", "Autre"]
        for i in range(len(list_cheveux)):
            if (list_cheveux[i] == c):
                liste_acc[1][2] = i+1

        # test for the accessoiries
        for i in range(3, 5):
            if (acc[i-3] == 1): 
                liste_acc[1][i] = 1
            else:
                if acc[4] == 1:
                    liste_acc[1][i] = 0
                else:  
                    liste_acc[1][i] = 2
        if acc[3] == 1: 
            liste_acc[1][3] = 2
            liste_acc[1][4] = 2
            liste_acc[1][5] = 2

        return liste_acc

    def openf3(env_path):
        """
        Event linked to the "Envoyer" button : destroys the current
        window and opens the window
        """

        test = verif_reponses()
        ans_user = liste_db()
        logging.info('Metadata summitted')

        # if some questions were not answered, an error pops up and the
        # user must complete the questions
        if test == FALSE:
            # Retrieve 5 images according to the answers
            array_metadata = database.create_query_array(ans_user[1][0],
                                                         ans_user[1][1],
                                                         ans_user[1][2],
                                                         ans_user[1][3],
                                                         ans_user[1][4],
                                                         ans_user[1][5])
            
            img_list = database.get_5_img(env_path, array_metadata)
            f2_flr.destroy()
            f3(env_path, img_list)
        elif test == TRUE:
            showinfo('ATTENTION', 'Veuillez remplir tous les champs.')


    # creation of the window

    f2_flr = Tk()
    w, h = f2_flr.winfo_screenwidth(), f2_flr.winfo_screenheight()
    f2_flr.geometry("%dx%d" % (w, h))
    f2_flr.configure(bg='white')

    boutSend = Button(f2_flr, text="Envoyer", font='Arial 12', height=2,
                      width=20, borderwidth=4, bg='#BDECB6',
                      command=lambda: openf3(env_path))
    boutSend.place(anchor=tk.N, relheight=0.07,
                   relwidth=0.10, relx=0.5, rely=0.8)

    labelT = Label(f2_flr, text="""Ce formulaire vise à affiner la base de\
 données pour vous présenter les solutions les plus pertinentes dans un\
 temps minimal.""", bg="white", font="Arial 14 italic")
    labelT.pack()

    labelSEXE = Label(f2_flr, text="Quel est le genre de l'individu ?",
                      font='Helvetica 12 bold')
    labelSEXE.pack()

    vari = StringVar()
    bF = Radiobutton(f2_flr, text="Femme", font='Helvetica 12',
                     variable=vari, value=1, command=recup_RB_genre,
                     bg='white')
    bH = Radiobutton(f2_flr, text="Homme", font='Helvetica 12',
                     variable=vari, value=2, command=recup_RB_genre,
                     bg='white')
    bg_nsp = Radiobutton(f2_flr, text="Ne sais pas", font='Helvetica 12',
                         variable=vari, value=0, command=recup_RB_genre,
                         bg='white')
    bF.pack()
    bH.pack()
    bg_nsp.pack()

    labelAGE = Label(f2_flr, text="Quelle tranche d'âge ?",
                     font='Helvetica 12 bold')
    labelAGE.pack()

    valeur = StringVar()
    bJ = Radiobutton(f2_flr, text="Jeune", font='Helvetica 12',
                     variable=valeur, value=3, command=recup_RB_age,
                     bg='white')
    bA = Radiobutton(f2_flr, text="Âgé", font='Helvetica 12',
                     variable=valeur, value=4, command=recup_RB_age,
                     bg='white')
    bA_nsp = Radiobutton(f2_flr, text="Ne sais pas", font='Helvetica 12',
                         variable=valeur, value=0,
                         command=recup_RB_age, bg='white')
    bJ.pack()
    bA.pack()
    bA_nsp.pack()

    labelChoix = tk.Label(
        f2_flr, text=" Quelle couleur de cheveux ?", font='Helvetica 12 bold')
    labelChoix.pack()
    listtest = ["Veuillez choisir un élément", "Noirs", "Blonds",
                "Bruns", "Gris", "Chauve", "Autre", "Ne sais pas"]
    menucombo = ttk.Combobox(f2_flr, values=listtest, font='Helvetica 12')
    menucombo.current(0)
    menucombo.pack()

    vlun = IntVar()
    vmous = IntVar()
    vbrd = IntVar()
    vauc = IntVar()
    vnsp = IntVar()
    labelChoix = tk.Label(f2_flr, text="Veuillez cocher les accessoires particuliers:",
                          font='Helvetica 12 bold')
    labelChoix.pack()
    boutLun = Checkbutton(f2_flr, text="Lunettes", font='Helvetica 12',
                          variable=vlun, onvalue=1, offvalue=0,
                          command=recup_valCkB, bg='white')
    boutMoust = Checkbutton(f2_flr, text="Moustache", font='Helvetica 12',
                            variable=vmous, onvalue=1, offvalue=0,
                            command=recup_valCkB, bg='white')
    boutbrd = Checkbutton(f2_flr, text="Barbe", font='Helvetica 12',
                          variable=vbrd, onvalue=1, offvalue=0,
                          command=recup_valCkB, bg='white')
    boutauc = Checkbutton(f2_flr, text="Aucun", font='Helvetica 12',
                          variable=vauc, onvalue=1, offvalue=0,
                          command=recup_valCkB, bg='white')
    boutnsp = Checkbutton(f2_flr, text="Ne sais pas", font='Helvetica 12',
                          variable=vnsp, onvalue=1, offvalue=0,
                          command=recup_valCkB, bg='white')

    boutLun.pack()
    boutMoust.pack()
    boutbrd.pack()
    boutauc.pack()
    boutnsp.pack()

    f2_flr.mainloop()




def f3(env_path, img_list):
    """
    Creates the window 3 from the execution of openf3

    Parameters : 
    ------------
    env_path : str
        The relative path to file Setup_dev
    img_list : list[str]
        A list containing the paths of each image to display
        (from the database, or the auto-encoder)
    """

    def recup_valCheckB():
        """
        Event : gets the index of the images selected by the user in the
         CheckButton widgets
        Returns
        -------
        rep_tot: list
            Contient l'ensemble des valeurs des checkbuttons après
            que l'utilisateur ait fait son choix
        """
        repb1 = vb1.get()
        repb2 = vb2.get()
        repb3 = vb3.get()
        repb4 = vb4.get()
        repb5 = vb5.get()
        rep_finale = vfinal.get()
        rep_tot = [repb1, repb2, repb3, repb4, repb5, rep_finale]
        return rep_tot

    def verif_rep_3():
        """
        Checks if 3 images where chosen by the user 

        Returns
        -------
        stop : bool
            TRUE if the user chose exactly 3 images
        index_choix : list[int]
            list containing the indexes of the chosen images
        """
        checkbut = recup_valCheckB()
        compte = 0
        index_choix = []
        for i in range(len(checkbut) - 1):
            if checkbut[i] == 1:
                compte += 1
                index_choix.append(i)
        stop = FALSE
        if compte == 3:
            stop = TRUE
        return stop, index_choix

    def verif_rep_1():
        """
        Checks if 1 image was chosen by the user, and which one

        Returns
        -------
        stop : <boolean> : TRUE if the user has chosen exactly 1 image 
        final : <boolean> : TRUE if the user check the "final" checkButton
        index : <int> : the index of the selected image 
        """
        checkbfinal = recup_valCheckB()
        index = 0
        compte = 0
        for i in range(len(checkbfinal) - 1):
            if checkbfinal[i] == 1:
                compte += 1
                index = i
        stop = FALSE
        finale = FALSE
        for i in range(len(checkbfinal) - 1):
            if compte == 1:
                stop = TRUE
            if checkbfinal[5] == 1:
                finale = TRUE
        return stop, finale, index

    def chemin_choix(index_choix):
        """
        Gets the paths of all the selected images 
        Parameter
        -------
        index_choix : list[int]
            List containing the indexes of the selected images
        Returns
        -------
        liste_chemin : list[str]
            List containing the paths
        """
        liste_chemin = []
        for i in range(len(img_list)):
            if (i == index_choix[0]) or (i == index_choix[1]) or \
                    (i == index_choix[2]):
                liste_chemin.append(img_list[i])
        return liste_chemin

    def openf3(env_path, img_list):
        """
        Opens the current window (f3) : gives a illusion of refreshing
        the window
        Parameters
        --------
        env_path : str
            Relative path to file Setup_dev
        img_list : list[str]
            List containing the paths towards the images to display
        """
        f3(env_path, img_list)

    def openf4(env_path):
        """
        Event  linked to the "Valider" button : depending on the number
        of images the user selected, refresh the window 3 or opens
        the window 4
        """
        refresh_3 = verif_rep_3()
        pass_4 = verif_rep_1()
        i_fin = pass_4[2]
        

        # if 3 images were selected : open f3 and display the new images 
        # produced by the genetic algorithm
        if (refresh_3[0] == TRUE and pass_4[1] == FALSE):
            
            idx_chx = verif_rep_3()
            img_path = chemin_choix(idx_chx[1])
            
            # wait for the new images to be created and stored
            img_ready = algen.create_new_images(img_path, env_path)
            
            if (img_ready==TRUE):
                logging.info('Creating 5 new images')
                chemin_dossier = utils.get_path(env_path, 'gen_img')
                # replace the old paths by the new
                for i in range(len(img_list)):
                    nom_img_ac = "image"+str(i)+".png"
                    img_list[i] = os.path.join(chemin_dossier, nom_img_ac)

            f3_img.destroy()
            openf3(env_path, img_list)

        # if 1 image was selected : close f3 and open f4
        elif (pass_4[0] == TRUE and pass_4[1] == TRUE):
            path_final_img = img_list[i_fin]
            f3_img.destroy()
            f4(env_path, path_final_img)

        # if the user selected the images incorrectly : display a message of error
        elif ((pass_4[0] == FALSE and refresh_3[0] == FALSE) or
              (pass_4[0] == TRUE and pass_4[1] == FALSE) or
              (pass_4[0] == FALSE and pass_4[1] == TRUE) or
              (refresh_3[0] == TRUE and pass_4[1] == TRUE)):
            showinfo('ATTENTION', '''Veuillez ne sélectionner qu'une image et\
                      cocher la case "Finale" ou sélectionner 3 images.''')

    def aidef3():
        """
        Event linked to the "Aide" button : display an "Aide" pannel when
        the user clicks on the button
        """
        showinfo('Aide', """Veuillez choisir les images correspondant le mieux\
 au suspect parmi les images proposées. Choisissez les 3 images\
 les plus exactes jusqu'à ce que l'une d'elles vous satisfasse.\
 Pour sélectionner l'image finale, veuillez cocher la case\
 "Finale" pour confirmer votre choix.""")

    # creation of the 3rd window
    f3_img = Tk()
    w, h = f3_img.winfo_screenwidth(), f3_img.winfo_screenheight()
    f3_img.geometry("%dx%d" % (w, h))
    f3_img.configure(bg='white')

    boutHelp = Button(text='Aide', command=aidef3,
                      font='Arial 14', borderwidth=4, bg="#D2B48C")
    boutHelp.place(anchor=tk.E, relheight=0.1,
                   relwidth=0.1, relx=0.6, rely=0.75)

    vfinal = IntVar()
    checkbfinal = Checkbutton(f3_img, text="Finale", font='Helvetica 10',
                              variable=vfinal, onvalue=1, offvalue=0,
                              bg='white')
    checkbfinal.place(anchor=tk.E, relheight=0.1, relwidth=0.1,
                      relx=0.6, rely=0.65)

    # Creation of an error message if less than 5 images can fit the user's criteres
    if img_list == 0:
        showinfo('ATTENTION', """Il n'existe pas assez d'images correspondantes\
 à cette sélection dans la base de données. Veuillez élargir\
 vos critères en sélectionnant "ne sais pas" pour certains\
 attributs.""")
        f3_img.destroy()
        f2(env_path)

    frame = Frame(f3_img, width=200, height=200)
    frame.place(anchor='center', relx=0.10, rely=0.4)
    img = Image.open(img_list[0])
    resized_image = img.resize((200, 200), Image.ANTIALIAS)
    new_image = ImageTk.PhotoImage(resized_image)
    label = Label(frame, image=new_image)
    label.pack()

    frame2 = Frame(f3_img, width=200, height=200)
    frame2.place(anchor='center', relx=0.3, rely=0.4)
    img2 = Image.open(img_list[1])
    resized_image2 = img2.resize((200, 200), Image.ANTIALIAS)
    new_image2 = ImageTk.PhotoImage(resized_image2)
    label2 = Label(frame2, image=new_image2)
    label2.pack()

    frame3 = Frame(f3_img, width=200, height=200)
    frame3.place(anchor='center', relx=0.5, rely=0.4)
    img3 = Image.open(img_list[2])
    resized_image3 = img3.resize((200, 200), Image.ANTIALIAS)
    new_image3 = ImageTk.PhotoImage(resized_image3)
    label3 = Label(frame3, image=new_image3)
    label3.pack()

    frame4 = Frame(f3_img, width=200, height=200)
    frame4.place(anchor='center', relx=0.7, rely=0.4)
    img4 = Image.open(img_list[3])
    resized_image4 = img4.resize((200, 200), Image.ANTIALIAS)
    new_image4 = ImageTk.PhotoImage(resized_image4)
    label4 = Label(frame4, image=new_image4)
    label4.pack()

    frame5 = Frame(f3_img, width=200, height=200)
    frame5.place(anchor='center', relx=0.9, rely=0.4)
    img5 = Image.open(img_list[4])
    resized_image5 = img5.resize((200, 200), Image.ANTIALIAS)
    new_image5 = ImageTk.PhotoImage(resized_image5)
    label5 = Label(frame5, image=new_image5)
    label5.pack()

    labelChoix = tk.Label(f3_img,
                          text=""" Veuillez choisir la ou les images correspondant le mieux\ 
 au suspect. Pour plus de détails, cliquez sur le bouton Aide.""",
                          font='Helvetica 16 bold', bg='white')
    labelChoix.pack()

    vb1 = IntVar()
    vb2 = IntVar()
    vb3 = IntVar()
    vb4 = IntVar()
    vb5 = IntVar()

    b1 = Checkbutton(f3_img, text="image 1", font='Helvetica 12',
                     variable=vb1, onvalue=1, offvalue=0,
                     command=recup_valCheckB)
    b2 = Checkbutton(f3_img, text="image 2", font='Helvetica 12',
                     variable=vb2, onvalue=1, offvalue=0,
                     command=recup_valCheckB)
    b3 = Checkbutton(f3_img, text="image 3", font='Helvetica 12',
                     variable=vb3, onvalue=1, offvalue=0,
                     command=recup_valCheckB)
    b4 = Checkbutton(f3_img, text="image 4", font='Helvetica 12',
                     variable=vb4, onvalue=1, offvalue=0,
                     command=recup_valCheckB)
    b5 = Checkbutton(f3_img, text="image 5", font='Helvetica 12',
                     variable=vb5, onvalue=1, offvalue=0,
                     command=recup_valCheckB)

    b1.place(anchor=tk.N, relheight=0.1, relwidth=0.1, relx=0.10, rely=0.08)
    b2.place(anchor=tk.N, relheight=0.1, relwidth=0.1, relx=0.30, rely=0.08)
    b3.place(anchor=tk.N, relheight=0.1, relwidth=0.1, relx=0.50, rely=0.08)
    b4.place(anchor=tk.N, relheight=0.1, relwidth=0.1, relx=0.70, rely=0.08)
    b5.place(anchor=tk.N, relheight=0.1, relwidth=0.1, relx=0.90, rely=0.08)

    boutVal = Button(f3_img, text="Valider", font='Arial 12', height=2,
                     width=20, borderwidth=4, bg='#BDECB6',
                     command=lambda: openf4(env_path))
    boutVal.place(anchor=tk.N, relheight=0.1, relwidth=0.1,
                  relx=0.4, rely=0.7)

    f3_img.mainloop()


def f4(env_path, path_final_img):
    """
    Creates the window 4 from the execution of openf4
    Parameters : 
    ------------
    env_path : <string> : the relative path to file Setup_dev 
    path_final_img :the relative path of the final image selected by the
    user
    """

    def export():
        """
        Event linked to the "Exporter" menu option
        Opens the window 5 where the user can input the values for
        the export
        """
        f5(env_path, img_f5)

    def quitter():
        """
        Event linked to the "Quitter" menu option : destruction of
        the current window
        """
        f4_final.destroy()

    def openf1(env_path):
        """
        Event linked to the "Nouveau" menu option : destroys of the
        current window and opens the window 1
        Parameters : 
        ------------
        env_path : <string> : the relative path to file Setup_dev 
        """
        f4_final.destroy()
        f1(env_path)

    # creation of the window 4
    f4_final = Tk()
    w, h = f4_final.winfo_screenwidth(), f4_final.winfo_screenheight()
    f4_final.geometry("%dx%d" % (w, h))
    f4_final.configure(bg='white')

    labelexpl = Label(f4_final, text="""Voici l'image finale. Vous pouvez utiliser\
 le menu en onglet pour l'exporter, recommencer une session\
 ou quitter l'application.""",
                      bg="white", font="Arial 14 italic")
    labelexpl.pack()

    menubar = Menu(f4_final)

    menu1 = Menu(menubar, tearoff=0)
    menu1.add_command(label="Exporter", command=export)
    menu1.add_command(label="Nouveau", command=lambda: openf1(env_path))
    menu1.add_separator()
    menu1.add_command(label="Quitter", command=quitter)
    menubar.add_cascade(label="Fichier", menu=menu1)

    frame_final = Frame(f4_final, width=400, height=400)
    frame_final.place(anchor='center', relx=0.5, rely=0.45)
    img_f5 = Image.open(path_final_img)
    resized_image = img_f5.resize((600, 600), Image.ANTIALIAS)
    image_finale = ImageTk.PhotoImage(img_f5)
    label_final = Label(frame_final, image=image_finale)
    label_final.pack()

    f4_final.config(menu=menubar)

    f4_final.mainloop()


def f5(env_path, img_f5):
    """
    Creates the window 5 from the execution of openf5

    Parameters : 
    ------------
    env_path : str
        The relative path to file Setup_dev
    img_f5 : PIL image 
        The final image chosen by the user
    """
    
    def aide():
        """
        Event linked to the 'Aide' button.
        Displays a message explaning which format to use for the path,
        and how to write a valid name for the image. 
        """
        format_path = utils.get_sub_sys()
        showinfo('Aide', """ Vous opérez sous %s .\
 Voici les formats de chemin approriés pour chaque système d'exploitation : 
                     
                     - Windows : C:\Repertoire\dossierfinal
                         
                     - Linux : /home/utilisateur/dossierfinal
                         
                     - MacOS : /Repertoire/dossierfinal       
                     
                Par défaut, le nom de l'image enregistrée est 'image_finale'
                et elle est enregistrée dans le dossier Result.
                 
                 """ % format_path)
        
        f5_xprt.destroy()
        
    def export():
        """
        Exports the final image selected by the user. The images is
        exported in the directory "Result" under the name "image_finale".
        
        If the user enters a specific path or name, it is taken into
        account for the export. If the path or the name are incorrect, a
        message of error pops up. The image can only be exported in
        ".jpg" format
        """
        chemin = T1.get("1.0", "end-1c")
        nom = T2.get("1.0", "end-1c")
        logging.info('Exporting image')

        if nom != '':
            jpg = '.jpg'
            nom_final = nom+jpg
        elif nom == '':
            nom_final = 'image_finale.jpg'

        if chemin == '':
            chemin = utils.get_path(env_path, 'Result')

        path_defaut = chemin
        img_save = img_f5

        # export if possible
        try:
            img_save.save(os.path.join(path_defaut, nom_final), "JPEG")
            showinfo('Info', """Image enregistrée. Vous pouvez recommencer ou\
 fermer le logiciel à partir de l'onglet menu de la\
 fenêtre précédente.""")
        except:
            showinfo('Attention', """Le chemin ou le nom de l'image ne sont pas\
 au bon format.""")

        f5_xprt.destroy()
        
        
    
    # creation of the 5th window
    
    f5_xprt = Tk()
    f5_xprt.geometry("500x300")
    f5_xprt.configure(bg='white')

    txt1 = Label(f5_xprt, text="Chemin de sauvegarde",
                 bg="white", font="Arial 14 italic")
    txt1.pack()
    T1 = Text(f5_xprt, height=2, width=52)
    T1.pack()

    txt2 = Label(f5_xprt, text="Nom de l'image",
                 bg="white", font="Arial 14 italic")
    txt2.pack()
    T2 = Text(f5_xprt, height=2, width=52)
    T2.pack()

    boutSend = Button(f5_xprt, text="OK", font='Arial 10', height=2,
                      width=20, borderwidth=4, bg='#BDECB6', command=export)
    boutSend.place(anchor=tk.N, relheight=0.3,
                   relwidth=0.2, relx=0.4, rely=0.7)
    
    boutAide = Button(f5_xprt, text="Aide", font='Arial 10', height=2,
                      width=20, borderwidth=4, bg='#D2B48C', command=aide)
    boutAide.place(anchor=tk.N, relheight=0.3,
                   relwidth=0.2, relx=0.6, rely=0.7)

    f5_xprt.mainloop()


if __name__ == '__main__':
    f1("../")
