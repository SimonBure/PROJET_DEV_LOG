from tkinter import *
import tkinter as tk
import tkinter.font as font
from PIL import ImageTk, Image
from tkinter.messagebox import *
from tkinter import ttk
import sys
import os
import utils
import database
import utils

################################################# FENETRE 1 #########################################################
def f1(env_path):

    """
    Création de la fenetre 1 depuis l'execution de main.py
    """

    f1_acc = Tk()

    w, h = f1_acc.winfo_screenwidth(), f1_acc.winfo_screenheight()
    f1_acc.geometry("%dx%d" % (w, h))
    f1_acc.configure(bg='white')


    def aide():
        """
        Evenement associé au bouton Help: affichage d'un panneau aide suite à un clic sur le boutton Aide
        """
        showinfo('Aide', 'Sprint 1 : interface graphique minimaliste capable de récupérer et afficher des images de la base de données')

    def openf2():
        """
        Evenement associé au boutton Start: destruction de la fenetre courante et ouverture de la fenetre 2
        """
        f1_acc.destroy()
        f2(env_path)

    logo_path = utils.get_path(env_path, "Interface")
    logo_path = os.path.join(logo_path, "idkit.png")

    framelogo = Frame(f1_acc, width=400, height=400)
    framelogo.pack()
    framelogo.place(anchor='center', relx=0.5, rely=0.45)
    logo = ImageTk.PhotoImage(Image.open(logo_path))
    label_imagee = Label(framelogo, image = logo)
    label_imagee.pack()

    canvas = Canvas(f1_acc, width=1800, height=100, bg='ivory')
    canvas.pack(side=TOP, padx=5, pady=5)
    txtitre = canvas.create_text(500, 60, text="IdKit", font="Arial 50 italic", fill="green")
    txt = canvas.create_text(750, 75, text="Le logiciel de constitution de portaits-robots",font = "Arial 12 italic", fill="green")


    boutS=Button(f1_acc, text="Commencer", font='Arial 13', borderwidth = 4, bg = '#BDECB6', padx=5, pady=5, command = openf2)
    boutS.place(anchor=tk.W, relheight=0.1, relwidth=0.15, relx=0.3, rely = 0.8)

    boutH = Button(text='Aide', command=aide, font='Arial 13',borderwidth=4, bg = "#D2B48C")
    boutH.place(anchor=tk.E, relheight=0.1, relwidth=0.15, relx=0.7, rely= 0.8)




    f1_acc.mainloop()


################################################# FENETRE 2 #########################################################

def f2(env_path):

    """
    Création de la fenetre 2 depuis l'execution de openf2
    """

    f2_flr = Tk()
    w, h = f2_flr.winfo_screenwidth(), f2_flr.winfo_screenheight()
    f2_flr.geometry("%dx%d" % (w, h))

    def recup_RB_genre():
        """
        Evenement : recuperer la valeur saisie par l'utilisateur dans le widget Button genre
        Returns
        -------
        <int>
        """
        rbg = vari.get()
        return(rbg)

    def recup_RB_age():
        """
        Evenement : recuperer la valeur saisie par l'utilisateur dans le widget Button age
        Returns
        -------
        <int>
        """
        rba = valeur.get( )
        return(rba)

    def recup_valCBX(menucombo):
        """
        Evenement : recuperer la valeur saisie par l'utilisateur dans le widget Combobox cheveux
        Parametrs :
        -------
        Le
        Returns :
        -------
        <string>
        """
        cb = menucombo.get()
        return cb

    def recup_valCkB():
        """
        Evenement : recuperer la valeur saisie par l'utilisateur dans le widget CheckButton accessoires
        Returns
        -------
        <list>
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
        Verifier que tous les champs ont été remplis par l'utilisateur
        Returns
        -------
        <boolean>
        """
        g=recup_RB_genre()
        a=recup_RB_age()
        c=recup_valCBX(menucombo)
        acc=recup_valCkB()
        stop = FALSE
        if (g==None or a==None or c=='Veuillez choisir un élément' or acc==[]):
            stop = TRUE

        testliste=liste_db()
        return stop

    def liste_db():
        """
        Renvoie un array 2D correspondant à la sélection des paramètres par l'utilisateur
        Returns
        -------
        <2D array>
        """
        g=recup_RB_genre()
        a=recup_RB_age()
        c=recup_valCBX(menucombo)
        acc=recup_valCkB()

        liste_acc = [['Genre', 'Age', "Cheveux", "Lunettes", "Moustache", "Barbe"],[0,0,0,0,0,0]]

        if (g=='1'): # 0 : nsp, 1 : femme, 2 : homme
            liste_acc[1][0]=1
        elif (g=='2'):
            liste_acc[1][0]=2

        if (a=='3'): # 0 : nsp, 1 : jeune, 2 : vieux
            liste_acc[1][1]=1
        elif (a=='4'):
            liste_acc[1][1]=2

        list_cheveux=["Noirs", "Blonds","Bruns","Gris", "Chauve", "Autre"]
        # 0 : ne sait pas, noirs : 1, blonds : 2, bruns : 3, gris : 4, chauve : 5, Autre : 6
        for i in range(len(list_cheveux)):
            if (list_cheveux[i] == c):
                liste_acc[1][2]=i+1

        # accessoires : dans l'ordre des cases 3 à 6, 0 : ne sait pas ou non présent, 1 : présent
        for i in range(3,5):
            if (acc[i-3] == 1): # Si acc est choisi on le met dans la requête
                liste_acc[1][i]=1
            else :
                if acc[4] == 1 : # Si pas choisi + pas sur on ne sait pas donc 0
                    liste_acc[1][i]=0
                else : # Si pas choisi + on est sur, donc 2 : pas l'attribut
                    liste_acc[1][i]=2
        if acc[3] == 1 : # Si Aucun alors peut importe les précédants choix on met tout à 2
            liste_acc[1][3]=2
            liste_acc[1][4]=2
            liste_acc[1][5]=2

        return liste_acc


    def openf3(env_path):
        """
        Evenement associé au bouton Envoyer: destruction de la fenetre courante et ouverture de la fenetre 3
        """

        test = verif_reponses()
        ans_user = liste_db()
            # Retrieve 5 images
        array_metadata = database.create_querry_array(ans_user[1][0], ans_user[1][1], ans_user[1][2], ans_user[1][3], ans_user[1][4], ans_user[1][5])
        img_list = database.get_5_img(env_path, array_metadata) # liste de path

        if test==FALSE:
            f2_flr.destroy()
            f3(env_path, img_list)
        elif(test==TRUE):
            showinfo('ATTENTION', 'Veuillez remplir tous les champs')


    boutSend=Button(f2_flr, text="Envoyer", font='Arial 12', height = 2, width = 20, borderwidth = 4, bg = '#BDECB6', command= lambda : openf3(env_path))
    boutSend.place(anchor=tk.N, relheight=0.07, relwidth=0.10, relx=0.5, rely= 0.7)

    labelT = Label(f2_flr, text="Ce formulaire vise à affiner la base de données pour vous présenter les solutions les plus pertinentes dans un temps minimal", bg="white", font = "Arial 14 italic")
    labelT.pack()

    labelSEXE = Label(f2_flr, text="Quel est le genre de l'individu ?", font='Helvetica 12 bold')
    labelSEXE.pack()

    vari = StringVar()
    bF = Radiobutton(f2_flr, text="Femme", font='Helvetica 12', variable=vari, value=1, command=recup_RB_genre)
    bH = Radiobutton(f2_flr, text="Homme", font='Helvetica 12', variable=vari, value=2, command=recup_RB_genre)
    bg_nsp = Radiobutton(f2_flr, text="Ne sais pas", font='Helvetica 12', variable=vari, value=0, command=recup_RB_genre)# à laisser pour qu'on puisse être sûrs qu'on a tout rempli
    bF.pack()
    bH.pack()
    bg_nsp.pack()


    labelAGE = Label(f2_flr, text="Quelle tranche d'âge ?", font='Helvetica 12 bold')
    labelAGE.pack()

    valeur = StringVar()
    bJ = Radiobutton(f2_flr, text="Jeune", font='Helvetica 12', variable=valeur, value=3, command=recup_RB_age)
    bA = Radiobutton(f2_flr, text="Âgé", font='Helvetica 12', variable=valeur, value=4, command=recup_RB_age)
    bA_nsp = Radiobutton(f2_flr, text="Ne sais pas", font='Helvetica 12', variable=valeur, value=0, command=recup_RB_age)# à laisser pour qu'on puisse être sûrs qu'on a tout rempli
    bJ.pack()
    bA.pack()
    bA_nsp.pack()

    labelChoix = tk.Label(f2_flr, text = " Quelle couleur de cheveux ?", font='Helvetica 12 bold')
    labelChoix.pack()
    listtest=["Veuillez choisir un élément", "Noirs", "Blonds","Bruns","Gris", "Chauve", "Autre", "Ne sais pas"]
    menucombo = ttk.Combobox(f2_flr, values=listtest, font='Helvetica 12')
    menucombo.current(0)
    menucombo.pack()


    vlun = IntVar()
    vmous = IntVar()
    vbrd = IntVar()
    vauc = IntVar()
    vnsp = IntVar()
    labelChoix = tk.Label(f2_flr, text = " Veuillez cocher les accessoires particuliers:", font='Helvetica 12 bold')
    labelChoix.pack()
    boutLun = Checkbutton(f2_flr, text="Lunettes", font='Helvetica 12', variable=vlun, onvalue=1, offvalue=0, command = recup_valCkB)
    boutMoust = Checkbutton(f2_flr, text="Moustache", font='Helvetica 12', variable=vmous, onvalue=1, offvalue=0, command = recup_valCkB)
    boutbrd = Checkbutton(f2_flr, text="Barbe", font='Helvetica 12', variable=vbrd, onvalue=1, offvalue=0, command = recup_valCkB)
    boutauc = Checkbutton(f2_flr, text="Aucun", font='Helvetica 12', variable=vauc, onvalue=1, offvalue=0, command = recup_valCkB)
    boutnsp = Checkbutton(f2_flr, text="Ne sais pas", font='Helvetica 12', variable=vnsp, onvalue=1, offvalue=0, command = recup_valCkB)

    boutLun.pack()
    boutMoust.pack()
    boutbrd.pack()
    boutauc.pack()
    boutnsp.pack()


    f2_flr.mainloop()

################################################# FENETRE 3 #########################################################

def f3(env_path, img_list):
    """
    Création de la fenetre 3 depuis l'execution de openf3
    
    parameters : 
    ------------
    env_path : 
    img_list : une liste contenant les chemins de chaque image dans l'environnement
    """
    f3_img = Tk()
    w, h = f3_img.winfo_screenwidth(), f3_img.winfo_screenheight()
    f3_img.geometry("%dx%d" % (w, h))


    def recup_valCheckB():
        """
        Evenement : recuperer la valeur saisie par l'utilisateur dans le widget CheckButton image
        Returns
        -------
        <list> : ensemble des valeurs des checkbuttons après que l'utilisateur ait fait son choix
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
        Verifier si 3 images ont étées choisies par l'utilisateur

        Returns
        -------
        stop : <boolean> : TRUE si l'utilisateur a bien choisi 3 images
        index_choix : <list of int> : liste contenant les index des images choisies par l'utilisateur
        """
        checkbut=recup_valCheckB()
        compte = 0
        index_choix = []
        for i in range(len(checkbut)-1):
            if (checkbut[i]==1):
                compte+=1
                index_choix.append(i)
        stop = FALSE
        if (compte==3):
            stop = TRUE
        return stop, index_choix

    def verif_rep_1():
        """
        Verifier qu'une seule image a étée choisie par l'utilisateur, et laquelle.

        Returns
        -------
        stop : <boolean> : TRUE si l'utilisateur a bien choisi 1 seule image
        final : <boolean> : TRUE si l'utilisateur a bien coché "final"
        index : <int> : correspond à l'index de l'image choisie par l'utilisateur
        """
        checkbfinal=recup_valCheckB()
        index = 0
        compte = 0
        for i in range(len(checkbfinal)-1):
            if (checkbfinal[i]==1):
                compte+=1
                index = i
        stop = FALSE
        finale = FALSE
        for i in range(len(checkbfinal)-1):
            if (compte==1):
                stop = TRUE
            if (checkbfinal[5]==1):
                finale = TRUE
        return stop, finale, index


    def chemin_choix(index_choix):
        """
        Récuperer les chemins des images sélectionnées par l'utilisateur pour les envoyer à l'algorithme génétique
        Returns
        -------
        liste_chemin : <list> : liste contenant les chemins des 3 images sélectionnées
        """
        liste_chemin = []
        for i in range(img_list):
            if ((i == index_choix[0]) or (i == index_choix[1]) or (i == index_choix[2])):
                liste_chemin.append(img_list[i])
        return liste_chemin


    def openf3(env_path, img_list):
        """
        Refresh la fenetre courante
        """
        f3(env_path, img_list)

    def openf4(env_path):
        """
        Evenement associé au bouton Valider: selon les résultats des fonctions verif_rep_1 et
        verif_rep_3, destruction de la fenetre courante et ouverture de la fenetre 4 ou actualisation de la fenetre 3
        """
        refresh_3 = verif_rep_3()
        pass_4 = verif_rep_1()
        i_fin=pass_4[2]

    

        ####### SI ON REFRESH LA PAGE ########
        if (refresh_3[0]==TRUE and pass_4[1]==FALSE):
           
            # - boucle de temps : à voir combien de temps prend l'algo gen, mais AC rapide
            # - si boucle de temps ne marche pas, faire un break de temps de 30 sec
            # - sur cette boucle de temps, appeler à un intervalle dt une fonction de l'AC qui renvoie un
            # booléen quand les nouvelles images ont bien été encodées
            # - une fois que ce booléen (le résultat de la fonction) est true, aller chercher les nouvelles img
            # - écraser les anciennes images dans img_list
            # - refresh la fenetre : appeller openf3 ?

            ## code pour refresh : récupérer le path d'une image de l'AC, modifier img_list avec puis tout réafficher
            #directory_test = utils.get_path(env_path, "Encoder")
            #path1 = os.path.join(directory_test, "base_im.png")
            ## le seul souci : il faudrait que les 5 images de l'autoencodeur aient toujours le meme nom --> ok, se mettre d'accord du nom avec Jesus
            # img_refresh = path1
            # img_list[0]=path1
           
           #### récupérer les 5 nouvelles images de l'autoencodeur
           # POUR QUE CA FONCTIONNE, il faut img0.jpg à img4.jpg dans env/Auto_encoder/gen_img
           chemin_dossier = utils.get_path(env_path, 'gen_img')
            
           # joindre les path et les noms des images, et écraser les anciens path d'image_list par les nouveaux
           # attention les images sont numérotées de 0 à 4
           for i in range(len(img_list)):
               nom_img_ac = "img"+str(i)+".jpg"
               img_list[i]=os.path.join(chemin_dossier, nom_img_ac)
               
           f3_img.destroy()
           openf3(env_path, img_list)
            
        
        ####### SI PASSE A LA P.4 ########
        elif (pass_4[0]==TRUE and pass_4[1]==TRUE):
            path_final_img = img_list[i_fin]
            f3_img.destroy()
            f4(env_path, path_final_img)
            
        ####### S'IL Y A UN PB DANS LES CHOIX ########    
        elif((pass_4[0]==FALSE and refresh_3[0]==FALSE)or(pass_4[0]==TRUE and pass_4[1]==FALSE)or(pass_4[0]==FALSE and pass_4[1]==TRUE)or(refresh_3[0]==TRUE and pass_4[1]==TRUE)):
             showinfo('ATTENTION', '''Veuillez ne sélectionner qu'une image et cocher la case "Finale" ou sélectionner 3 images''')

    def aidef3():
        """
        Evenement associé au bouton Help: affichage d'un panneau aide suite à un clic sur le boutton Aide
        """
        showinfo('Aide', """Veuillez choisir les images correspondant le mieux au suspect, parmi les images proposées. Choisissez les 3 images les plus exactes jusqu'à ce que l'une d'elles vous satisfasse. Pour sélectionner l'image finale, veuillez cocher la case "Finale" """)


    boutHelp = Button(text='Aide', command=aidef3, font='Arial 14',borderwidth=4, bg = "#D2B48C")
    boutHelp.place(anchor=tk.E, relheight=0.1, relwidth=0.1, relx=0.6, rely= 0.75)

    vfinal = IntVar()
    checkbfinal = Checkbutton(f3_img, text="Finale", font='Helvetica 10', variable=vfinal, onvalue=1, offvalue=0)
    checkbfinal.place(anchor=tk.E, relheight=0.1, relwidth=0.1, relx=0.6, rely= 0.65)

    """
    # Retrieve an image created by the encoder
    directory_test = utils.get_path(env_path, "Encoder")
    print(directory_test)
    path1 = os.path.join(directory_test, "base_im.png")
    print(path1)
    path2 = os.path.join(directory_test, "recon_im.png")
    """


    # Créer une fenêtre d'erreur :
    if img_list == 0 :
        showinfo('ATTENTION', '''Il n'existe pas assez d'images correspondantes à cette sélection dans la base de données. Veuillez élargir vos critères.''')
        f3_img.destroy()
        f2(env_path)
        # Fenêtre d'erreur
        a = 0 # sert à rien mais pour ne pas avoir de problèmes d'indentation


    frame = Frame(f3_img, width=200, height=200)
    frame.place(anchor='center', relx=0.10, rely=0.4)
    img = Image.open(img_list[0])
    resized_image= img.resize((200,200), Image.ANTIALIAS)
    new_image= ImageTk.PhotoImage(resized_image)
    label = Label(frame, image = new_image)
    label.pack()


    frame2 = Frame(f3_img, width=200, height=200)
    frame2.place(anchor='center', relx=0.3, rely=0.4)
    img2 = Image.open(img_list[1])
    resized_image2= img2.resize((200,200), Image.ANTIALIAS)
    new_image2= ImageTk.PhotoImage(resized_image2)
    label2 = Label(frame2, image = new_image2)
    label2.pack()

    frame3 = Frame(f3_img, width=200, height=200)
    frame3.place(anchor='center', relx=0.5, rely=0.4)
    img3 = Image.open(img_list[2])
    resized_image3= img3.resize((200,200), Image.ANTIALIAS)
    new_image3= ImageTk.PhotoImage(resized_image3)
    label3 = Label(frame3, image = new_image3)
    label3.pack()

    frame4 = Frame(f3_img, width=200, height=200)
    frame4.place(anchor='center', relx=0.7, rely=0.4)
    img4 = Image.open(img_list[3])
    resized_image4= img4.resize((200,200), Image.ANTIALIAS)
    new_image4= ImageTk.PhotoImage(resized_image4)
    label4 = Label(frame4, image = new_image4)
    label4.pack()

    frame5 = Frame(f3_img, width=200, height=200)
    frame5.place(anchor='center', relx=0.9, rely=0.4)
    img5 = Image.open(img_list[4])
    resized_image5= img5.resize((200,200), Image.ANTIALIAS)
    new_image5= ImageTk.PhotoImage(resized_image5)
    label5 = Label(frame5, image = new_image5)
    label5.pack()

    labelChoix = tk.Label(f3_img, text = " Veuillez cocher l'image la plus juste:", font='Helvetica 16 bold')
    labelChoix.pack()

    vb1 = IntVar()
    vb2 = IntVar()
    vb3 = IntVar()
    vb4 = IntVar()
    vb5 = IntVar()

    b1 = Checkbutton(f3_img, text="image 1", font='Helvetica 12', variable=vb1, onvalue=1, offvalue=0, command = recup_valCheckB)
    b2 = Checkbutton(f3_img, text="image 2", font='Helvetica 12', variable=vb2, onvalue=1, offvalue=0, command = recup_valCheckB)
    b3 = Checkbutton(f3_img, text="image 3", font='Helvetica 12', variable=vb3, onvalue=1, offvalue=0, command = recup_valCheckB)
    b4 = Checkbutton(f3_img, text="image 4", font='Helvetica 12', variable=vb4, onvalue=1, offvalue=0, command = recup_valCheckB)
    b5 = Checkbutton(f3_img, text="image 5", font='Helvetica 12', variable=vb5, onvalue=1, offvalue=0, command = recup_valCheckB)

    b1.place(anchor=tk.N, relheight=0.1, relwidth=0.1, relx=0.10, rely= 0.05)
    b2.place(anchor=tk.N, relheight=0.1, relwidth=0.1, relx=0.30, rely= 0.05)
    b3.place(anchor=tk.N, relheight=0.1, relwidth=0.1, relx=0.50, rely= 0.05)
    b4.place(anchor=tk.N, relheight=0.1, relwidth=0.1, relx=0.70, rely= 0.05)
    b5.place(anchor=tk.N, relheight=0.1, relwidth=0.1, relx=0.90, rely= 0.05)

    boutVal=Button(f3_img, text="Valider", font='Arial 12', height = 2, width = 20, borderwidth = 4, bg = '#BDECB6', command = lambda : openf4(env_path))
    boutVal.place(anchor=tk.N, relheight=0.1, relwidth=0.1, relx=0.4, rely= 0.7)


    f3_img.mainloop()

################################################# FENETRE 4 #########################################################

def f4(env_path, path_final_img):
    """
    Création de la fenetre 4 depuis l'execution de openf4
    """
    f4_final = Tk()
    w, h = f4_final.winfo_screenwidth(), f4_final.winfo_screenheight()
    f4_final.geometry("%dx%d" % (w, h))

    def export():
        """
        DOCSTRING A MAJ
        Evenement associé au menu Exporter: export de l'image en format JGP
        L'export se fait par défaut dans l'environnement, ou l'utilisateur entre un chemin d'accès
        """
        f5(env_path, img_f5)

    def quit():
        """
        Evenement associé au menu Quitter: destruction de la fenetre courante
        """
        f4_final.destroy()

    def openf1(env_path):
        """
        Evenement associé au menu Nouveau: destruction de la fenetre courante et ouverture de la fenetre 1
        """
        f4_final.destroy()
        f1(env_path)


    labelexpl = Label(f4_final, text="Voici l'image finale. Vous pouvez utiliser le menu en onglet pour l'exporter, recommencer une session ou quitter l'application.", bg="white", font = "Arial 14 italic")
    labelexpl.pack()


    menubar = Menu(f4_final)

    menu1 = Menu(menubar, tearoff=0)
    menu1.add_command(label="Exporter", command=export)
    menu1.add_command(label="Nouveau", command= lambda : openf1(env_path))
    menu1.add_separator()
    menu1.add_command(label="Quitter", command=quit)
    menubar.add_cascade(label="Fichier", menu=menu1)


    frame_final = Frame(f4_final, width=400, height=400)
    frame_final.place(anchor='center', relx=0.5, rely=0.45)
    img_f5 = Image.open(path_final_img)
    image_finale = ImageTk.PhotoImage(img_f5)
    print(type(image_finale))
    label_final = Label(frame_final, image = image_finale)
    label_final.pack()


    f4_final.config(menu=menubar)


    f4_final.mainloop()

################################################# FENETRE 5 #########################################################

def f5(env_path, img_f5):

    f5_xprt = Tk()

    def export():
       '''
       DOCSTRING A FAIRE
       '''
       chemin = T1.get("1.0","end-1c")
       nom    = T2.get("1.0","end-1c")

       if (nom != ''):
           jpg = '.jpg'
           nom_final = nom+jpg
       elif (nom == ''):
           nom_final = 'image_finale.jpg'

       if (chemin == ''):
           chemin = utils.get_path(env_path, 'Result')

       path_defaut = chemin
       img_save = img_f5
       img_save.save(os.path.join(path_defaut, nom_final), "JPEG" )

       showinfo('Info', """Image enregistrée. Vous pouvez recommencer ou fermer le logiciel à partir de l'onglet menu de la fenêtre précédente.""")

    ## VOIR QUE FAIRE QUAND LE CHEMIN N EST PAS BON (afficher une erreur), ou nom existe déjà (AFFICHER UNE ERREUR)

    def quitter():
        """
        Evenement associé au menu Quitter: destruction de la fenetre courante
        """
        f5_xprt.destroy()


    txt1 = Label(f5_xprt, text="chemin de sauvegarde", bg="white", font = "Arial 14 italic")
    txt1.pack()
    T1 = Text(f5_xprt, height = 3, width = 52)
    T1.pack()

    txt2 = Label(f5_xprt, text="Nom de l'image", bg="white", font = "Arial 14 italic")
    txt2.pack()
    T2 = Text(f5_xprt, height = 3, width = 52)
    T2.pack()

    boutSend=Button(f5_xprt, text="OK", font='Arial 8', height = 2, width = 20, borderwidth = 4, bg = '#BDECB6', command= export)
    boutSend.place(anchor=tk.N, relheight=0.1, relwidth=0.1, relx=0.4, rely= 0.8)

    boutSend=Button(f5_xprt, text="Fermer", font='Arial 8', height = 2, width = 20, borderwidth = 4, bg = '#BDECB6', command= quitter)
    boutSend.place(anchor=tk.N, relheight=0.1, relwidth=0.1, relx=0.6, rely= 0.8)

    f5_xprt.mainloop()



if __name__ == '__main__':
    f1("../")
