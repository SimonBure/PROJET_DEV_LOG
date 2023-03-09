from tkinter import *
import tkinter as tk
import tkinter.font as font
from PIL import ImageTk, Image
from tkinter.messagebox import *
from tkinter import ttk
#!pip install torchvision
from PIL import ImageTk, Image
from create_db import get_database_cursor, get_dataset_path, request_data_by_id


################################################# FENETRE 1 #########################################################
def f1():

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
        f2()


    frame = Frame(f1_acc, width=800, height=400)
    frame.pack()
    frame.place(anchor='center', relx=0.5, rely=0.37)

    canvas = Canvas(f1_acc, width=1800, height=100, bg='ivory')
    canvas.pack(side=TOP, padx=5, pady=5)
    txtitre = canvas.create_text(500, 60, text="IdKey", font="Arial 50 italic", fill="green")
    txt = canvas.create_text(750, 75, text="Le logiciel de constitution de portaits-robots",font = "Arial 12 italic", fill="green")


    boutS=Button(f1_acc, text="Commencer", font='Arial 13', borderwidth = 4, bg = '#BDECB6', padx=5, pady=5, command = openf2)
    boutS.place(anchor=tk.S, relheight=0.15, relwidth=0.15, relx=0.5, rely = 0.9)

    boutH = Button(text='Aide', command=aide, font='Arial 13',borderwidth=4, bg = "#D2B48C")
    boutH.place(anchor=tk.N, relheight=0.15, relwidth=0.15, relx=0.5, rely= 0.6)

    '''
    label = Label(f1_acc, text="Version 1 - 09.03.23", bg="white")
    label.pack()


    photo = ImageTk.PhotoImage(master = f1_acc,file="logo.png")

    canvas = Canvas(f1_acc, bg = 'yellow')
    canvas.create_image(0, 0, anchor=NW, image=photo)
    canvas.pack()
    '''

    f1_acc.mainloop()


################################################# FENETRE 2 #########################################################

def f2():

    """
    Création de la fenetre 2 depuis l'execution de openf2
    """

    f2_flr = Tk()
    w, h = f2_flr.winfo_screenwidth(), f2_flr.winfo_screenheight()
    f2_flr.geometry("%dx%d" % (w, h))


    def openf3():
        """
        Evenement associé au bouton Envoyer: destruction de la fenetre courante et ouverture de la fenetre 3
        """
        f2_flr.destroy()
        f3()

    # A TESTER : afficher dans un label
# A FAIRE POUR TOUS LES WIDGETS
    # récupération de la valeur du radiobutton
    def recup_genre():
        """
        Evenement : recuperer la valeur saisie par l'utilisateur dans le widget
        Returns
        -------
        <type>
        """
        genre = value.get()
        return(genre)

    # bouton ok
### A AJOUTER : message d'erreur si tout n'est pas rempli + "ne sait pas"
#if (...pas tout rempli : par exemple, si les valeurs n'ont pas été récupérées ==> mais alor vérifier le .get()! )
#   messagebox.showinfo("Titre : erreur", "bouh t'as pas tout rempli")
# donc il faut détecter, puis if pb, msg d'erreur et on renvoie à l'étape du remplissage
# https://www.delftstack.com/fr/tutorial/tkinter-tutorial/tkinter-message-box/
    boutSend=Button(f2_flr, text="Envoyer", font='Arial 12', height = 2, width = 20, borderwidth = 4, bg = '#BDECB6', command=openf3)
    boutSend.place(x=545, y = 500)

# label titre
#    labelT = Label(f1_acc, text="Ce formulaire vise à affiner la base de données pour vous présenter les solutions les plus pertinentes dans un temps minimal", bg="white", font = "Arial 15 bold")
#    labelT.pack()

    labelSEXE = Label(f2_flr, text="Quel est le genre de l'individu ?", font='Helvetica 16 bold')
    labelSEXE.pack()

    value = StringVar()
    bF = Radiobutton(f2_flr, text="Femme", font='Helvetica 12', variable=value, value=1)
    bH = Radiobutton(f2_flr, text="Homme", font='Helvetica 12', variable=value, value=2)
    bF.pack()
    bH.pack()

    labelAGE = Label(f2_flr, text="Quelle tranche d'âge ?", font='Helvetica 16 bold')
    labelAGE.pack()

    value = StringVar()
    bJ = Radiobutton(f2_flr, text="Jeune", font='Helvetica 12', variable=value, value=1)
    bA = Radiobutton(f2_flr, text="Âgé", font='Helvetica 12', variable=value, value=2)
    bJ.pack()
    bA.pack()

    labelPEAU = Label(f2_flr, text="Quelle couleur de peau ?", font='Helvetica 16 bold')
    labelPEAU.pack()

    value = StringVar()
    bP = Radiobutton(f2_flr, text="Pâle", font='Helvetica 12', variable=value, value=1)
    bD = Radiobutton(f2_flr, text="Foncée", font='Helvetica 12', variable=value, value=2)
    bP.pack()
    bD.pack()


    labelChoix = tk.Label(f2_flr, text = " Quelle couleur de cheveux ?", font='Helvetica 16 bold')
    labelChoix.pack()
    listtest=["Veuillez choisir un élément", "Noirs", "Blonds","Bruns","Gris"]
    menucombo = ttk.Combobox(f2_flr, values=listtest, font='Helvetica 12')
    menucombo.current(0)
    menucombo.pack()


    labelChoix = tk.Label(f2_flr, text = " Veuillez cocher les accessoires particuliers:", font='Helvetica 16 bold')
    labelChoix.pack()
    boutLun = Checkbutton(f2_flr, text="Lunettes", font='Helvetica 12')
    boutMoust = Checkbutton(f2_flr, text="Moustache", font='Helvetica 12')
    boutHat = Checkbutton(f2_flr, text="Chapeau", font='Helvetica 12')

    boutLun.pack()
    boutMoust.pack()
    boutHat.pack()

    f2_flr.mainloop()

################################################# FENETRE 3 #########################################################

def f3():
    """
    Création de la fenetre 2 depuis l'execution de openf3
    """
    f3_img = Tk()
    w, h = f3_img.winfo_screenwidth(), f3_img.winfo_screenheight()
    f3_img.geometry("%dx%d" % (w, h))

    def openf4():
        """
        Evenement associé au bouton Valider: destruction de la fenetre courante et ouverture de la fenetre 4
        """
        f3_img.destroy()
        f4()


    frame = Frame(f3_img, width=600, height=400)
    frame.pack()
    frame.place(anchor='center', relx=0.5, rely=0.5)


    chemin = request_data_by_id(3)

    #Create an object of tkinter ImageTk
    img = ImageTk.PhotoImage(master = f3_img, file = chemin)

    #Create a Label Widget to display the Image
    label = Label(frame, image = img)
    label.pack()


    labelChoix = tk.Label(f3_img, text = " Veuillez cocher les trois images les plus justes:", font='Helvetica 16 bold')
    labelChoix.pack()
    b1 = Checkbutton(f3_img, text="image 1", font='Helvetica 12')
    b2 = Checkbutton(f3_img, text="image 2", font='Helvetica 12')
    b3 = Checkbutton(f3_img, text="image 3", font='Helvetica 12')
    b4 = Checkbutton(f3_img, text="image 4", font='Helvetica 12')
    b5 = Checkbutton(f3_img, text="image 5", font='Helvetica 12')

    b1.pack()
    b2.pack()
    b3.pack()
    b4.pack()
    b5.pack()
# message d'erreur si coche plus que 3


    ### ATTENTION, MSG D ERREUR SI TOUT N EST PAS REMPLI !
    boutVal=Button(f3_img, text="Valider", command = openf4)
    boutVal.pack()


    f3_img.mainloop()

################################################# FENETRE 4 #########################################################

def f4():
    """
    Création de la fenetre 4 depuis l'execution de openf4
    """
    f4_xprt = Tk()
    w, h = f4_xprt.winfo_screenwidth(), f4_xprt.winfo_screenheight()
    f4_xprt.geometry("%dx%d" % (w, h))

    def export():
        """
        Evenement associé au menu Exporter: export de l'image en format <A DEFINIR>
        """
        showinfo("alerte", "Pas encore fonctionnel")

    def quit():
        """
        Evenement associé au menu Quitter: destruction de la fenetre courante
        """
        f4_xprt.destroy()

    def openf1():
        """
        Evenement associé au menu Nouveau: destruction de la fenetre courante et ouverture de la fenetre 1
        """
        f4_xprt.destroy()
        f1()



    menubar = Menu(f4_xprt)

    menu1 = Menu(menubar, tearoff=0)
    menu1.add_command(label="Exporter", command=export)
    menu1.add_command(label="Nouveau", command=openf1)
    menu1.add_separator()
    menu1.add_command(label="Quitter", command=quit)
    menubar.add_cascade(label="Fichier", menu=menu1)


    f4_xprt.config(menu=menubar)


    f4_xprt.mainloop()






if __name__ == '__main__':
    f1()
