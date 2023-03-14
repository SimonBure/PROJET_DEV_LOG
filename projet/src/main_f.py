from tkinter import *
import tkinter as tk
import tkinter.font as font
from PIL import ImageTk, Image
from tkinter.messagebox import *
from tkinter import ttk
from PIL import ImageTk, Image
#from create_db import get_database_cursor, get_database_path, request_data_by_id


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
    
    framelogo = Frame(f1_acc, width=400, height=400)
    framelogo.pack()
    framelogo.place(anchor='center', relx=0.5, rely=0.45)
    logo = Image.open("idkit.png")
    logo_resized= logo.resize((400,400), Image.ANTIALIAS)
    new_logo= ImageTk.PhotoImage(logo_resized)
    label_imagee = Label(framelogo, image = new_logo)
    label_imagee.pack() 

    canvas = Canvas(f1_acc, width=400, height=100, bg='ivory')
    canvas.place(anchor='center', relx=0.5, rely=0.15)
    txtitre = canvas.create_text(200, 45, text="IdKit", font="Arial 50 italic", fill="green")
    txt = canvas.create_text(200, 85, text="Le logiciel de constitution de portaits-robots",font = "Arial 12 italic", fill="green")


    boutS=Button(f1_acc, text="Commencer", font='Arial 13', borderwidth = 4, bg = '#BDECB6', padx=5, pady=5, command = openf2)
    boutS.place(anchor=tk.W, relheight=0.1, relwidth=0.15, relx=0.3, rely = 0.8)

    boutH = Button(text='Aide', command=aide, font='Arial 13',borderwidth=4, bg = "#D2B48C")
    boutH.place(anchor=tk.E, relheight=0.1, relwidth=0.15, relx=0.7, rely= 0.8)


    
    
    f1_acc.mainloop()


################################################# FENETRE 2 #########################################################

def f2():

    """
    Création de la fenetre 2 depuis l'execution de openf2
    """

    f2_flr = Tk()
    w, h = f2_flr.winfo_screenwidth(), f2_flr.winfo_screenheight()
    f2_flr.geometry("%dx%d" % (w, h))
    

# PB  : ecraser la valeur pour ne prendre que la valeur finale cliquée par l'utilisateur (s'il change d'avis)
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
    
    def recup_RB_peau():
        """
        Evenement : recuperer la valeur saisie par l'utilisateur dans le widget Button peau
        Returns
        -------
        <int>
        """
        rbp = value.get()
        return(rbp)
    
    def recup_valCBX(menucombo):
        """
        Evenement : recuperer la valeur saisie par l'utilisateur dans le widget Combobox cheveux
        Returns
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
        hat = vhat.get()
        liste_acc = [lunet, mous, hat]
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
        p=recup_RB_peau()
        acc=recup_valCkB()
        stop = FALSE
        if (g==None or a==None or c=='Veuillez choisir un élément' or p==None or acc==[]):
            stop = TRUE
        return stop

    def openf3():
        """
        Evenement associé au bouton Envoyer: destruction de la fenetre courante et ouverture de la fenetre 3
        """
        
        test = verif_reponses()
        if test==FALSE:
            f2_flr.destroy()
            f3()
        elif(test==TRUE):
            showinfo('ATTENTION', 'Veuillez remplir tous les champs')



    boutSend=Button(f2_flr, text="Envoyer", font='Arial 12', height = 2, width = 20, borderwidth = 4, bg = '#BDECB6', command=openf3)
    boutSend.place(anchor=tk.N, relheight=0.07, relwidth=0.10, relx=0.5, rely= 0.7)

    labelT = Label(f2_flr, text="Ce formulaire vise à affiner la base de données pour vous présenter les solutions les plus pertinentes dans un temps minimal", bg="white", font = "Arial 14 italic")
    labelT.pack()

    labelSEXE = Label(f2_flr, text="Quel est le genre de l'individu ?", font='Helvetica 16 bold')
    labelSEXE.pack()

    vari = StringVar()
    bF = Radiobutton(f2_flr, text="Femme", font='Helvetica 12', variable=vari, value=1, command=recup_RB_genre)
    bH = Radiobutton(f2_flr, text="Homme", font='Helvetica 12', variable=vari, value=2, command=recup_RB_genre)
    bF.pack()
    bH.pack()

    
    labelAGE = Label(f2_flr, text="Quelle tranche d'âge ?", font='Helvetica 16 bold')
    labelAGE.pack()

    valeur = StringVar()
    bJ = Radiobutton(f2_flr, text="Jeune", font='Helvetica 12', variable=valeur, value=3, command=recup_RB_age)
    bA = Radiobutton(f2_flr, text="Âgé", font='Helvetica 12', variable=valeur, value=4, command=recup_RB_age)
    bJ.pack()
    bA.pack()

    
    labelPEAU = Label(f2_flr, text="Quelle couleur de peau ?", font='Helvetica 16 bold')
    labelPEAU.pack()

    value = StringVar()
    bP = Radiobutton(f2_flr, text="Pâle", font='Helvetica 12', variable=value, value=5, command=recup_RB_peau)
    bD = Radiobutton(f2_flr, text="Foncée", font='Helvetica 12', variable=value, value=6, command=recup_RB_peau)
    bP.pack()
    bD.pack()


    labelChoix = tk.Label(f2_flr, text = " Quelle couleur de cheveux ?", font='Helvetica 16 bold')
    labelChoix.pack()
    listtest=["Veuillez choisir un élément", "Noirs", "Blonds","Bruns","Gris"]
    menucombo = ttk.Combobox(f2_flr, values=listtest, font='Helvetica 12')
    menucombo.current(0)
    menucombo.pack()

    
    vlun = IntVar()
    vmous = IntVar()
    vhat = IntVar()
    labelChoix = tk.Label(f2_flr, text = " Veuillez cocher les accessoires particuliers:", font='Helvetica 16 bold')
    labelChoix.pack()
    boutLun = Checkbutton(f2_flr, text="Lunettes", font='Helvetica 12', variable=vlun, onvalue=1, offvalue=0, command = recup_valCkB)
    boutMoust = Checkbutton(f2_flr, text="Moustache", font='Helvetica 12', variable=vmous, onvalue=1, offvalue=0, command = recup_valCkB)
    boutHat = Checkbutton(f2_flr, text="Chapeau", font='Helvetica 12', variable=vhat, onvalue=1, offvalue=0, command = recup_valCkB)

    boutLun.pack()
    boutMoust.pack()
    boutHat.pack()
    

    f2_flr.mainloop()

################################################# FENETRE 3 #########################################################

def f3():
    """
    Création de la fenetre 3 depuis l'execution de openf3
    """
    f3_img = Tk()
    w, h = f3_img.winfo_screenwidth(), f3_img.winfo_screenheight()
    f3_img.geometry("%dx%d" % (w, h))
        
            
    def recup_valCheckB():
        """
        Evenement : recuperer la valeur saisie par l'utilisateur dans le widget CheckButton image
        Returns
        -------
        <list>
        """
        repb1 = vb1.get()
        repb2 = vb2.get()
        repb3 = vb3.get()
        repb4 = vb4.get()
        repb5 = vb5.get()
        rep_tot = [repb1, repb2, repb3, repb4, repb5]
        return rep_tot
    
#####################################################################################################################
# A VOIR : comment "refresh" la fenetre 3 quand on choisit 3 images ? pour l'instant on passe direct à la fenetre 4 #
#####################################################################################################################

    def verif_rep():
        """
        Verifier que seules 3 images ont été choisies par l'utilisateur
        Returns
        -------
        <boolean>
        """
        checkbut=recup_valCheckB()
        compte = 0
        for i in range(len(checkbut)):
            if (checkbut[i]==1):
                compte+=1           
        stop = FALSE
        if (compte==3):
            stop = TRUE 
        return stop
    
    def openf4():
        """
        Evenement associé au bouton Valider: destruction de la fenetre courante et ouverture de la fenetre 4
        """
        pass4 = verif_rep()
        if pass4==TRUE:
            f3_img.destroy()
            f4()
        elif(pass4==FALSE):
            showinfo('ATTENTION', 'Veuillez ne sélectionner que 3 images')
            
      
    
    frame = Frame(f3_img, width=200, height=200)
    frame.pack()
    frame.place(anchor='center', relx=0.10, rely=0.4)
    img = Image.open("idkit.png")
    resized_image= img.resize((200,200), Image.ANTIALIAS)
    new_image= ImageTk.PhotoImage(resized_image)
    label = Label(frame, image = new_image)
    label.pack()   

    frame2 = Frame(f3_img, width=200, height=200)
    frame2.pack()
    frame2.place(anchor='center', relx=0.3, rely=0.4)
    img2 = Image.open("idkit.png")
    resized_image2= img2.resize((200,200), Image.ANTIALIAS)
    new_image2= ImageTk.PhotoImage(resized_image2)
    label2 = Label(frame2, image = new_image2)
    label2.pack()   
    
    frame3 = Frame(f3_img, width=200, height=200)
    frame3.pack()
    frame3.place(anchor='center', relx=0.5, rely=0.4)
    img3 = Image.open("idkit.png")
    resized_image3= img3.resize((200,200), Image.ANTIALIAS)
    new_image3= ImageTk.PhotoImage(resized_image3)
    label3 = Label(frame3, image = new_image3)
    label3.pack()   
    
    frame4 = Frame(f3_img, width=200, height=200)
    frame4.pack()
    frame4.place(anchor='center', relx=0.7, rely=0.4)
    img4 = Image.open("idkit.png")
    resized_image4= img4.resize((200,200), Image.ANTIALIAS)
    new_image4= ImageTk.PhotoImage(resized_image4)
    label4 = Label(frame4, image = new_image4)
    label4.pack() 
    
    frame5 = Frame(f3_img, width=200, height=200)
    frame5.pack()
    frame5.place(anchor='center', relx=0.9, rely=0.4)
    img5 = Image.open("idkit.png")
    resized_image5= img5.resize((200,200), Image.ANTIALIAS)
    new_image5= ImageTk.PhotoImage(resized_image5)
    label5 = Label(frame5, image = new_image5)
    label5.pack() 

    labelChoix = tk.Label(f3_img, text = " Veuillez cocher les trois images les plus justes:", font='Helvetica 16 bold')
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

    boutVal=Button(f3_img, text="Valider", font='Arial 12', height = 2, width = 20, borderwidth = 4, bg = '#BDECB6', command = openf4)
    boutVal.place(anchor=tk.N, relheight=0.1, relwidth=0.1, relx=0.5, rely= 0.7)


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
