import os
import tkinter as tk
from validation import Validation
from tkinter import *
from tkinter import filedialog


def selection_img_folder():
    root.img_folder = filedialog.askdirectory(initialdir = '/', title = 'Folder containing images')
    if not root.img_folder.endswith('/'):
        root.img_folder += '/'
    label2['text'] = root.img_folder
        
def selection_csv():
    root.csv = filedialog.askopenfilename(initialdir = '/',
                                        title = "Select csv",
                                        filetypes = (("csv files","*.csv"),("all files","*.*")))
    label1['text'] = root.csv
        
def selection_pdf():
    root.save_folder = filedialog.askdirectory(initialdir = "/", title = 'Folder to save report')
    if not root.save_folder.endswith('/'):
        root.save_folder += '/'
    label3['text'] = root.save_folder

def set_name(sv):
    root.save_name = sv.get()
    
def report():
    Validation(root.csv, root.img_folder).report(root.save_folder + root.save_name, tkinter = True)
    file_to_remove = os.listdir()
    file_to_remove = [i for i in file_to_remove if i.startswith('temp_')]
    for i in file_to_remove:
        os.remove(i)
    bouton4['text'] = 'Done !'
    
root = tk.Tk()
bouton1 = tk.Button(root, text='Select csv', command = selection_csv)
label1 = Label(root, text='..')
bouton2 = tk.Button(root, text='Select image folder', command = selection_img_folder)
label2 = Label(root, text='..')
bouton3 = tk.Button(root, text='Select report save folder', command = selection_pdf)
label3 = Label(root, text='..')
label4 = Label(root, text='Name of the file : ')
value = StringVar() 
value.set('')
value.trace('w', lambda name, index, mode, value=value: set_name(value))
entree = Entry(root, textvariable=value, width=30)
bouton4 = tk.Button(root, text='Create report', 
                        command = report)

bouton1.grid(column = 0, row = 0, sticky = 'ew', pady = 1)
label1.grid(column = 2, row = 0, columnspan = 5, pady = 1)
bouton2.grid(column = 0, row = 1, sticky = 'ew', pady = 1)
label2.grid(column = 2, row = 1, columnspan = 5, pady = 1)
bouton3.grid(column = 0, row = 2, sticky = 'ew', pady = 1)
label3.grid(column = 2, row = 2, columnspan = 5, pady = 1)
label4.grid(column = 0, row = 3, sticky = 'ew')
entree.grid(column = 2, row = 3, columnspan = 5, pady = 1)
bouton4.grid(column = 0, row = 4, columnspan = 7, pady = 6)
    
root.mainloop()