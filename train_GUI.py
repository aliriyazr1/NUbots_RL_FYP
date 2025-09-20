import tkinter as tk
from tkinter import Label, Entry, Button, filedialog

# GUI because it was such a pain to CTRL + C/CTRL + V file paths in terminal
# Inspired by https://www.geeksforgeeks.org/python/file-explorer-in-python-using-tkinter/
class TrainGUI:
    def __init__(self):
        self.master = tk.Tk()
        self.master.title("Train GUI")

        self.label_file_explorer = Label(self.master, text="File Explorer using Tkinter", width=100, height=4, fg="blue")
        self.label_file_explorer.grid(column=1, row=1)

        self.browse_button = Button(self.master, text="Browse Files", command=self.open_file_dialog)
        self.browse_button.grid(column=1, row=2)

        self.button_exit = Button(self.master, text="Exit", command=exit)
        self.button_exit.grid(column=1, row=3)

        Label(self.master, text='File Name').grid(row=1)

    def open_file_dialog(self) -> str:
        filename = filedialog.askopenfilename(initialdir=".", title="Select model to retrain", filetypes=[("Zip files", "*.zip"), ("All files", "*.*")])
        # Change label contents
        self.label_file_explorer.configure(text="File Opened: " + filename)
        return filename
    
    # def display(self):
    #     self.master.mainloop()

    def close_window(self):
        self.master.destroy()

if __name__ == "__main__":
    app = TrainGUI()
    file_name = app.open_file_dialog()
    print(f"Selected file: {file_name}")
    # app.display()
    if file_name:
        app.close_window()
# # Create a File Explorer label
# label_file_explorer = Label(root_window, 
#                             text = "File Explorer using Tkinter",
#                             width = 100, height = 4, 
#                             fg = "blue")

# Label(root_window, text='Select the model you want to fine-tune/retrain:').grid(row=0)
# browse_button = Button(root_window, text="Browse Files", command = open_file_dialog)
# button_exit = Button(root_window, text = "Exit", command = exit)
# label_file_explorer.grid(column = 1, row = 1)
# browse_button.grid(column = 1, row = 2)
# button_exit.grid(column = 1,row = 3)

# Label(root_window, text='Last Name').grid(row=1)
# Entry(root_window).grid(row=0, column=1)
# Entry(root_window).grid(row=1, column=1)

# root_window.mainloop()