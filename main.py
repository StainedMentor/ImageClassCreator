import os
import shutil
import json
from tkinter import *
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

CONFIG_FILE = "label_config.json"

class ImageLabelTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Labeling Tool")

        self.source_folder = ""
        self.dest_folder = ""
        self.class_names = []

        self.image_files = []
        self.current_image_index = 0
        self.current_image_path = None
        self.tk_image = None
        self.resize_job = None

        self.setup_gui()
        self.load_config()

    def setup_gui(self):
        self.tab_control = ttk.Notebook(self.root)

        # Config Tab
        self.config_tab = Frame(self.tab_control)
        self.tab_control.add(self.config_tab, text='Config')

        Button(self.config_tab, text="Select Source Folder", command=self.select_source).pack(pady=5)
        Button(self.config_tab, text="Select Destination Folder", command=self.select_destination).pack(pady=5)

        Label(self.config_tab, text="Enter Class Names (comma-separated):").pack()
        self.class_entry = Entry(self.config_tab, width=50)
        self.class_entry.pack(pady=5)

        Button(self.config_tab, text="Start Labeling", command=self.start_labeling).pack(pady=10)

        # Labeling Tab
        self.label_tab = Frame(self.tab_control)
        self.tab_control.add(self.label_tab, text='Label')

        # Layout: Image top, buttons fixed bottom
        self.label_tab.rowconfigure(0, weight=1)
        self.label_tab.rowconfigure(1, weight=0)
        self.label_tab.rowconfigure(2, weight=0)
        self.label_tab.columnconfigure(0, weight=1)

        self.image_panel = Label(self.label_tab, anchor='center', bg='gray90')
        self.image_panel.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        self.image_panel.bind("<Configure>", self.on_resize)

        self.button_frame = Frame(self.label_tab)
        self.button_frame.grid(row=1, column=0, sticky='ew', pady=5)

        self.status_label = Label(self.label_tab, text="", fg="blue")
        self.status_label.grid(row=2, column=0, sticky='ew', pady=(0, 5))

        self.tab_control.pack(expand=1, fill='both')

    def select_source(self):
        self.source_folder = filedialog.askdirectory()
        print(f"Selected source folder: {self.source_folder}")

    def select_destination(self):
        self.dest_folder = filedialog.askdirectory()
        print(f"Selected destination folder: {self.dest_folder}")

    def start_labeling(self):
        class_text = self.class_entry.get().strip()
        self.class_names = [name.strip() for name in class_text.split(',') if name.strip()]

        if not (self.source_folder and self.dest_folder and self.class_names):
            self.status_label.config(text="Please complete config: source, destination, and classes.", fg="red")
            return

        self.save_config()

        for class_name in self.class_names:
            os.makedirs(os.path.join(self.dest_folder, class_name), exist_ok=True)

        self.image_files = [f for f in os.listdir(self.source_folder)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heic'))]
        self.image_files.sort()
        self.current_image_index = 0

        if not self.image_files:
            self.status_label.config(text="No images found in source folder.", fg="red")
            return

        self.build_class_buttons()
        self.load_next_image()
        self.tab_control.select(1)

    def build_class_buttons(self):
        for widget in self.button_frame.winfo_children():
            widget.destroy()

        button_style = {
            "width": 15,
            "height": 2,
            "padx": 5,
            "pady": 5,
            "bg": "#e0e0e0",
            "activebackground": "#c0c0c0",
            "font": ("Arial", 10, "bold"),
            "relief": RAISED,
            "bd": 2
        }

        max_columns = 4
        for i, class_name in enumerate(self.class_names):
            btn = Button(self.button_frame, text=class_name,
                         command=lambda c=class_name: self.label_image(c),
                         **button_style)
            row = i // max_columns
            col = i % max_columns
            btn.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

    def load_next_image(self):
        if self.current_image_index >= len(self.image_files):
            self.image_panel.config(image='', text='All images labeled!')
            self.status_label.config(text="Labeling complete.", fg="green")
            return

        filename = self.image_files[self.current_image_index]
        self.current_image_path = os.path.join(self.source_folder, filename)

        try:
            image = Image.open(self.current_image_path)
            self.original_image = image
            self.update_displayed_image()
            self.status_label.config(text=f"{filename} ({self.current_image_index + 1}/{len(self.image_files)})", fg="blue")
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
            self.current_image_index += 1
            self.load_next_image()

    def on_resize(self, event):
        if self.resize_job:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(200, self.update_displayed_image)

    def update_displayed_image(self):
        if not hasattr(self, 'original_image'):
            return

        panel_width = self.image_panel.winfo_width()
        panel_height = self.image_panel.winfo_height()

        if panel_width < 10 or panel_height < 10:
            panel_width, panel_height = 400, 400

        image = self.original_image.copy()
        image.thumbnail((panel_width, panel_height), Image.ANTIALIAS)
        self.tk_image = ImageTk.PhotoImage(image)
        self.image_panel.config(image=self.tk_image)

    def label_image(self, class_name):
        dest_path = os.path.join(self.dest_folder, class_name, os.path.basename(self.current_image_path))

        try:
            shutil.move(self.current_image_path, dest_path)
        except Exception as e:
            print(f"Error moving file: {e}")
            self.status_label.config(text="Error moving file.", fg="red")
            return

        self.current_image_index += 1
        self.load_next_image()

    def save_config(self):
        config = {
            "source_folder": self.source_folder,
            "dest_folder": self.dest_folder,
            "class_names": self.class_names
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                try:
                    config = json.load(f)
                    self.source_folder = config.get("source_folder", "")
                    self.dest_folder = config.get("dest_folder", "")
                    self.class_names = config.get("class_names", [])
                    self.class_entry.delete(0, END)
                    self.class_entry.insert(0, ", ".join(self.class_names))
                except Exception as e:
                    print(f"Error loading config: {e}")

if __name__ == "__main__":
    root = Tk()
    root.geometry("800x600")
    app = ImageLabelTool(root)
    root.mainloop()