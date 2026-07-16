"""The 'Utilities' view: small helper tools kept from the original project."""

import tkinter as tk
from tkinter import messagebox, ttk

from utils.text_tools import get_working_directory, remove_dirty_chars, simulate_file_upload


class UtilitiesMixin:
    """Miscellaneous utility tools exposed through the GUI."""

    def show_utilities_interface(self):
        self.main_menu_frame.pack_forget()

        self.utilities_frame = ttk.Frame(self.root, padding="10")
        self.utilities_frame.pack(expand=True, fill="both")

        ttk.Button(self.utilities_frame, text="< Back to Main Menu", command=self.back_to_main_menu).pack(pady=5, anchor="nw")

        content = ttk.Frame(self.utilities_frame, padding="10")
        content.pack(pady=10, padx=10, expand=True, fill="both")

        ttk.Button(content, text="Print Working Directory", command=self.print_working_dir).pack(pady=5)
        self.working_dir_label = ttk.Label(content, text="")
        self.working_dir_label.pack(pady=2)

        ttk.Button(content, text="Simulate File Upload", command=self.handle_file_upload).pack(pady=5)

        ttk.Label(content, text="String 1:").pack(pady=2)
        self.string1_entry = ttk.Entry(content, width=50)
        self.string1_entry.insert(tk.END, "This is the first string.")
        self.string1_entry.pack(pady=2)

        ttk.Label(content, text="String 2 (chars to remove):").pack(pady=2)
        self.string2_entry = ttk.Entry(content, width=50)
        self.string2_entry.insert(tk.END, "aeiou")
        self.string2_entry.pack(pady=2)

        ttk.Button(content, text="Remove Chars", command=self.remove_chars_action).pack(pady=5)

        ttk.Label(content, text="Result:").pack(pady=2)
        self.remove_chars_result_label = ttk.Label(content, text="")
        self.remove_chars_result_label.pack(pady=2)

    def print_working_dir(self):
        directory = get_working_directory()
        self.working_dir_label.config(text=f"Current Working Directory: {directory}")
        messagebox.showinfo("Working Directory", f"Current Working Directory: {directory}")

    def handle_file_upload(self):
        messagebox.showinfo(
            "File Upload",
            "Simulating file upload...\nIn a real Tkinter app, this would open a file dialog to select a file.",
        )
        uploaded_files = simulate_file_upload()
        if uploaded_files:
            file_name = next(iter(uploaded_files))
            content = uploaded_files[file_name].decode("utf-8")
            messagebox.showinfo(
                "Simulated Upload Result",
                f"File '{file_name}' uploaded. Content preview:\n{content[:100]}...",
            )
        else:
            messagebox.showinfo("Simulated Upload Result", "No file simulated for upload.")

    def remove_chars_action(self):
        s1 = self.string1_entry.get()
        s2 = self.string2_entry.get()
        result = remove_dirty_chars(s1, s2)
        self.remove_chars_result_label.config(text=f"Removed: '{result}'")
        messagebox.showinfo("Remove Characters", f"Original: '{s1}'\nChars to remove: '{s2}'\nResult: '{result}'")
