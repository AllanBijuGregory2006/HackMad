import tkinter as tk

root = tk.Tk()
root.title("SoundGuard")
root.geometry("800x500")
root.configure(bg="#1a1a1a")

status_label = tk.Label(root, text="IDLE", font=("Helvetica", 24), bg="#1a1a1a", fg="white")
status_label.pack(pady=200)

root.mainloop()
