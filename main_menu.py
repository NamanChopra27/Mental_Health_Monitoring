import tkinter as tk
from tkinter import messagebox
import subprocess
from PIL import Image, ImageTk
import requests
from io import BytesIO

# Function to fetch and load the image from a web URL
def load_image_from_web(url):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        image_data = BytesIO(response.content)
        return Image.open(image_data)
    else:
        raise Exception(f"Failed to load image from {url}. HTTP Status Code: {response.status_code}")

# Main function to create the application window
def create_main_menu():
    window = tk.Tk()
    window.title("Mental Health Monitoring System")

    # Get screen dimensions
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    # Set the window size to full screen
    window.geometry(f"{screen_width}x{screen_height}")

    # Load the high-quality background image
    try:
        url = "https://tse4.mm.bing.net/th?id=OIP.6CYDBjIp1G1G-eQZkdAYfgHaEc&pid=Api&P=0&h=180"  # Replace with a high-res image URL
        bg_image = load_image_from_web(url)
        bg_photo = ImageTk.PhotoImage(bg_image.resize((screen_width, screen_height), Image.Resampling.LANCZOS))
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image: {str(e)}")
        return

    # Set the background image
    label_bg = tk.Label(window, image=bg_photo)
    label_bg.place(x=0, y=0, relwidth=1, relheight=1)

    # Create a frame for buttons
    frame_buttons = tk.Frame(window, bg="black", bd=0)
    frame_buttons.place(relx=0.5, rely=0.5, anchor="center")

    # Add a heading label
    heading_label = tk.Label(
        frame_buttons,
        text="Mental Health Monitoring System",
        font=("Helvetica", 18, "bold"),
        fg="white",
        bg="black"
    )
    heading_label.pack(pady=40)

    heading_label = tk.Label(
        frame_buttons,
        text="Please Select A Model to Run",
        font=("Helvetica", 18, "bold"),
        fg="white",
        bg="black"
    )
    heading_label.pack(pady=20)

    # Button style
    button_style = {
        'width': 25,
        'height': 2,
        'font': ("Helvetica", 12),
        'bd': 0,
        'relief': "solid",
        'bg': "#4CAF50",
        'fg': "white",
        'activebackground': "#45a049",
        'activeforeground': "white"
    }

    # Add buttons to the frame
    tk.Button(frame_buttons, text="Depression Status Predictor", command=lambda: subprocess.run(["python", "model.py"]), **button_style).pack(pady=15)
    tk.Button(frame_buttons, text="Respiration Monitoring", command=lambda: subprocess.run(["python", "respiration.py"]), **button_style).pack(pady=15)
    tk.Button(frame_buttons, text="Facial Expression Detection", command=lambda: subprocess.run(["python", "facial_expression.py"]), **button_style).pack(pady=15)
    tk.Button(frame_buttons, text="Body Temperature Detection", command=lambda: subprocess.run(["python", "body_temperature.py"]), **button_style).pack(pady=15)

    # Footer label
    footer_label = tk.Label(window, text="© 2024 Mental Health Project | All rights reserved.", font=("Helvetica", 8), fg="#777", bg="black")
    footer_label.pack(side="bottom", pady=5)

    # Keep a reference to the image to avoid garbage collection
    label_bg.image = bg_photo

    window.mainloop()

# Run the application
if __name__ == "__main__":
    create_main_menu()
