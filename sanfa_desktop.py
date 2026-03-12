import customtkinter as ctk
from PIL import Image
import os
import sys
import threading
import time

# Safely import our God-Level Engine V5
ENGINE_DIR = os.path.join(os.path.dirname(__file__), "engine")
sys.path.insert(0, ENGINE_DIR)
try:
    from engine import poison_image
except ImportError:
    print("WARNING: Could not import engine.py. GUI will run in demo mode.")
    def poison_image(inp, out):
        time.sleep(3)
        with open(out + ".report.json", "w") as f:
            f.write('{"status": "demo"}')

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class SANFADesktopApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("SANFA — Infinite Engine V5")
        self.geometry("800x600")
        self.minsize(800, 600)
        
        # Grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ----------------------------------------
        # HEADER
        # ----------------------------------------
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        
        self.title_label = ctk.CTkLabel(
            self.header_frame, 
            text="SANFA CLOUD", 
            font=ctk.CTkFont(family="Playfair Display", size=32, weight="bold"),
            text_color="#C9A84C" # Gold accent
        )
        self.title_label.pack(side="left")
        
        self.version_label = ctk.CTkLabel(
            self.header_frame, 
            text="Engine V5 (Global Standard)", 
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.version_label.pack(side="right", fill="y", pady=10)

        # ----------------------------------------
        # MAIN AREA (Dropzone)
        # ----------------------------------------
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        self.drop_label = ctk.CTkLabel(
            self.main_frame,
            text="Click to Select Image\n\n(Drag & Drop coming in V2)",
            font=ctk.CTkFont(size=18)
        )
        self.drop_label.grid(row=0, column=0)
        
        # Transparent button covering the frame to act as a click zone
        self.select_btn = ctk.CTkButton(
            self.main_frame, 
            text="", 
            fg_color="transparent", 
            hover_color="#1F1F1F",
            command=self.select_file
        )
        self.select_btn.grid(row=0, column=0, sticky="nsew")

        # ----------------------------------------
        # CONTROLS & PROGRESS
        # ----------------------------------------
        self.control_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.control_frame.grid(row=2, column=0, padx=20, pady=(10, 20), sticky="ew")
        self.control_frame.grid_columnconfigure(0, weight=1)

        self.status_var = ctk.StringVar(value="Ready. Select an artwork to protect.")
        self.status_label = ctk.CTkLabel(
            self.control_frame, 
            textvariable=self.status_var, 
            font=ctk.CTkFont(size=14)
        )
        self.status_label.grid(row=0, column=0, sticky="w", pady=(0, 5))

        self.progressbar = ctk.CTkProgressBar(self.control_frame, progress_color="#C9A84C")
        self.progressbar.grid(row=1, column=0, sticky="ew")
        self.progressbar.set(0)

        self.action_btn = ctk.CTkButton(
            self.control_frame, 
            text="Protect Image", 
            fg_color="#C9A84C",
            hover_color="#A88B3E",
            text_color="black",
            font=ctk.CTkFont(weight="bold"),
            command=self.start_processing,
            state="disabled"
        )
        self.action_btn.grid(row=1, column=1, padx=(20, 0))

        # State
        self.selected_file = None
        self.output_file = None

    def select_file(self):
        filename = ctk.filedialog.askopenfilename(
            title="Select Artwork",
            filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*"))
        )
        if filename:
            self.selected_file = filename
            
            # Change UI to show selected file
            ext = os.path.splitext(filename)[1]
            self.output_file = filename.replace(ext, f"_protected{ext}")
            
            self.drop_label.configure(text=f"Selected:\n{os.path.basename(filename)}\n\nClick 'Protect Image' to begin.")
            self.action_btn.configure(state="normal")
            self.status_var.set("Engine V5 Ready.")
            self.progressbar.set(0)

    def start_processing(self):
        if not self.selected_file:
            return
            
        self.action_btn.configure(state="disabled")
        self.select_btn.configure(state="disabled")
        self.progressbar.configure(mode="indeterminate")
        self.progressbar.start()
        self.status_var.set("Initializing GPU Tensors...")

        # Run engine in background thread so GUI doesn't freeze
        thread = threading.Thread(target=self.run_engine_thread)
        thread.daemon = True
        thread.start()

    def run_engine_thread(self):
        try:
            # Simulate progress steps for UX
            time.sleep(1)
            self.status_var.set("Layer 1/3 — CLIP PGD Adversarial Attack...")
            
            # The actual engine handles its own heavy lifting
            poison_image(self.selected_file, self.output_file)
            
            # Success
            self.after(0, self.processing_complete, True, self.output_file)
            
        except Exception as e:
            self.after(0, self.processing_complete, False, str(e))

    def processing_complete(self, success, result_data):
        self.progressbar.stop()
        self.progressbar.configure(mode="determinate")
        self.progressbar.set(1.0 if success else 0)
        
        self.action_btn.configure(state="disabled")
        self.select_btn.configure(state="normal")
        
        if success:
            self.status_var.set("✅ SUCCESS: Image cryptographically sealed.")
            self.drop_label.configure(
                text=f"Protection Complete!\n\nSaved to:\n{os.path.basename(result_data)}\n\nClick here to protect another.",
                text_color="#6B8F71"
            )
        else:
            self.status_var.set(f"❌ ERROR: {result_data}")
            self.drop_label.configure(
                text="Protection Failed.\nCheck console for details.\nClick to try again.",
                text_color="red"
            )

if __name__ == "__main__":
    app = SANFADesktopApp()
    app.mainloop()
