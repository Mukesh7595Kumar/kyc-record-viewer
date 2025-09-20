import os
import re
import shutil
import time
import threading
import json
import mimetypes
from pathlib import Path

# --- LIBRARY IMPORT HANDLING ---
# Attempt to import all necessary libraries, with error handling for missing ones.
try:
    # Libraries from Script 1
    import cv2
    import numpy as np
    import pyautogui
    import pyperclip
    import keyboard
    from natsort import natsorted

    # Libraries from Script 2
    import google.generativeai as genai

except ImportError as e:
    print(f"Error: A required library is missing: {e.name}")
    print("Please install all required libraries by running:")
    print("pip install opencv-python numpy pyautogui pyperclip keyboard natsort google-generativeai")
    exit()

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# --- UTILITY FUNCTIONS & CONSTANTS (from both scripts) ---

# Gemini Tool Configuration
CONFIG_FILE = Path("gemini_config.json")
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

# Gemini Tool Prompt
FORMATTING_PROMPT = (
    """ Please convert the following raw text data into a semicolon-separated format with exactly 20 fields per record. Each field must be separated by a semicolon (;), with no missing semicolons, even if a field is empty. Field Order (20 Fields Total): 1. Record ID — e.g., KuT_HnT-NuV-8536 2. Company Name — include any suffixes like (SBS), (NBS), etc. 3. Street Address — e.g., 50 banding place 4. Primary Contact Name — full name, including any titles 5. Secondary Contact Name — if present 6. Date of Birth — preserve original formatting (e.g., Sunday, 05-27-1956) 7. Gender — preserve punctuation like periods (e.g., FEMALE.) 8. Country or Nationality — e.g., American, United States 9. Additional Address Info — neighborhood, building name, etc. 10. City — e.g., mill valley 11. State or Province — abbreviated (e.g., CA) 12. ZIP or Postal Code — e.g., 94941 13. Mobile Number — always comes before the email address 14. Email Address(es) — keep original casing 15. Contact 1 — first phone number after email 16. Contact 2 — second phone number after email 17. Verification Status — e.g., VERIFIED, N.A 18. Confirmation — e.g., YES, NA 19. Notes or Keywords — e.g., whitewall, verified by agent 20. Final Code or Token — any remaining hash-like or alphanumeric code Formatting Rules: - If a field is missing, leave it blank but preserve the semicolon (e.g., ;;). - Do not split words across fields. - Preserve internal punctuation, spaces, and original capitalization in all fields. - If multiple phone numbers or emails appear in one field, separate them with a single space. - Ensure the final output contains exactly 19 semicolons, resulting in 20 fields. For EACH IMAGE you receive in this request, output EXACTLY ONE line with 20 semicolon-separated fields (19 semicolons), in the SAME ORDER as the images are provided. Do not include any headers, labels, bullets, extra commentary, or blank lines. Do not merge multiple images into one line. The number of output lines MUST equal the number of input images in this request. Example Output: KuT_HnT-NuV-8536;BANK OF HAWAII (SBS);50 banding place;Pascal Reide;Ryann Pickervance;Sunday, 05-27-1956;FEMALE.;American;35666 Kenwood Center;mill valley;CA;94941;86 # 898 / 218-0575;Ihixson3v@posterous.com;33 # 423 / 959-6306;880 (442) 821-5156;VERIFIED;YES;whitewall;1WZ8rLwo4mKVqcJEtztcZrB4HAZDIRse2 """
).strip()


def natural_sort_key(s):
    """A key for natural sorting of strings (e.g., 'image10.png' after 'image2.png')."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def load_config():
    """Loads Gemini tool configuration from a JSON file."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}  # Return empty if file is corrupt
    return {}


def save_config(api_key, folder_path, batch_size):
    """Saves Gemini tool configuration to a JSON file."""
    cfg = {"api_key": api_key, "folder": folder_path, "batch_size": batch_size}
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def list_images_sorted(folder: Path):
    """Lists and naturally sorts image files in a directory."""
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    files.sort(key=lambda p: natural_sort_key(p.name))
    return files


def to_inline_part(path: Path):
    """Converts an image file path to a Gemini API inline data part."""
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        ext = path.suffix.lower()
        mime_map = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
                    '.tif': 'image/tiff', '.tiff': 'image/tiff', '.bmp': 'image/bmp', '.webp': 'image/webp'}
        mime = mime_map.get(ext, 'application/octet-stream')
    return {"mime_type": mime, "data": path.read_bytes()}


# --- GEMINI PROCESSING LOGIC ---
class GeminiBatchProcessor:
    """Handles the batch processing of images using the Gemini API."""

    def __init__(self, api_key: str, folder: Path, batch_size: int, ui_callback=None):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-pro")
        self.folder = folder
        self.batch_size = max(1, int(batch_size))
        self.ui = ui_callback
        self.batches_dir = self.folder / "batches"
        self.batches_dir.mkdir(exist_ok=True)
        self.combined_path = self.folder / "combined_results.txt"
        self.safety_settings = [
            {"category": c, "threshold": "BLOCK_NONE"}
            for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                      "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
        ]
        self.stop_event = threading.Event()

    def log(self, msg: str):
        if self.ui:
            self.ui(msg)
        else:
            print(msg)

    def process(self):
        images = list_images_sorted(self.folder)
        if not images:
            raise RuntimeError("No supported images found in the selected folder.")

        total_batches = (len(images) + self.batch_size - 1) // self.batch_size
        self.log(f"Found {len(images)} images → {total_batches} batch(es) of up to {self.batch_size}.")

        for bi in range(total_batches):
            if self.stop_event.is_set():
                self.log("Stopping immediately!")
                break

            batch = images[bi * self.batch_size: (bi + 1) * self.batch_size]
            batch_index = bi + 1
            batch_file = self.batches_dir / f"batch_{batch_index}.txt"

            if batch_file.exists():
                self.log(f"Skipping batch {batch_index}/{total_batches} (already processed)")
                with self.combined_path.open("a", encoding="utf-8") as cf:
                    cf.write(batch_file.read_text(encoding="utf-8").rstrip("\n") + "\n")
                yield batch_index, total_batches
                continue

            self.log(f"Processing batch {batch_index}/{total_batches} ({len(batch)} images)...")
            parts = [FORMATTING_PROMPT] + [to_inline_part(p) for p in batch]

            # Infinite retry loop
            while not self.stop_event.is_set():
                try:
                    resp = self.model.generate_content(parts, safety_settings=self.safety_settings)
                    text = resp.text.strip()
                    if not text:
                        raise RuntimeError("No usable text in response (API returned empty).")
                    break  # Success
                except Exception as e:
                    self.log(f"  Attempt failed: {e}. Retrying…")

            if self.stop_event.is_set():
                self.log(f"Batch {batch_index} stopped by user.")
                break

            batch_file.write_text(text, encoding="utf-8")
            self.log(f"  Saved → {batch_file}")

            with self.combined_path.open("a", encoding="utf-8") as cf:
                cf.write(text.rstrip("\n") + "\n")

            yield batch_index, total_batches

        self.log(f"Processing terminated or completed. Combined results → {self.combined_path}")


# --- MAIN APPLICATION CLASS ---
class MultiToolApp:
    def __init__(self, root):
        """Initializes the main application window and sets up the tabbed interface."""
        self.root = root
        self.root.title("Image Processing Suite")
        self.root.geometry("800x650")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- Instance Variables ---
        self.is_running = True
        self.gemini_processor = None

        # --- GUI STYLING ---
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.95)
        self.root.configure(bg="#D6EAF8")

        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure("TLabel", background="#D6EAF8", foreground="#154360", font=('Segoe UI', 10))
        style.configure("Bold.TLabel", font=('Segoe UI', 10, 'bold'))
        style.configure("TButton", background="#5DADE2", foreground="white", font=('Segoe UI', 9, 'bold'),
                        borderwidth=0)
        style.map("TButton", background=[('active', '#2E86C1')])
        style.configure("TCheckbutton", background="#D6EAF8", foreground="#154360", font=('Segoe UI', 9))
        style.configure("TProgressbar", thickness=20, troughcolor="#AED6F1", background="#2E86C1")
        style.configure("TFrame", background="#D6EAF8")
        style.configure("TNotebook", background="#D6EAF8", borderwidth=0)
        style.configure("TNotebook.Tab", background="#AED6F1", foreground="#154360", font=('Segoe UI', 10, 'bold'),
                        padding=[10, 5])
        style.map("TNotebook.Tab", background=[("selected", "#5DADE2"), ("active", "#D6EAF8")],
                  foreground=[("selected", "white")])

        # Create the tab controller
        self.notebook = ttk.Notebook(self.root)

        # Create frames for each tab
        self.splitter_tab = ttk.Frame(self.notebook, padding=10)
        self.stitcher_tab = ttk.Frame(self.notebook, padding=10)
        self.ocr_tab = ttk.Frame(self.notebook, padding=10)
        self.gemini_tab = ttk.Frame(self.notebook, padding=15)

        # Add tabs to the notebook
        self.notebook.add(self.splitter_tab, text='1. Image Splitter')
        self.notebook.add(self.stitcher_tab, text='2. Image Stitcher')
        self.notebook.add(self.ocr_tab, text='3. Photos OCR')
        self.notebook.add(self.gemini_tab, text='4. Gemini OCR')

        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        # Populate each tab with its specific widgets
        self.create_splitter_widgets()
        self.create_stitcher_widgets()
        self.create_ocr_widgets()
        self.create_gemini_widgets()

    def on_closing(self):
        """Handle the window closing event."""
        if messagebox.askokcancel("Quit", "Do you want to quit? This will stop any running processes."):
            self.is_running = False
            # Also stop Gemini process if it's running
            if self.gemini_processor:
                self.gemini_processor.stop_event.set()
            self.root.destroy()

    # --- TAB 1: IMAGE SPLITTER ---
    def create_splitter_widgets(self):
        ttk.Label(self.splitter_tab, text="Splits images of tables into individual cells.", style="Bold.TLabel").pack(
            pady=(5, 10))
        folder_frame = ttk.Frame(self.splitter_tab)
        folder_frame.pack(fill='x', padx=20, pady=5)
        ttk.Label(folder_frame, text="📁 Folder:").pack(side=tk.LEFT)
        self.splitter_folder_entry = ttk.Entry(folder_frame, width=42)
        self.splitter_folder_entry.pack(side=tk.LEFT, expand=True, fill='x', padx=5)
        ttk.Button(folder_frame, text="Browse", command=self.select_splitter_folder).pack(side=tk.LEFT)
        self.splitter_button = ttk.Button(self.splitter_tab, text="🚀 Start Image Splitting",
                                          command=self.start_splitter_processing)
        self.splitter_button.pack(pady=15, fill='x', padx=50)
        self.splitter_progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(self.splitter_tab, variable=self.splitter_progress_var, maximum=100)
        progress_bar.pack(fill='x', padx=20, pady=10)
        self.splitter_status_label = ttk.Label(self.splitter_tab, text="Status: Waiting...")
        self.splitter_status_label.pack(pady=10)

    def select_splitter_folder(self):
        folder_path = filedialog.askdirectory(title="Select Input Folder")
        if folder_path:
            self.splitter_folder_entry.delete(0, tk.END)
            self.splitter_folder_entry.insert(0, folder_path)

    def start_splitter_processing(self):
        folder = self.splitter_folder_entry.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Please select a valid folder.")
            return
        self.splitter_button.config(state='disabled')
        threading.Thread(target=self._run_splitter, args=(folder,), daemon=True).start()

    def _run_splitter(self, input_folder):
        output_folder = os.path.join(input_folder, 'split_output')
        image_extensions = ('.png', '.jpg', '.jpeg')
        filenames = [f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]
        filenames.sort(key=natural_sort_key)
        total = len(filenames)
        if total == 0:
            messagebox.showwarning("No Images", "No image files found in the selected folder.")
            self.splitter_button.config(state='normal')
            return
        counter = 1
        for idx, filename in enumerate(filenames, start=1):
            if not self.is_running:
                self.splitter_status_label.config(text="Status: Process terminated.")
                break
            image_path = os.path.join(input_folder, filename)
            self.splitter_status_label.config(text=f"Processing: {filename}")
            cropped_img = self._crop_main_table_area(image_path)
            if cropped_img is not None:
                counter = self._split_table_with_borders(cropped_img, output_folder, counter_start=counter)
            self.splitter_progress_var.set((idx / total) * 100)
            self.root.update_idletasks()
        if self.is_running:
            final_message = f"✅ Done! Output saved to:\n{output_folder}"
            self.splitter_status_label.config(text=final_message)
            messagebox.showinfo("Finished", f"All images processed.\nOutput saved to:\n{output_folder}")
        self.splitter_button.config(state='normal')

    def _crop_main_table_area(self, image_path):
        original_img = cv2.imread(image_path)
        if original_img is None: return None
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        all_points = np.concatenate(contours, axis=0)
        x, y, w, h = cv2.boundingRect(all_points)
        return original_img[y:y + h, x:x + w]

    def _split_table_with_borders(self, table_image, output_dir, counter_start=1):
        os.makedirs(output_dir, exist_ok=True)
        gray = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        height, width = thresh.shape
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 30, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 30))
        detected_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        detected_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        horz_contours, _ = cv2.findContours(detected_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vert_contours, _ = cv2.findContours(detected_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not horz_contours or not vert_contours:
            print(f"Warning: No grid lines detected for an image. Cannot split.")
            return counter_start
        horz_boxes = sorted([cv2.boundingRect(c) for c in horz_contours], key=lambda b: b[1])
        vert_boxes = sorted([cv2.boundingRect(c) for c in vert_contours], key=lambda b: b[0])
        cell_count = 0
        for i in range(len(horz_boxes) - 1):
            for j in range(len(vert_boxes) - 1):
                y1 = horz_boxes[i][1] + horz_boxes[i][3]
                x1 = vert_boxes[j][0] + vert_boxes[j][2]
                y2 = horz_boxes[i + 1][1]
                x2 = vert_boxes[j + 1][0]
                cell = table_image[y1:y2, x1:x2]
                if cell.size > 100:  # Avoid saving tiny/empty cells
                    cell_filename = os.path.join(output_dir, f'{counter_start + cell_count}.png')
                    cv2.imwrite(cell_filename, cell)
                    cell_count += 1
        return counter_start + cell_count

    # --- TAB 2: IMAGE STITCHER ---
    def create_stitcher_widgets(self):
        ttk.Label(self.stitcher_tab, text="Stitches record images together vertically.", style="Bold.TLabel").pack(
            pady=(5, 10))
        folder_frame = ttk.Frame(self.stitcher_tab)
        folder_frame.pack(fill='x', padx=20, pady=5)
        ttk.Label(folder_frame, text="📁 Folder:").pack(side=tk.LEFT)
        self.stitcher_folder_entry = ttk.Entry(folder_frame, width=42)
        self.stitcher_folder_entry.pack(side=tk.LEFT, expand=True, fill='x', padx=5)
        ttk.Button(folder_frame, text="Browse", command=self.select_stitcher_folder).pack(side=tk.LEFT)
        num_frame = ttk.Frame(self.stitcher_tab)
        num_frame.pack(pady=5, padx=20, fill='x')
        ttk.Label(num_frame, text="Start Filename Number:").pack(side=tk.LEFT)
        self.stitcher_start_num_entry = ttk.Entry(num_frame, width=10)
        self.stitcher_start_num_entry.insert(0, "1")
        self.stitcher_start_num_entry.pack(side=tk.LEFT, padx=5)
        limit_frame = ttk.Frame(self.stitcher_tab)
        limit_frame.pack(pady=5, padx=20, fill='x')
        ttk.Label(limit_frame, text="Max Height to Stitch (px):").pack(side=tk.LEFT)
        self.stitcher_height_limit_entry = ttk.Entry(limit_frame, width=10)
        self.stitcher_height_limit_entry.insert(0, "75")
        self.stitcher_height_limit_entry.pack(side=tk.LEFT, padx=5)
        self.stitcher_button = ttk.Button(self.stitcher_tab, text="🚀 Start Stitching",
                                          command=self.start_stitcher_processing)
        self.stitcher_button.pack(pady=15, fill='x', padx=50)
        self.stitcher_progress = ttk.Progressbar(self.stitcher_tab, mode='determinate')
        self.stitcher_progress.pack(fill='x', padx=20, pady=5)
        self.stitcher_status_label = ttk.Label(self.stitcher_tab, text="Progress: 0% | ETA: --")
        self.stitcher_status_label.pack()
        self.stitcher_shutdown_var = tk.BooleanVar()
        ttk.Checkbutton(self.stitcher_tab, text="🔌 Shutdown after completion",
                        variable=self.stitcher_shutdown_var).pack(pady=10)

    def select_stitcher_folder(self):
        folder_path = filedialog.askdirectory(title="Select Input Folder")
        if folder_path:
            self.stitcher_folder_entry.delete(0, tk.END)
            self.stitcher_folder_entry.insert(0, folder_path)

    def start_stitcher_processing(self):
        input_folder = self.stitcher_folder_entry.get().strip()
        try:
            start_number = int(self.stitcher_start_num_entry.get().strip())
            height_limit = int(self.stitcher_height_limit_entry.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Input", "Start number and height limit must be integers.")
            return
        if not os.path.isdir(input_folder):
            messagebox.showerror("Error", "Please select a valid input folder.")
            return
        self.stitcher_button.config(state='disabled')
        threading.Thread(target=self._run_stitcher, args=(input_folder, start_number, height_limit),
                         daemon=True).start()

    def _run_stitcher(self, input_folder, start_number, height_limit):
        start_time = time.time()
        image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
                             key=natural_sort_key)
        output_folder = os.path.join(input_folder, "stitched_output")
        os.makedirs(output_folder, exist_ok=True)
        stitched_images = []
        total = len(image_files)
        i = 0
        while i < total:
            if not self.is_running:
                self.stitcher_status_label.config(text="Status: Process terminated.")
                break
            img_path = os.path.join(input_folder, image_files[i])
            img = cv2.imread(img_path)
            if img is not None and img.shape[0] < height_limit:
                if i + 3 < total:
                    set_paths = [os.path.join(input_folder, image_files[j]) for j in range(i, i + 4)]
                    s1 = self._stitch_vertically(set_paths[0], set_paths[2])
                    s2 = self._stitch_vertically(set_paths[1], set_paths[3])
                    if s1 is not None: stitched_images.append(s1)
                    if s2 is not None: stitched_images.append(s2)
                    i += 4
                else:
                    stitched_images.append(img)
                    i += 1
            else:
                if img is not None: stitched_images.append(img)
                i += 1
            percent = int((i / total) * 100) if total > 0 else 100
            eta_str = f"{int((time.time() - start_time) / i * (total - i))}s" if i > 0 else "--"
            self.stitcher_progress["value"] = percent
            self.stitcher_status_label.config(text=f"Progress: {percent}% | ETA: {eta_str}")
            self.root.update_idletasks()
        if self.is_running:
            for idx, img in enumerate(stitched_images):
                cv2.imwrite(os.path.join(output_folder, f"{start_number + idx}.png"), img)
            self.stitcher_status_label.config(text=f"✅ Completed: {len(stitched_images)} images saved.")
            if self.stitcher_shutdown_var.get():
                os.system("shutdown /s /t 10")
        self.stitcher_button.config(state='normal')

    def _stitch_vertically(self, img1_path, img2_path):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None: return None
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        if w1 != w2:
            width = max(w1, w2)
            img1 = cv2.copyMakeBorder(img1, 0, 0, 0, width - w1, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            img2 = cv2.copyMakeBorder(img2, 0, 0, 0, width - w2, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        return cv2.vconcat([img1, img2])

    # --- TAB 3: PHOTOS OCR ---
    def create_ocr_widgets(self):
        ttk.Label(self.ocr_tab, text="Automates OCR using the Windows Photos App.", style="Bold.TLabel").pack(
            pady=(5, 10))
        ttk.Label(self.ocr_tab, text="⚠️ This will take control of your mouse & keyboard! ⚠️", foreground="red").pack()

        folder_frame = ttk.Frame(self.ocr_tab)
        folder_frame.pack(pady=10, fill='x', padx=20)
        ttk.Label(folder_frame, text="📁 Folder:").pack(side=tk.LEFT)
        self.ocr_folder_entry = ttk.Entry(folder_frame, width=42)
        self.ocr_folder_entry.pack(side=tk.LEFT, expand=True, fill='x', padx=5)
        ttk.Button(folder_frame, text="Browse", command=self.select_ocr_folder).pack(side=tk.LEFT)

        self.ocr_progress_bar = ttk.Progressbar(self.ocr_tab, orient="horizontal", mode="determinate")
        self.ocr_progress_bar.pack(pady=(5, 8), fill='x', padx=20)
        self.ocr_status_label = ttk.Label(self.ocr_tab, text="Status: Waiting to start...")
        self.ocr_status_label.pack()
        self.ocr_eta_label = ttk.Label(self.ocr_tab, text="⏳ ETA: --")
        self.ocr_eta_label.pack(pady=(3, 5))
        self.ocr_shutdown_var = tk.BooleanVar()
        ttk.Checkbutton(self.ocr_tab, text="🔌 Shutdown after completion", variable=self.ocr_shutdown_var).pack()
        self.ocr_button = ttk.Button(self.ocr_tab, text="▶ Start Photos OCR", command=self.start_ocr_processing)
        self.ocr_button.pack(pady=(8, 10), fill='x', padx=50)

    def select_ocr_folder(self):
        folder_path = filedialog.askdirectory(title="Select Input Folder")
        if folder_path:
            self.ocr_folder_entry.delete(0, tk.END)
            self.ocr_folder_entry.insert(0, folder_path)

    def start_ocr_processing(self):
        folder = self.ocr_folder_entry.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Please select a valid folder.")
            return
        response = messagebox.askokcancel("Important",
                                          "The OCR process will take control of your mouse and keyboard. Please do not interfere.\n\nYou can press the 'ESC' key at any time to safely stop the process.")
        if not response:
            return
        self.ocr_button.config(state='disabled')
        threading.Thread(target=self._run_ocr, args=(folder,), daemon=True).start()

    def _trigger_photos_ocr(self):
        for _ in range(17):
            if not self.is_running or keyboard.is_pressed('esc'): raise KeyboardInterrupt
            pyautogui.press('tab')
            time.sleep(0.02)
        pyautogui.press('enter')
        time.sleep(0.6)
        pyautogui.hotkey('ctrl', 'a')
        time.sleep(0.1)
        pyautogui.hotkey('ctrl', 'c')
        time.sleep(0.2)
        return pyperclip.paste().replace("\n", " ").strip()

    def _run_ocr(self, folder_path):
        exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
        images = natsorted([f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in exts])
        total = len(images)
        start_time = time.time()
        if total == 0:
            self.ocr_status_label.config(text="❌ No images found!")
            self.ocr_button.config(state='normal')
            return
        temp_output_file = "photos_ocr_clean_output.txt"
        try:
            with open(temp_output_file, "w", encoding="utf-8") as out_file:
                for idx, img_name in enumerate(images, start=1):
                    if not self.is_running or keyboard.is_pressed('esc'):
                        self.ocr_status_label.config(text="⛔ Terminated by user.")
                        break
                    path = os.path.join(folder_path, img_name)
                    self.ocr_status_label.config(text=f"📷 Processing {idx}/{total}: {img_name}")
                    text, retry_count = "", 0
                    while not text.strip():
                        if not self.is_running or keyboard.is_pressed('esc'): raise KeyboardInterrupt
                        retry_count += 1
                        os.startfile(path)
                        time.sleep(1.5)
                        try:
                            text = self._trigger_photos_ocr()
                        except KeyboardInterrupt:
                            raise
                        except Exception as e:
                            print(f"Error triggering OCR for {img_name}: {e}")
                            text = ""
                        pyautogui.hotkey('alt', 'f4')
                        time.sleep(0.3)
                        if not text.strip():
                            self.ocr_status_label.config(
                                text=f"⚠ No data from {img_name}. Retrying... (Attempt {retry_count})")
                            self.root.update_idletasks()
                            time.sleep(1)
                    out_file.write(text.strip() + "\n")
                    progress = (idx / total) * 100
                    self.ocr_progress_bar["value"] = progress
                    elapsed = time.time() - start_time
                    eta = (elapsed / idx * (total - idx)) if idx > 0 else 0
                    h, m, s = int(eta // 3600), int((eta % 3600) // 60), int(eta % 60)
                    self.ocr_eta_label.config(text=f"⏳ ETA: {h:02d}h:{m:02d}m:{s:02d}s")
                    self.root.update_idletasks()
            if self.is_running and not keyboard.is_pressed('esc'):
                self.ocr_status_label.config(text="🎉 Completed. Moving output file...")
                output_folder = os.path.join(folder_path, "output_folder")
                os.makedirs(output_folder, exist_ok=True)
                final_path = os.path.join(output_folder, os.path.basename(temp_output_file))
                if os.path.exists(final_path): os.remove(final_path)
                shutil.move(temp_output_file, final_path)
                self.ocr_status_label.config(text=f"✅ Output saved to 'output_folder'")
                if self.ocr_shutdown_var.get():
                    os.system("shutdown /s /t 10")
        except KeyboardInterrupt:
            self.ocr_status_label.config(text="⛔ Terminated by user.")
        except Exception as e:
            self.ocr_status_label.config(text=f"An error occurred: {e}")
        finally:
            self.ocr_button.config(state='normal')
            if os.path.exists(temp_output_file):
                os.remove(temp_output_file)

    # --- TAB 4: GEMINI OCR ---
    def create_gemini_widgets(self):
        cfg = load_config()
        self.gemini_api_var = tk.StringVar(value=cfg.get("api_key", ""))
        self.gemini_folder_var = tk.StringVar(value=cfg.get("folder", ""))
        self.gemini_batch_var = tk.StringVar(value=str(cfg.get("batch_size", "10")))
        self.gemini_shutdown_var = tk.BooleanVar()

        frm = self.gemini_tab
        pad = 5

        # Row 0: API Key
        ttk.Label(frm, text="Gemini API Key:").grid(row=0, column=0, sticky="w", pady=pad)
        key_frame = ttk.Frame(frm)
        key_frame.grid(row=0, column=1, columnspan=2, sticky="we")
        self.gemini_api_entry = ttk.Entry(key_frame, textvariable=self.gemini_api_var, show="*", width=60)
        self.gemini_api_entry.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, pad))
        self.gemini_show_key_btn = ttk.Button(key_frame, text="Show", command=self._toggle_gemini_key)
        self.gemini_show_key_btn.pack(side=tk.LEFT)

        # Row 1: Folder
        ttk.Label(frm, text="Image Folder:").grid(row=1, column=0, sticky="w", pady=pad)
        folder_frame = ttk.Frame(frm)
        folder_frame.grid(row=1, column=1, columnspan=2, sticky="we")
        self.gemini_folder_entry = ttk.Entry(folder_frame, textvariable=self.gemini_folder_var, width=60)
        self.gemini_folder_entry.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, pad))
        ttk.Button(folder_frame, text="Browse", command=self._browse_gemini_folder).pack(side=tk.LEFT)

        # Row 2: Batch Size
        ttk.Label(frm, text="Batch Size (img/req):").grid(row=2, column=0, sticky="w", pady=pad)
        self.gemini_batch_entry = ttk.Entry(frm, textvariable=self.gemini_batch_var, width=10)
        self.gemini_batch_entry.grid(row=2, column=1, sticky="w")

        # Row 3: New shutdown checkbox
        ttk.Checkbutton(frm, text="🔌 Shutdown after completion", variable=self.gemini_shutdown_var).grid(row=3,
                                                                                                         column=0,
                                                                                                         columnspan=3,
                                                                                                         sticky='w',
                                                                                                         padx=5,
                                                                                                         pady=pad)

        # Row 4: Buttons
        self.gemini_start_btn = ttk.Button(frm, text="🚀 Start Gemini Processing", command=self._start_gemini_processing)
        self.gemini_start_btn.grid(row=4, column=0, columnspan=2, pady=pad, sticky="we")
        self.gemini_stop_btn = ttk.Button(frm, text="🛑 Stop", command=self._stop_gemini_processing, state=tk.DISABLED)
        self.gemini_stop_btn.grid(row=4, column=2, pady=pad, sticky="we")

        # Row 5: Progress Bar
        self.gemini_progress = ttk.Progressbar(frm, mode="determinate")
        self.gemini_progress.grid(row=5, column=0, columnspan=3, sticky="we", pady=pad)

        # Row 6 & 7: Log
        ttk.Label(frm, text="Status Log:").grid(row=6, column=0, sticky="w", pady=pad)
        self.gemini_log = tk.Text(frm, height=10, wrap="word", relief="solid", borderwidth=1, font=('Consolas', 9))
        self.gemini_log.grid(row=7, column=0, columnspan=3, sticky="nsew", pady=(0, pad))

        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(7, weight=1)

    def _toggle_gemini_key(self):
        if self.gemini_api_entry.cget('show') == '*':
            self.gemini_api_entry.config(show='')
            self.gemini_show_key_btn.config(text="Hide")
        else:
            self.gemini_api_entry.config(show='*')
            self.gemini_show_key_btn.config(text="Show")

    def _browse_gemini_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.gemini_folder_var.set(path)

    def _append_gemini_log(self, msg: str):
        self.gemini_log.insert(tk.END, msg + "\n")
        self.gemini_log.see(tk.END)
        self.root.update_idletasks()

    def _stop_gemini_processing(self):
        if self.gemini_processor:
            self.gemini_processor.stop_event.set()
        self._append_gemini_log("\n🛑 Stop requested! Terminating gracefully...")
        self.gemini_stop_btn.config(state=tk.DISABLED)

    def _start_gemini_processing(self):
        api_key = self.gemini_api_var.get().strip()
        folder_str = self.gemini_folder_var.get().strip()
        batch_str = self.gemini_batch_var.get().strip()

        if not api_key:
            messagebox.showerror("Missing Input", "Please enter your Gemini API key.")
            return
        if not folder_str or not Path(folder_str).is_dir():
            messagebox.showerror("Missing Input", "Please choose a valid images folder.")
            return
        try:
            batch_size = int(batch_str)
            if batch_size < 1: raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Batch size must be a positive integer.")
            return

        save_config(api_key, folder_str, batch_size)
        folder = Path(folder_str)

        self.gemini_start_btn.config(state=tk.DISABLED)
        self.gemini_stop_btn.config(state=tk.NORMAL)
        self.gemini_progress.config(value=0, maximum=1)
        self.gemini_log.delete("1.0", tk.END)

        def run_thread():
            try:
                # ✅ FIX: Call 'after' on self.root, not self
                self.gemini_processor = GeminiBatchProcessor(api_key, folder, batch_size,
                                                             ui_callback=lambda m: self.root.after(0,
                                                                                                   self._append_gemini_log,
                                                                                                   m))
                images = list_images_sorted(folder)
                total_batches = (len(images) + batch_size - 1) // batch_size
                # ✅ FIX: Call 'after' on self.root
                self.root.after(0, self.gemini_progress.config, {"maximum": total_batches, "value": 0})

                if self.gemini_processor.combined_path.exists():
                    self.gemini_processor.combined_path.unlink()

                for done, total in self.gemini_processor.process():
                    # ✅ FIX: Call 'after' on self.root
                    self.root.after(0, self.gemini_progress.config, {"value": done})

                if not self.gemini_processor.stop_event.is_set():
                    # ✅ FIX: Call 'after' on self.root
                    self.root.after(0, messagebox.showinfo, "Done",
                                    f"Processing completed.\nCombined file saved to:\n{self.gemini_processor.combined_path}")
                    if self.gemini_shutdown_var.get():
                        # ✅ FIX: Call 'after' on self.root
                        self.root.after(100, lambda: os.system("shutdown /s /t 10"))

            except Exception as e:
                # ✅ FIX: Call 'after' on self.root
                self.root.after(0, messagebox.showerror, "Error", str(e))
            finally:
                # ✅ FIX: Call 'after' on self.root
                self.root.after(0, self.gemini_start_btn.config, {"state": tk.NORMAL})
                self.root.after(0, self.gemini_stop_btn.config, {"state": tk.DISABLED})
                self.gemini_processor = None

        threading.Thread(target=run_thread, daemon=True).start()


# --- APPLICATION ENTRY POINT ---
if __name__ == "__main__":
    main_root = tk.Tk()
    app = MultiToolApp(main_root)
    main_root.mainloop()