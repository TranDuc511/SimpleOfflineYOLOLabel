# Local Label

A small offline Windows desktop app for drawing bounding boxes on images.

Double-click **Start Labeler.bat**. Requires Python with Tkinter and Pillow (already available on this computer). On another computer: `python -m pip install -r requirements.txt`.

1. Open a folder containing JPG, PNG, BMP, WebP or TIFF images.
2. Type or select a class name, then drag a box around an object.
3. Use Previous / Next or the arrow keys to move through images.
4. Select a box in the sidebar and choose Delete box to remove it. Ctrl+Z undoes box changes on the current image. Escape cancels drawing.

Labels save automatically in `local-label.annotations.json` inside the image folder. Reopen that folder to resume. Keep this file with your images. Originals are never edited. The folder must be writable. Images are shown fitted to the window; enlarge the window for finer boxes. To change a box, delete it and draw it again.

**Export YOLO** creates a new export subfolder with `classes.txt`, `images/`, and `labels/`. It includes only images you have viewed, including empty label files for viewed images without objects. PNG copies have EXIF orientation applied so exported coordinates match the exported images. Each label line is `class_id center_x center_y width height`, normalized to 0–1. Class IDs follow the order in `classes.txt`. Training configuration and train/validation splitting are not included.

No server, account, internet connection, or image upload is used.
