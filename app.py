"""Local bounding-box image labeler. Run with Python 3 and Pillow."""
import json
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageOps, ImageTk


EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, path)


def yolo_line(box, classes, width, height):
    x1, y1, x2, y2 = box['box']
    return f"{classes.index(box['label'])} {(x1+x2)/2/width:.6f} {(y1+y2)/2/height:.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}"


class Labeler:
    def __init__(self, root):
        self.root = root
        self.folder = None
        self.files = []
        self.index = 0
        self.data = {'version': 1, 'classes': [], 'images': {}}
        self.image = None
        self.photo = None
        self.scale = 1
        self.offset = (0, 0)
        self.start = None
        self.draft = None
        self.history = []
        self.dirty = False
        root.title('Local Label — Image annotation')
        root.geometry('1180x780')
        root.minsize(800, 560)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', padding=(12, 7))
        style.configure('TLabel', padding=3)
        toolbar = ttk.Frame(root, padding=12)
        toolbar.pack(fill='x')
        ttk.Label(toolbar, text='LOCAL LABEL', font=('Segoe UI', 15, 'bold')).pack(side='left', padx=(0, 24))
        ttk.Button(toolbar, text='Open folder', command=self.open_folder).pack(side='left')
        ttk.Button(toolbar, text='Export YOLO', command=self.export).pack(side='left', padx=8)
        self.progress = ttk.Label(toolbar, text='No folder open')
        self.progress.pack(side='right')
        body = ttk.Frame(root, padding=(12, 0, 12, 0))
        body.pack(fill='both', expand=True)
        sidebar = ttk.Frame(body, width=250, padding=(0, 0, 12, 0))
        sidebar.pack(side='left', fill='y')
        sidebar.pack_propagate(False)
        ttk.Label(sidebar, text='IMAGES', font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        self.image_list = tk.Listbox(sidebar, exportselection=False, borderwidth=0, font=('Segoe UI', 10))
        self.image_list.pack(fill='both', expand=True, pady=6)
        self.image_list.bind('<<ListboxSelect>>', self.choose_image)
        ttk.Label(sidebar, text='CLASS NAME', font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(12, 0))
        self.label = tk.StringVar(value='object')
        self.classes = ttk.Combobox(sidebar, textvariable=self.label)
        self.classes.pack(fill='x', pady=6)
        ttk.Label(sidebar, text='Type a class, then drag a box.\nSelect a box below to delete it.', wraplength=225).pack(anchor='w')
        ttk.Label(sidebar, text='BOXES', font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(16, 0))
        self.box_list = tk.Listbox(sidebar, height=7, exportselection=False, borderwidth=0, font=('Segoe UI', 10))
        self.box_list.pack(fill='x', pady=6)
        self.box_list.bind('<<ListboxSelect>>', lambda event: self.render())
        actions = ttk.Frame(sidebar)
        actions.pack(fill='x')
        ttk.Button(actions, text='Delete box', command=self.delete_box).pack(side='left')
        ttk.Button(actions, text='Undo', command=self.undo).pack(side='right')
        self.canvas = tk.Canvas(body, background='#18212b', highlightthickness=0, cursor='crosshair')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda event: self.render())
        self.canvas.bind('<ButtonPress-1>', self.begin_box)
        self.canvas.bind('<B1-Motion>', self.drag_box)
        self.canvas.bind('<ButtonRelease-1>', self.end_box)
        footer = ttk.Frame(root, padding=12)
        footer.pack(fill='x')
        ttk.Button(footer, text='← Previous', command=lambda: self.navigate(-1)).pack(side='left')
        ttk.Button(footer, text='Next →', command=lambda: self.navigate(1)).pack(side='left', padx=8)
        self.status = tk.StringVar(value='Open an image folder to begin. Images stay on this computer.')
        ttk.Label(footer, textvariable=self.status).pack(side='left', padx=10)
        root.bind('<Left>', lambda e: self.shortcut(lambda: self.navigate(-1)))
        root.bind('<Right>', lambda e: self.shortcut(lambda: self.navigate(1)))
        root.bind('<Delete>', lambda e: self.shortcut(self.delete_box))
        root.bind('<Control-z>', lambda e: self.shortcut(self.undo))
        root.bind('<Control-s>', lambda e: self.save())
        root.bind('<Escape>', self.cancel_box)
        root.protocol('WM_DELETE_WINDOW', self.close)

    def shortcut(self, action):
        if self.root.focus_get() != self.classes:
            action()

    def open_folder(self):
        if not self.save():
            return
        selected = filedialog.askdirectory(title='Choose a folder of images')
        if not selected:
            return
        folder = Path(selected)
        try:
            files = sorted((p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS and p.is_file()), key=lambda p: p.name.lower())
            if not files:
                messagebox.showinfo('No images', 'This folder has no supported images.')
                return
            path = folder / 'local-label.annotations.json'
            data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'version': 1, 'classes': [], 'images': {}}
            if data.get('version') != 1 or not isinstance(data.get('images'), dict) or not isinstance(data.get('classes'), list):
                raise ValueError('Unsupported annotation file.')
            if any(not isinstance(c, str) or not c.strip() or '\n' in c or '\r' in c for c in data['classes']):
                raise ValueError('Invalid class names.')
            for record in data['images'].values():
                for box in record['boxes']:
                    if box['label'] not in data['classes'] or len(box['box']) != 4 or not all(isinstance(v, (int, float)) for v in box['box']):
                        raise ValueError('Invalid bounding box data.')
        except Exception as exc:
            messagebox.showerror('Cannot open folder', str(exc))
            return
        self.folder, self.files, self.data = folder, files, data
        self.classes['values'] = data['classes']
        if data['classes']:
            self.label.set(data['classes'][0])
        self.image_list.delete(0, 'end')
        for path in files:
            self.image_list.insert('end', path.name)
        self.index = 0
        self.load_image()

    def load_image(self):
        self.cancel_box()
        self.history.clear()
        path = self.files[self.index]
        try:
            with Image.open(path) as source:
                self.image = ImageOps.exif_transpose(source).convert('RGB')
            record = self.data['images'].get(path.name)
            if record and (record['width'], record['height']) != self.image.size:
                raise ValueError('Image dimensions changed since labeling. Restore the original image or use another folder.')
            if record is None:
                self.data['images'][path.name] = {'width': self.image.width, 'height': self.image.height, 'boxes': []}
                self.dirty = True
            self.status.set('Drag to draw • Esc cancels • ← / → navigate • Ctrl+Z undo')
            self.save()
        except Exception as exc:
            self.image = None
            messagebox.showerror('Cannot load image', f'{path.name}\n{exc}')
        self.image_list.selection_clear(0, 'end')
        self.image_list.selection_set(self.index)
        self.image_list.see(self.index)
        self.progress.config(text=f'{self.index + 1} / {len(self.files)}  ·  {path.name}')
        self.refresh_boxes()

    def boxes(self):
        if self.image is None or not self.files:
            return []
        return self.data['images'][self.files[self.index].name]['boxes']

    def choose_image(self, event):
        selection = self.image_list.curselection()
        if selection and selection[0] != self.index and self.save():
            self.index = selection[0]
            self.load_image()

    def navigate(self, step):
        if self.files and 0 <= self.index + step < len(self.files) and self.save():
            self.index += step
            self.load_image()

    def render(self):
        self.canvas.delete('all')
        if self.image is None:
            self.canvas.create_text(self.canvas.winfo_width()/2, self.canvas.winfo_height()/2, text='Your images, your computer.\n\nOpen a folder to start labeling.', fill='#b6c6d8', font=('Segoe UI', 17), justify='center')
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.scale = max(.001, min((cw-32)/self.image.width, (ch-32)/self.image.height))
        w, h = max(1, round(self.image.width*self.scale)), max(1, round(self.image.height*self.scale))
        self.offset = ((cw-w)/2, (ch-h)/2)
        self.photo = ImageTk.PhotoImage(self.image.resize((w, h), Image.Resampling.LANCZOS))
        self.canvas.create_image(*self.offset, image=self.photo, anchor='nw')
        selected = self.box_list.curselection()
        for i, box in enumerate(self.boxes()):
            color = '#ffcc66' if selected and selected[0] == i else '#52dfb1'
            coords = self.screen_coords(box['box'])
            self.canvas.create_rectangle(*coords, outline=color, width=2)
            self.canvas.create_text(coords[0]+4, coords[1]+3, text=f"{i+1} · {box['label']}", fill=color, anchor='nw', font=('Segoe UI', 11, 'bold'))

    def screen_coords(self, box):
        return [value*self.scale + self.offset[i % 2] for i, value in enumerate(box)]

    def image_point(self, event):
        return (max(0, min(self.image.width, (event.x-self.offset[0])/self.scale)), max(0, min(self.image.height, (event.y-self.offset[1])/self.scale)))

    def begin_box(self, event):
        if self.image is None:
            return
        x, y = self.image_point(event)
        if not (0 < x < self.image.width and 0 < y < self.image.height):
            return
        self.start = (x, y)
        self.draft = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='#ffcc66', width=2, dash=(5, 3))

    def drag_box(self, event):
        if self.start:
            self.canvas.coords(self.draft, *self.screen_coords((*self.start, *self.image_point(event))))

    def end_box(self, event):
        if not self.start:
            return
        x1, y1 = self.start
        x2, y2 = self.image_point(event)
        self.cancel_box()
        if abs(x2-x1)*self.scale < 4 or abs(y2-y1)*self.scale < 4:
            return
        label = self.label.get().strip()
        if not label or '\n' in label or '\r' in label:
            messagebox.showinfo('Class name needed', 'Enter a class name before drawing a box.')
            return
        self.remember()
        if label not in self.data['classes']:
            self.data['classes'].append(label)
            self.classes['values'] = self.data['classes']
        self.boxes().append({'label': label, 'box': [round(min(x1,x2), 2), round(min(y1,y2), 2), round(max(x1,x2), 2), round(max(y1,y2), 2)]})
        self.changed()

    def cancel_box(self, event=None):
        self.start = None
        if self.draft:
            self.canvas.delete(self.draft)
        self.draft = None

    def remember(self):
        self.history.append(json.loads(json.dumps(self.boxes())))

    def delete_box(self):
        selected = self.box_list.curselection()
        if selected and self.image is not None:
            self.remember()
            del self.boxes()[selected[0]]
            self.changed()

    def undo(self):
        if self.history:
            self.data['images'][self.files[self.index].name]['boxes'] = self.history.pop()
            self.changed()

    def changed(self):
        self.dirty = True
        self.refresh_boxes()
        self.save()

    def refresh_boxes(self):
        self.box_list.delete(0, 'end')
        for i, box in enumerate(self.boxes()):
            self.box_list.insert('end', f"{i+1}. {box['label']}")
        self.render()

    def save(self):
        if self.folder is None or not self.dirty:
            return True
        try:
            atomic_json(self.folder / 'local-label.annotations.json', self.data)
            self.dirty = False
            self.status.set('Saved locally • Drag to draw • ← / → navigate • Ctrl+Z undo')
            return True
        except OSError as exc:
            self.status.set('NOT SAVED — press Ctrl+S to retry')
            messagebox.showerror('Could not save labels', str(exc))
            return False

    def export(self):
        if not self.files or not self.save():
            return
        target = filedialog.askdirectory(title='Choose a folder for YOLO export')
        if not target:
            return
        # A fresh subfolder protects existing datasets from accidental overwrites.
        destination = Path(target) / 'local-label-export'
        n = 2
        while destination.exists():
            destination = Path(target) / f'local-label-export-{n}'
            n += 1
        records = [(p, self.data['images'][p.name]) for p in self.files if p.name in self.data['images']]
        stems = [p.stem.casefold() for p, _ in records]
        if len(stems) != len(set(stems)):
            messagebox.showerror('Duplicate names', 'YOLO needs unique image names without extensions. Rename images that share the same name before exporting.')
            return
        try:
            destination.mkdir()
            labels = destination / 'labels'
            images = destination / 'images'
            labels.mkdir()
            images.mkdir()
            (destination / 'classes.txt').write_text('\n'.join(self.data['classes']) + '\n', encoding='utf-8')
            for path, record in records:
                with Image.open(path) as source:
                    oriented = ImageOps.exif_transpose(source).convert('RGB')
                    if oriented.size != (record['width'], record['height']):
                        raise ValueError(f'Image dimensions changed: {path.name}')
                    oriented.save(images / (path.stem + '.png'))
                lines = [yolo_line(box, self.data['classes'], record['width'], record['height']) for box in record['boxes']]
                (labels / (path.stem + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
            messagebox.showinfo('Export complete', f'Exported {len(records)} viewed images and labels to:\n{destination}')
        except Exception as exc:
            messagebox.showerror('Export incomplete', f'{exc}\nPartial output: {destination}')

    def close(self):
        if self.save():
            self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    Labeler(root)
    root.mainloop()
