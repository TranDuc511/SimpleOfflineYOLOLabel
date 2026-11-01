"""Local bounding-box image labeler. Run with Python 3 and Pillow."""
import json
import math
import os
from collections import deque
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageOps, ImageTk
from auto_label import CancelledError, detect_objects


EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def file_stamp(path):
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, path)


def yolo_line(box, classes, width, height):
    x1, y1, x2, y2 = box['box']
    return f"{classes.index(box['label'])} {(x1+x2)/2/width:.6f} {(y1+y2)/2/height:.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}"


def read_yolo(path, classes, width, height):
    boxes = []
    if not path.exists():
        return boxes
    for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            category, cx, cy, w, h = line.split()
            category = int(category)
            cx, cy, w, h = map(float, (cx, cy, w, h))
            if not 0 <= category < len(classes) or not all(math.isfinite(v) for v in (cx, cy, w, h)):
                raise ValueError()
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError()
            edges = (cx-w/2, cy-h/2, cx+w/2, cy+h/2)
            if min(edges) < -0.000001 or max(edges) > 1.000001:
                raise ValueError()
            boxes.append({'label': classes[category], 'box': [
                max(0, min(1, value)) * (width if i % 2 == 0 else height)
                for i, value in enumerate(edges)]})
        except (ValueError, IndexError) as exc:
            raise ValueError(f'{path}: invalid YOLO bounding box on line {number}.') from exc
    return boxes


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    os.replace(temporary, path)


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
        self.histories = {}
        self.dirty = False
        self.exporting = False
        self.dataset = None
        self.class_file = None
        self.image_stamp = None
        self.unsaved = set()
        self.auto_jobs = {}
        self.auto_pending = deque()
        self.auto_events = Queue()
        self.auto_running = 0
        self.auto_polling = False
        self.closing = False
        root.title('Local Label — Image annotation')
        root.geometry('1180x780')
        root.minsize(800, 720)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', padding=(12, 7))
        style.configure('TLabel', padding=3)
        toolbar = ttk.Frame(root, padding=12)
        toolbar.pack(fill='x')
        ttk.Label(toolbar, text='LOCAL LABEL', font=('Segoe UI', 15, 'bold')).pack(side='left', padx=(0, 24))
        ttk.Button(toolbar, text='Open folder', command=self.open_folder).pack(side='left')
        ttk.Button(toolbar, text='Open YOLO dataset', command=self.open_dataset).pack(side='left', padx=(8, 0))
        ttk.Button(toolbar, text='Export YOLO', command=self.export).pack(side='left', padx=8)
        self.progress = ttk.Label(toolbar, text='No folder open')
        self.progress.pack(side='right')
        body = ttk.Frame(root, padding=(12, 0, 12, 0))
        body.pack(fill='both', expand=True)
        sidebar = ttk.Frame(body, width=250, padding=(0, 0, 12, 0))
        sidebar.pack(side='left', fill='y')
        sidebar.pack_propagate(False)
        ttk.Label(sidebar, text='IMAGES', font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        self.image_list = tk.Listbox(sidebar, height=5, exportselection=False, borderwidth=0, font=('Segoe UI', 10))
        self.image_list.pack(fill='both', expand=True, pady=6)
        self.image_list.bind('<<ListboxSelect>>', self.choose_image)
        ttk.Label(sidebar, text='CLASS NAME', font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(12, 0))
        self.label = tk.StringVar(value='object')
        self.classes = ttk.Combobox(sidebar, textvariable=self.label)
        self.classes.pack(fill='x', pady=6)
        ttk.Label(sidebar, text='Type a class, then drag a box.\nSelect a box below to delete it.', wraplength=225).pack(anchor='w')
        auto_run = ttk.Frame(sidebar)
        auto_run.pack(fill='x', pady=(8, 0))
        self.auto_button = ttk.Button(auto_run, text='Auto-label', command=self.auto_label)
        self.auto_button.pack(side='left')
        self.cancel_auto_button = ttk.Button(auto_run, text='Cancel queue', command=self.cancel_auto_label, state='disabled')
        self.cancel_auto_button.pack(side='left', padx=4)
        ttk.Label(sidebar, text='Luna · Light (low)\nAuto-apply and save. Images go online.', wraplength=225).pack(anchor='w')
        self.auto_status = tk.StringVar(value='Queue images and keep working.')
        ttk.Label(sidebar, textvariable=self.auto_status, wraplength=225).pack(anchor='w')
        ttk.Label(sidebar, text='BOXES', font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(16, 0))
        self.box_list = tk.Listbox(sidebar, height=5, exportselection=False, borderwidth=0, font=('Segoe UI', 10))
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
        if self.auto_busy():
            self.status.set('Wait for auto-label or Cancel queue before opening another folder.')
            return
        if self.exporting or not self.save():
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
        self.dataset = None
        self.class_file = None
        self.show_files()

    def image_key(self, path):
        return path.relative_to(self.folder).as_posix() if self.dataset else path.name

    def label_path(self, path):
        return (self.dataset / 'labels' / path.relative_to(self.folder)).with_suffix('.txt')

    def open_dataset(self):
        if self.auto_busy():
            self.status.set('Wait for auto-label or Cancel queue before opening another dataset.')
            return
        if self.exporting or not self.save():
            return
        selected = filedialog.askdirectory(title='Choose YOLO dataset root (contains images and labels)')
        if not selected:
            return
        dataset = Path(selected)
        folder = dataset / 'images'
        try:
            files = sorted((p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS), key=lambda p: p.as_posix().lower())
            if not files:
                raise ValueError('No images found in images/. Choose the dataset root containing images/ and labels/.')
            targets = [p.relative_to(folder).with_suffix('.txt').as_posix().casefold() for p in files]
            if len(targets) != len(set(targets)):
                raise ValueError('Images in the same folder must have unique names without extensions.')
            class_file = dataset / 'classes.txt'
            if not class_file.exists():
                chosen = filedialog.askopenfilename(title='Choose classes.txt (one class per line, in class ID order)', filetypes=[('Class names', '*.txt')])
                if not chosen:
                    return
                class_file = Path(chosen)
            classes = class_file.read_text(encoding='utf-8-sig').splitlines()
            if any(not c.strip() or c != c.strip() for c in classes) or len(classes) != len(set(classes)):
                raise ValueError('Class names must be unique, nonempty lines in class ID order.')
        except Exception as exc:
            messagebox.showerror('Cannot open dataset', str(exc))
            return
        self.dataset, self.class_file = dataset, class_file
        self.folder, self.files = folder, files
        self.data = {'version': 1, 'classes': classes, 'images': {}}
        self.show_files()

    def show_files(self):
        self.histories = {}
        self.unsaved.clear()
        self.auto_jobs.clear()
        self.file_indices = {self.image_key(path): i for i, path in enumerate(self.files)}
        data, files = self.data, self.files
        self.classes['values'] = data['classes']
        if data['classes']:
            self.label.set(data['classes'][0])
        self.image_list.delete(0, 'end')
        for path in files:
            self.image_list.insert('end', self.image_key(path))
        self.index = 0
        self.load_image()

    def load_image(self):
        self.cancel_box()
        path = self.files[self.index]
        key = self.image_key(path)
        self.history = self.histories.setdefault(key, [])
        try:
            stamp = file_stamp(path)
            with Image.open(path) as source:
                self.image = (source if self.dataset else ImageOps.exif_transpose(source)).convert('RGB')
            if file_stamp(path) != stamp:
                raise ValueError('Image changed while opening. Please reopen it.')
            self.image_stamp = stamp
            record = self.data['images'].get(key)
            if record and (record['width'], record['height']) != self.image.size:
                raise ValueError('Image dimensions changed since labeling. Restore the original image or use another folder.')
            if record is None:
                boxes = read_yolo(self.label_path(path), self.data['classes'], *self.image.size) if self.dataset else []
                self.data['images'][key] = {'width': self.image.width, 'height': self.image.height, 'boxes': boxes}
                self.dirty = not self.dataset
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
        self.update_auto_status()

    def boxes(self):
        if self.image is None or not self.files:
            return []
        return self.data['images'][self.image_key(self.files[self.index])]['boxes']

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
        self.scale = max(.001, min((cw-32)/self.image.width, (ch-64)/self.image.height))
        w, h = max(1, round(self.image.width*self.scale)), max(1, round(self.image.height*self.scale))
        self.offset = ((cw-w)/2, (ch-h)/2)
        self.photo = ImageTk.PhotoImage(self.image.resize((w, h), Image.Resampling.LANCZOS))
        self.canvas.create_image(*self.offset, image=self.photo, anchor='nw')
        selected = self.box_list.curselection()
        for i, box in enumerate(self.boxes()):
            color = '#ffcc66' if selected and selected[0] == i else '#52dfb1'
            coords = self.screen_coords(box['box'])
            self.canvas.create_rectangle(*coords, outline=color, width=2)
            self.canvas.create_text(coords[0]+4, coords[1]-3, text=f"{i+1} · {box['label']}", fill=color, anchor='sw', font=('Segoe UI', 11, 'bold'))

    def auto_busy(self):
        return bool(self.auto_running or self.auto_pending)

    def auto_label(self):
        if self.image is None or self.exporting or self.closing:
            return
        label = self.label.get().strip()
        if not label or '\n' in label or '\r' in label:
            messagebox.showinfo('Class name needed', 'Enter the object name to detect on a single line.')
            return
        if not self.save():
            return
        path = self.files[self.index]
        key = self.image_key(path)
        existing = self.auto_jobs.get((key, label))
        if existing and existing['state'] not in ('error', 'cancelled'):
            self.update_auto_status()
            return
        try:
            if file_stamp(path) != self.image_stamp:
                raise ValueError('Image changed on disk. Reopen it before auto-labeling.')
        except (OSError, ValueError) as exc:
            messagebox.showerror('Cannot queue image', str(exc))
            return
        job = {'key': key, 'path': path, 'label': label, 'size': self.image.size,
               'stamp': self.image_stamp, 'dataset': bool(self.dataset),
               'cancel': Event(), 'state': 'queued', 'boxes': [], 'error': None}
        self.auto_jobs[key, label] = job
        self.auto_pending.append(job)
        self.refresh_auto_row(key)
        self.start_auto_jobs()
        if not self.auto_polling:
            self.auto_polling = True
            self.root.after(50, self.poll_auto_jobs)

    def start_auto_jobs(self):
        # Queue paths, not decoded images, so waiting jobs do not fill memory.
        while self.auto_running < 2 and self.auto_pending and not self.closing:
            job = self.auto_pending.popleft()
            job['state'] = 'running'
            self.auto_running += 1
            self.refresh_auto_row(job['key'])

            def detect(job=job):
                try:
                    if job['cancel'].is_set():
                        raise CancelledError()
                    if file_stamp(job['path']) != job['stamp']:
                        raise ValueError('Image changed while queued. Reopen it and retry.')
                    with Image.open(job['path']) as source:
                        image = (source if job['dataset'] else ImageOps.exif_transpose(source)).convert('RGB')
                    if image.size != job['size'] or file_stamp(job['path']) != job['stamp']:
                        raise ValueError('Image changed while opening. Reopen it and retry.')
                    result = detect_objects(image, job['label'], job['cancel'])
                    self.auto_events.put((job, 'done', result))
                except CancelledError:
                    self.auto_events.put((job, 'cancelled', None))
                except Exception as exc:
                    self.auto_events.put((job, 'error', str(exc)))

            Thread(target=detect, daemon=True).start()
        self.update_auto_status()

    def poll_auto_jobs(self):
        try:
            while True:
                job, kind, value = self.auto_events.get_nowait()
                self.auto_running -= 1
                if job['cancel'].is_set() or self.closing or kind == 'cancelled':
                    job['state'] = 'cancelled'
                elif kind == 'error':
                    job['state'], job['error'] = 'error', value
                    messagebox.showerror('Auto-label failed', f"{job['key']} · {job['label']}\n{value}")
                else:
                    job['boxes'] = value
                    try:
                        if file_stamp(job['path']) != job['stamp']:
                            raise ValueError('Image changed during detection. Result was not applied; reopen it and retry.')
                        self.apply_auto_result(job)
                    except (OSError, ValueError) as exc:
                        job['state'], job['error'] = 'error', str(exc)
                        messagebox.showerror('Auto-label failed', f"{job['key']}\n{exc}")
                self.refresh_auto_row(job['key'])
        except Empty:
            pass
        self.start_auto_jobs()
        if self.auto_busy():
            self.root.after(50, self.poll_auto_jobs)
        else:
            self.auto_polling = False

    def apply_auto_result(self, job):
        key = job['key']
        record = self.data['images'][key]
        if (record['width'], record['height']) != job['size']:
            raise ValueError('Image dimensions no longer match this result.')
        additions = [box for box in job['boxes'] if box not in record['boxes']]
        if additions:
            self.histories.setdefault(key, []).append(json.loads(json.dumps(record['boxes'])))
            for box in additions:
                if box['label'] not in self.data['classes']:
                    self.data['classes'].append(box['label'])
            record['boxes'].extend(additions)
            self.classes['values'] = self.data['classes']
            self.unsaved.add(key)
            job['state'] = 'done' if self.save() else 'save_error'
            if self.files and self.image_key(self.files[self.index]) == key:
                self.cancel_box()
                self.refresh_boxes()
        else:
            job['state'] = 'done' if job['boxes'] else 'empty'

    def refresh_auto_row(self, key):
        states = [job['state'] for job in self.auto_jobs.values() if job['key'] == key]
        state = next((s for s in ('save_error', 'error', 'cancelling', 'running', 'queued') if s in states), states[-1] if states else '')
        index = self.file_indices[key]
        selected = index in self.image_list.curselection()
        self.image_list.delete(index)
        self.image_list.insert(index, f'[{state}] {key}' if state else key)
        if selected:
            self.image_list.selection_set(index)

    def update_auto_status(self):
        self.cancel_auto_button.config(state='normal' if self.auto_busy() else 'disabled')
        detail = 'Queue images and keep working.'
        if self.files:
            key = self.image_key(self.files[self.index])
            jobs = [job for job in self.auto_jobs.values() if job['key'] == key]
            if jobs:
                job = jobs[-1]
                detail = f"{job['state']}: {job['label'][:32]} ({len(job['boxes'])} boxes)"
        self.auto_status.set(f'{self.auto_running} running · {len(self.auto_pending)} queued\n{detail}')

    def cancel_auto_label(self):
        for job in self.auto_jobs.values():
            if job['state'] in ('queued', 'running'):
                job['cancel'].set()
                job['state'] = 'cancelled' if job['state'] == 'queued' else 'cancelling'
                self.refresh_auto_row(job['key'])
        self.auto_pending.clear()
        self.update_auto_status()

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
            self.data['images'][self.image_key(self.files[self.index])]['boxes'] = self.history.pop()
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
        if self.folder is None:
            return True
        if self.dirty:
            self.unsaved.add(self.image_key(self.files[self.index]))
        if not self.unsaved:
            return True
        try:
            if self.dataset:
                # Write class names first so a new ID never points to a missing class.
                names = '\n'.join(self.data['classes']) + ('\n' if self.data['classes'] else '')
                if self.class_file.read_text(encoding='utf-8-sig') != names:
                    atomic_text(self.class_file, names)
                for key in list(self.unsaved):
                    record = self.data['images'][key]
                    lines = [yolo_line(box, self.data['classes'], record['width'], record['height']) for box in record['boxes']]
                    atomic_text(self.label_path(self.folder / key), '\n'.join(lines) + ('\n' if lines else ''))
                    self.unsaved.remove(key)
            else:
                atomic_json(self.folder / 'local-label.annotations.json', self.data)
                self.unsaved.clear()
            self.dirty = False
            for job in self.auto_jobs.values():
                if job['state'] == 'save_error':
                    job['state'] = 'done'
                    self.refresh_auto_row(job['key'])
            self.update_auto_status()
            self.status.set('Saved YOLO labels in place' if self.dataset else 'Saved locally • Drag to draw • ← / → navigate • Ctrl+Z undo')
            return True
        except OSError as exc:
            self.dirty = self.image_key(self.files[self.index]) in self.unsaved
            self.status.set('NOT SAVED — press Ctrl+S to retry')
            messagebox.showerror('Could not save labels', str(exc))
            return False

    def export(self):
        if self.auto_busy():
            self.status.set('Wait for auto-label or Cancel queue before exporting.')
            return
        if self.exporting or not self.files or not self.save():
            return
        if self.dataset:
            messagebox.showinfo('Labels saved in place', 'YOLO edits save automatically to this dataset. No export or image processing is needed.')
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
        records = [(p, self.data['images'][self.image_key(p)]) for p in self.files if self.image_key(p) in self.data['images']]
        stems = [p.stem.casefold() for p, _ in records]
        if len(stems) != len(set(stems)):
            messagebox.showerror('Duplicate names', 'YOLO needs unique image names without extensions. Rename images that share the same name before exporting.')
            return
        classes = self.data['classes'][:]
        events = Queue()
        dialog = tk.Toplevel(self.root)
        dialog.title('Export YOLO')
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.protocol('WM_DELETE_WINDOW', lambda: None)
        progress_text = tk.StringVar(value=f'Exporting 0 / {len(records)} images (0%)')
        ttk.Label(dialog, textvariable=progress_text, padding=16).pack(fill='x')
        bar = ttk.Progressbar(dialog, mode='determinate', maximum=max(1, len(records)), length=400)
        bar.pack(fill='x', padx=16, pady=(0, 16))
        dialog.grab_set()
        self.exporting = True

        def write_export():
            try:
                destination.mkdir()
                labels = destination / 'labels'
                images = destination / 'images'
                labels.mkdir()
                images.mkdir()
                (destination / 'classes.txt').write_text('\n'.join(classes) + '\n', encoding='utf-8')
                for count, (path, record) in enumerate(records, 1):
                    with Image.open(path) as source:
                        oriented = ImageOps.exif_transpose(source).convert('RGB')
                        if oriented.size != (record['width'], record['height']):
                            raise ValueError(f'Image dimensions changed: {path.name}')
                        oriented.save(images / (path.stem + '.png'))
                    lines = [yolo_line(box, classes, record['width'], record['height']) for box in record['boxes']]
                    (labels / (path.stem + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
                    events.put(('progress', count))
                events.put(('done', None))
            except Exception as exc:
                events.put(('error', str(exc)))

        def update_progress():
            try:
                while True:
                    kind, value = events.get_nowait()
                    if kind == 'progress':
                        bar['value'] = value
                        progress_text.set(f'Exporting {value} / {len(records)} images ({value * 100 // len(records)}%)')
                    else:
                        dialog.grab_release()
                        dialog.destroy()
                        self.exporting = False
                        if kind == 'done':
                            messagebox.showinfo('Export complete', f'Exported {len(records)} viewed images and labels to:\n{destination}')
                        else:
                            messagebox.showerror('Export incomplete', f'{value}\nPartial output: {destination}')
                        return
            except Empty:
                pass
            self.root.after(50, update_progress)

        Thread(target=write_export, daemon=True).start()
        self.root.after(50, update_progress)

    def close(self):
        if self.auto_busy():
            self.closing = True
            self.cancel_auto_label()
            self.root.after(100, self.close)
            return
        self.closing = False
        if not self.exporting and self.save():
            self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    Labeler(root)
    root.mainloop()
