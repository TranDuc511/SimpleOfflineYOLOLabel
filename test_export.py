import tempfile
import time
import tkinter as tk
import unittest
from pathlib import Path
from threading import Event
from tkinter import ttk
from unittest.mock import patch

from PIL import Image

from app import Labeler


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.app = Labeler(self.root)
        self.app.data['classes'] = ['object']
        for index in range(3):
            path = self.folder / f'image{index}.jpg'
            Image.new('RGB', (20, 10)).save(path)
            self.app.files.append(path)
            self.app.data['images'][path.name] = {
                'width': 20, 'height': 10,
                'boxes': [{'label': 'object', 'box': [0, 0, 20, 10]}] if index == 0 else [],
            }

    def wait_for_export(self):
        deadline = time.monotonic() + 5
        while self.app.exporting and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.005)
        self.assertFalse(self.app.exporting, 'Export did not finish')
        self.assertIsNone(self.root.grab_current())
        self.assertFalse(any(isinstance(w, tk.Toplevel) for w in self.root.winfo_children()))

    def test_progress_and_export_contents(self):
        original_save = Image.Image.save
        continue_export = Event()
        save_count = 0

        def slow_save(image, *args, **kwargs):
            nonlocal save_count
            save_count += 1
            if save_count == 2 and not continue_export.wait(5):
                raise TimeoutError('UI did not update during export')
            return original_save(image, *args, **kwargs)

        with patch('app.filedialog.askdirectory', return_value=str(self.folder)) as choose, \
                patch('app.messagebox.showinfo') as info, \
                patch('app.messagebox.showerror') as error, \
                patch.object(Image.Image, 'save', slow_save):
            self.app.export()
            self.app.export()
            choose.assert_called_once()
            self.app.close()
            self.assertTrue(self.root.winfo_exists())
            dialog = next(w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel))
            bar = next(w for w in dialog.winfo_children() if isinstance(w, ttk.Progressbar))
            label = next(w for w in dialog.winfo_children() if isinstance(w, ttk.Label))
            observations = []

            def observe_progress():
                if float(bar['value']) > 0:
                    observations.append((float(bar['value']), self.root.getvar(label['textvariable'])))
                    continue_export.set()
                elif self.app.exporting:
                    self.root.after(10, observe_progress)

            self.root.after(10, observe_progress)
            try:
                self.wait_for_export()
            finally:
                continue_export.set()
            self.assertEqual(observations, [(1, 'Exporting 1 / 3 images (33%)')])
            info.assert_called_once()
            error.assert_not_called()
        output = self.folder / 'local-label-export'
        self.assertEqual((output / 'classes.txt').read_text(), 'object\n')
        self.assertEqual((output / 'labels/image0.txt').read_text(), '0 0.500000 0.500000 1.000000 1.000000\n')
        self.assertEqual((output / 'labels/image1.txt').read_text(), '')
        self.assertEqual(len(list((output / 'images').glob('*.png'))), 3)

    def test_export_error_releases_dialog_and_allows_retry(self):
        self.app.data['images']['image0.jpg']['width'] = 99
        with patch('app.filedialog.askdirectory', return_value=str(self.folder)), \
                patch('app.messagebox.showinfo') as info, \
                patch('app.messagebox.showerror') as error:
            self.app.export()
            self.wait_for_export()
            info.assert_not_called()
            self.assertIn('Image dimensions changed', error.call_args.args[1])
            self.app.data['images']['image0.jpg']['width'] = 20
            self.app.export()
            self.wait_for_export()
            info.assert_called_once()
            self.assertTrue((self.folder / 'local-label-export-2/images/image0.png').exists())


if __name__ == '__main__':
    unittest.main()
