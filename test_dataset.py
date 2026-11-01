import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app import Labeler, read_yolo


class DatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        self.dataset = Path(temporary.name)
        (self.dataset / 'images/train').mkdir(parents=True)
        (self.dataset / 'labels/train').mkdir(parents=True)
        (self.dataset / 'classes.txt').write_text('cat\ndog\n')
        for name in ('a', 'b'):
            Image.new('RGB', (100, 50)).save(self.dataset / f'images/train/{name}.png')
            (self.dataset / f'labels/train/{name}.txt').write_text('1 0.5 0.5 0.4 0.4\n')
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.app = Labeler(self.root)

    def open_dataset(self):
        with patch('app.filedialog.askdirectory', return_value=str(self.dataset)):
            self.app.open_dataset()

    def test_lazy_load_edit_undo_and_reopen(self):
        original_images = {p: p.read_bytes() for p in (self.dataset / 'images').rglob('*.png')}
        other_label = self.dataset / 'labels/train/b.txt'
        original_label = other_label.read_bytes()
        original_time = other_label.stat().st_mtime_ns
        with patch('app.Image.open', wraps=Image.open) as opened:
            self.open_dataset()
            self.assertEqual(opened.call_count, 1)
        self.assertEqual(self.app.boxes(), [{'label': 'dog', 'box': [30.0, 15.0, 70.0, 35.0]}])
        self.app.box_list.selection_set(0)
        self.app.delete_box()
        label = self.dataset / 'labels/train/a.txt'
        self.assertEqual(label.read_text(), '')
        self.app.undo()
        self.assertEqual(label.read_text(), '1 0.500000 0.500000 0.400000 0.400000\n')
        self.app.data['classes'].append('bird')
        self.app.boxes().append({'label': 'bird', 'box': [0, 0, 100, 50]})
        self.app.changed()
        self.open_dataset()
        self.assertEqual([b['label'] for b in self.app.boxes()], ['dog', 'bird'])
        self.assertEqual(other_label.read_bytes(), original_label)
        self.assertEqual(other_label.stat().st_mtime_ns, original_time)
        for path, content in original_images.items():
            self.assertEqual(path.read_bytes(), content)
        with patch('app.messagebox.showinfo'):
            self.app.export()
        self.assertFalse((self.dataset / 'local-label-export').exists())

    def test_missing_labels_stay_missing_until_edited(self):
        label = self.dataset / 'labels/train/a.txt'
        label.unlink()
        self.open_dataset()
        self.assertEqual(self.app.boxes(), [])
        self.assertFalse(label.exists())
        self.app.boxes().append({'label': 'cat', 'box': [0, 0, 50, 25]})
        self.app.changed()
        self.assertTrue(label.exists())

    def test_invalid_label_is_not_overwritten(self):
        label = self.dataset / 'labels/train/a.txt'
        label.write_text('9 0.5 0.5 1 1\n')
        with patch('app.messagebox.showerror') as error:
            self.open_dataset()
        error.assert_called_once()
        self.assertIsNone(self.app.image)
        self.assertEqual(label.read_text(), '9 0.5 0.5 1 1\n')

    def test_rejects_invalid_coordinates(self):
        label = self.dataset / 'invalid.txt'
        for line in ('0 nan 0.5 1 1', '0 0.5 0.5 -1 1', '0 1 1 1 1', '0 0.5 0.5 1 1 extra'):
            with self.subTest(line=line):
                label.write_text(line)
                with self.assertRaises(ValueError):
                    read_yolo(label, ['cat'], 100, 50)

    def test_failed_save_blocks_navigation_and_can_retry(self):
        self.open_dataset()
        self.app.boxes().clear()
        self.app.dirty = True
        with patch('app.atomic_text', side_effect=OSError('disk full')), patch('app.messagebox.showerror'):
            self.app.navigate(1)
        self.assertEqual(self.app.index, 0)
        self.assertTrue(self.app.dirty)
        self.app.navigate(1)
        self.assertEqual(self.app.index, 1)
        self.assertEqual((self.dataset / 'labels/train/a.txt').read_text(), '')


if __name__ == '__main__':
    unittest.main()
