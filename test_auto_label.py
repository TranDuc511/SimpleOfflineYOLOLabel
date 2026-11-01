import json
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Event, Timer
import time
import tkinter as tk
import unittest
from unittest.mock import patch

from PIL import Image

from app import Labeler
from auto_label import CancelledError, detect_objects, parse_boxes


class DetectionTests(unittest.TestCase):
    def setUp(self):
        # Keep subprocess fixtures inside the writable workspace under Codex's sandbox.
        temporary_root = patch('tempfile.tempdir', str(Path(__file__).parent.resolve()))
        temporary_root.start()
        self.addCleanup(temporary_root.stop)

    def test_normalized_boxes_map_to_original_image(self):
        item = {'x1': .1, 'y1': .2, 'x2': .8, 'y2': .9}
        self.assertEqual(parse_boxes({'boxes': [item, item]}, 'mũ', (4000, 2000)),
                         [{'label': 'mũ', 'box': [400, 400, 3200, 1800]}])
        self.assertEqual(parse_boxes({'boxes': []}, 'cat', (100, 50)), [])

    def test_invalid_response_rejects_entire_batch(self):
        good = {'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1}
        invalid = [None, {}, {'boxes': None}, {'boxes': [None]},
                   {'boxes': [good], 'other': 1}]
        for value in (float('nan'), float('inf'), True, '0.1', -.1, 1.1):
            invalid.append({'boxes': [good, dict(good, x1=value)]})
        invalid += [{'boxes': [dict(good, x1=1)]}, {'boxes': [dict(good, y2=0)]},
                    {'boxes': [dict(good, label='other')]}]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                parse_boxes(payload, 'cat', (100, 50))

    def test_subprocess_contract_and_cleanup(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            script = Path(temporary) / 'fake_codex.py'
            capture = Path(temporary) / 'capture.json'
            script.write_text(
                'import json, sys\n'
                'from pathlib import Path\n'
                'from PIL import Image\n'
                'args = sys.argv[1:]\n'
                'image = Path(args[args.index("--image") + 1])\n'
                'prompt = sys.stdin.read()\n'
                f'Path({str(capture)!r}).write_text(json.dumps({{"args": args, "prompt": prompt, "size": Image.open(image).size, "exif": dict(Image.open(image).getexif()), "temporary": str(image.parent)}}))\n'
                'Path(args[args.index("--output-last-message") + 1]).write_text(json.dumps({"boxes": [{"x1": .1, "y1": .2, "x2": .8, "y2": .9}]}))\n',
                encoding='utf-8')
            with patch('auto_label.codex_command', return_value=[sys.executable, str(script)]):
                image = Image.new('RGB', (4096, 2048))
                exif = Image.Exif()
                exif[315] = 'private image author'
                image.info['exif'] = exif.tobytes()
                result = detect_objects(image, 'cat & $(echo hi)', Event())
            data = json.loads(capture.read_text())
            self.assertEqual(data['size'], [2048, 1024])
            self.assertEqual(data['exif'], {})
            self.assertIn('cat & $(echo hi)', data['prompt'])
            self.assertIn('exactly 2048 pixels wide and 1024 pixels high', data['prompt'])
            self.assertIn('--ignore-user-config', data['args'])
            self.assertEqual(data['args'][data['args'].index('--model') + 1], 'gpt-6-luna')
            self.assertIn('model_reasoning_effort="low"', data['args'])
            self.assertEqual(result[0]['box'], [409.6, 409.6, 3276.8, 1843.2])
            self.assertFalse(Path(data['temporary']).exists())

    def test_failed_process_and_invalid_json_are_errors_not_empty_results(self):
        cases = [
            ('import sys; print("login required"); sys.exit(1)', RuntimeError),
            ('import sys; from pathlib import Path; Path(sys.argv[sys.argv.index("--output-last-message")+1]).write_text("broken")', ValueError),
        ]
        for code, error in cases:
            with self.subTest(code=code), patch('auto_label.codex_command', return_value=[sys.executable, '-c', code]):
                with self.assertRaises(error):
                    detect_objects(Image.new('RGB', (20, 10)), 'cat', Event())

    def test_timeout_terminates_process(self):
        processes = []
        original = subprocess.Popen

        def capture_process(*args, **kwargs):
            process = original(*args, **kwargs)
            processes.append(process)
            return process

        with patch('auto_label.codex_command', return_value=[sys.executable, '-c', 'import time; time.sleep(60)']), \
                patch('auto_label.subprocess.Popen', side_effect=capture_process):
            with self.assertRaises(TimeoutError):
                detect_objects(Image.new('RGB', (20, 10)), 'cat', Event(), timeout=.3)
        self.assertIsNotNone(processes[0].poll())

    def test_cancelled_request_never_launches_codex(self):
        cancel = Event()
        cancel.set()
        with patch('auto_label.codex_command') as command, self.assertRaises(CancelledError):
            detect_objects(Image.new('RGB', (20, 10)), 'cat', cancel)
        command.assert_not_called()

    def test_cancel_running_subprocess(self):
        cancel = Event()
        timer = Timer(.3, cancel.set)
        with patch('auto_label.codex_command', return_value=[sys.executable, '-c', 'import time; time.sleep(60)']):
            timer.start()
            try:
                with self.assertRaises(CancelledError):
                    detect_objects(Image.new('RGB', (20, 10)), 'cat', cancel)
            finally:
                timer.cancel()


class AutoLabelUITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        self.dataset = Path(temporary.name)
        (self.dataset / 'images').mkdir()
        (self.dataset / 'labels').mkdir()
        (self.dataset / 'classes.txt').write_text('cat\ndog\n')
        for index, name in enumerate(('a', 'b', 'c', 'd')):
            Image.new('RGB', (100, 50), (index * 50, 0, 0)).save(self.dataset / f'images/{name}.png')
            (self.dataset / f'labels/{name}.txt').write_text('1 0.5 0.5 0.4 0.4\n')
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.app = Labeler(self.root)
        with patch('app.filedialog.askdirectory', return_value=str(self.dataset)):
            self.app.open_dataset()
        self.app.label.set('mũ')
        self.proposed = [{'label': 'mũ', 'box': [0, 0, 50, 25]}]

    def wait_until(self, predicate):
        deadline = time.monotonic() + 5
        while not predicate() and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.005)
        self.assertTrue(predicate(), 'Background processing did not finish')

    def wait_for_detection(self):
        self.wait_until(lambda: not self.app.auto_busy() and not self.app.auto_polling)

    def test_auto_apply_save_and_undo_survives_navigation(self):
        with patch('app.detect_objects', return_value=self.proposed) as detect:
            self.app.auto_label()
            self.wait_for_detection()
            self.app.auto_label()
        detect.assert_called_once()
        self.assertEqual(self.app.auto_jobs['a.png', 'mũ']['state'], 'done')
        self.assertEqual(self.app.auto_jobs['a.png', 'mũ']['boxes'], self.proposed)
        self.assertEqual((self.dataset / 'classes.txt').read_text(encoding='utf-8'), 'cat\ndog\nmũ\n')
        label_file = self.dataset / 'labels/a.txt'
        self.assertEqual(label_file.read_text(), '1 0.500000 0.500000 0.400000 0.400000\n2 0.250000 0.250000 0.500000 0.500000\n')
        self.app.navigate(1)
        self.app.navigate(-1)
        self.app.undo()
        self.assertEqual(label_file.read_text(), '1 0.500000 0.500000 0.400000 0.400000\n')
        self.assertEqual(len(self.app.boxes()), 1)

    def test_navigation_continues_and_saves_to_original_image(self):
        started, release = Event(), Event()
        other_file = self.dataset / 'labels/b.txt'
        original_other = other_file.read_bytes()

        def slow_detect(image, label, cancel):
            started.set()
            release.wait(5)
            self.assertFalse(cancel.is_set())
            return self.proposed

        with patch('app.detect_objects', side_effect=slow_detect):
            self.app.auto_label()
            try:
                self.assertTrue(started.wait(2))
                self.app.navigate(1)
                self.app.label.set('different class')
                self.assertEqual(self.app.auto_jobs['a.png', 'mũ']['state'], 'running')
            finally:
                release.set()
                self.wait_for_detection()
        self.assertEqual(self.app.index, 1)
        self.assertEqual(len(self.app.boxes()), 1)
        self.assertEqual(other_file.read_bytes(), original_other)
        self.assertEqual(len(self.app.data['images']['a.png']['boxes']), 2)
        self.assertIn('2 0.250000', (self.dataset / 'labels/a.txt').read_text())
        self.app.navigate(-1)
        self.assertEqual(len(self.app.boxes()), 2)
        self.assertEqual(len(self.app.history), 1)

    def test_two_workers_fifo_and_out_of_order_results_keep_class_ids(self):
        releases = {name: Event() for name in ('a', 'b', 'c')}
        started = {name: Event() for name in releases}

        def controlled(image, label, cancel):
            started[label].set()
            releases[label].wait(5)
            return [{'label': label, 'box': [0, 0, 50, 25]}]

        with patch('app.detect_objects', side_effect=controlled):
            try:
                for name in releases:
                    self.app.label.set(name)
                    self.app.auto_label()
                    if name != 'c':
                        self.app.navigate(1)
                self.assertTrue(started['a'].wait(2))
                self.assertTrue(started['b'].wait(2))
                self.assertEqual(self.app.auto_running, 2)
                self.assertEqual(len(self.app.auto_pending), 1)
                self.assertFalse(started['c'].is_set())
                releases['b'].set()
                self.wait_until(lambda: started['c'].is_set())
                self.assertEqual(self.app.auto_jobs['b.png', 'b']['state'], 'done')
                self.assertEqual(self.app.data['classes'], ['cat', 'dog', 'b'])
                releases['c'].set()
                self.wait_until(lambda: self.app.auto_jobs['c.png', 'c']['state'] == 'done')
                releases['a'].set()
                self.wait_for_detection()
            finally:
                for release in releases.values():
                    release.set()
                self.wait_for_detection()
        self.assertEqual(self.app.data['classes'], ['cat', 'dog', 'b', 'c', 'a'])
        for name, category in (('a', 4), ('b', 2), ('c', 3)):
            self.assertIn(f'{category} 0.250000', (self.dataset / f'labels/{name}.txt').read_text())
        self.assertEqual((self.dataset / 'labels/d.txt').read_text(), '1 0.5 0.5 0.4 0.4\n')
        self.assertEqual(self.app.index, 2)

    def test_empty_and_failure_do_not_write_labels_and_errors_can_retry(self):
        label_file = self.dataset / 'labels/a.txt'
        before = label_file.read_bytes()
        for label, outcome, state in (('empty', [], 'empty'), ('failure', RuntimeError('login required'), 'error')):
            self.app.label.set(label)
            with patch('app.detect_objects', side_effect=outcome if isinstance(outcome, Exception) else None,
                       return_value=outcome), patch('app.messagebox.showerror') as error:
                self.app.auto_label()
                self.wait_for_detection()
                self.assertEqual(self.app.auto_jobs['a.png', label]['state'], state)
                self.assertEqual(label_file.read_bytes(), before)
                if isinstance(outcome, Exception):
                    error.assert_called_once()
                else:
                    error.assert_not_called()
        with patch('app.detect_objects', return_value=[]):
            self.app.auto_label()
            self.wait_for_detection()
        self.assertEqual(self.app.auto_jobs['a.png', 'failure']['state'], 'empty')

    def test_cancel_queue_stops_running_and_never_starts_waiting_images(self):
        started = []

        def cancellable(image, label, cancel):
            started.append(label)
            if not cancel.wait(5):
                raise AssertionError('UI failed to cancel')
            return self.proposed  # Late success after cancel must not be applied.

        with patch('app.detect_objects', side_effect=cancellable) as detect:
            for name in ('a', 'b', 'c'):
                self.app.label.set(name)
                self.app.auto_label()
                self.app.auto_label()  # Duplicate enqueue must not consume another request.
                if name != 'c':
                    self.app.navigate(1)
            self.wait_until(lambda: len(started) == 2)
            self.root.after(1, self.app.cancel_auto_label)
            self.wait_for_detection()
            self.assertEqual(detect.call_count, 2)
        self.assertNotIn('c', started)
        self.assertTrue(all(job['state'] == 'cancelled' for job in self.app.auto_jobs.values()))
        self.assertTrue(all(len(record['boxes']) == 1 for record in self.app.data['images'].values()))

    def test_background_save_failure_keeps_result_and_ctrl_s_retries(self):
        started, release = Event(), Event()

        def slow_detect(*args):
            started.set()
            release.wait(5)
            return self.proposed

        with patch('app.detect_objects', side_effect=slow_detect):
            self.app.auto_label()
            try:
                self.assertTrue(started.wait(2))
                self.app.navigate(1)
                with patch('app.atomic_text', side_effect=OSError('disk full')), patch('app.messagebox.showerror'):
                    release.set()
                    self.wait_for_detection()
                    self.assertEqual(self.app.auto_jobs['a.png', 'mũ']['state'], 'save_error')
                    self.assertIn('a.png', self.app.unsaved)
                    self.app.navigate(1)
                    self.assertEqual(self.app.index, 1)
            finally:
                release.set()
                self.wait_for_detection()
        self.assertTrue(self.app.save())
        self.assertEqual(self.app.unsaved, set())
        self.assertEqual(self.app.auto_jobs['a.png', 'mũ']['state'], 'done')
        self.assertIn('2 0.250000', (self.dataset / 'labels/a.txt').read_text())

    def test_changed_image_is_not_auto_applied(self):
        started, release = Event(), Event()
        before = (self.dataset / 'labels/a.txt').read_bytes()

        def slow_detect(*args):
            started.set()
            release.wait(5)
            return self.proposed

        with patch('app.detect_objects', side_effect=slow_detect), patch('app.messagebox.showerror') as error:
            self.app.auto_label()
            try:
                self.assertTrue(started.wait(2))
                Image.new('RGB', (150, 50)).save(self.dataset / 'images/a.png')
            finally:
                release.set()
                self.wait_for_detection()
        error.assert_called_once()
        self.assertEqual(self.app.auto_jobs['a.png', 'mũ']['state'], 'error')
        self.assertEqual((self.dataset / 'labels/a.txt').read_bytes(), before)

    def test_manual_edits_while_running_are_preserved_by_auto_apply_and_undo(self):
        started, release = Event(), Event()

        def slow_detect(*args):
            started.set()
            release.wait(5)
            return self.proposed

        manual = {'label': 'cat', 'box': [20, 10, 40, 30]}
        with patch('app.detect_objects', side_effect=slow_detect):
            self.app.auto_label()
            try:
                self.assertTrue(started.wait(2))
                self.app.remember()
                self.app.boxes().append(manual)
                self.app.changed()
            finally:
                release.set()
                self.wait_for_detection()
        self.assertEqual(len(self.app.boxes()), 3)
        self.app.undo()
        self.assertIn(manual, self.app.boxes())
        self.assertNotIn(self.proposed[0], self.app.boxes())
        self.app.undo()
        self.assertEqual(len(self.app.boxes()), 1)

    def test_folder_switch_is_blocked_until_queue_finishes(self):
        release = Event()
        with patch('app.detect_objects', side_effect=lambda *args: (release.wait(5), [])[1]):
            self.app.auto_label()
            try:
                with patch('app.filedialog.askdirectory') as choose:
                    self.app.open_folder()
                    self.app.open_dataset()
                    choose.assert_not_called()
            finally:
                release.set()
                self.wait_for_detection()

    def test_close_cancels_workers_before_destroying_window(self):
        started = Event()

        def cancellable(image, label, cancel):
            started.set()
            if not cancel.wait(5):
                raise AssertionError('Close did not cancel the request')
            return self.proposed

        with patch('app.detect_objects', side_effect=cancellable), patch.object(self.root, 'destroy') as destroy:
            self.app.auto_label()
            self.assertTrue(started.wait(2))
            self.app.close()
            self.assertTrue(self.app.closing)
            self.wait_until(lambda: destroy.called)
            self.assertFalse(self.app.auto_busy())
            self.assertEqual(self.app.auto_jobs['a.png', 'mũ']['state'], 'cancelled')
            self.assertEqual(len(self.app.boxes()), 1)

    def test_ordinary_folder_persists_auto_labels_and_displayed_orientation(self):
        folder = self.dataset / 'ordinary'
        folder.mkdir()
        exif = Image.Exif()
        exif[274] = 6
        Image.new('RGB', (100, 50)).save(folder / 'rotated.jpg', exif=exif)
        with patch('app.filedialog.askdirectory', return_value=str(folder)):
            self.app.open_folder()
        self.assertEqual(self.app.image.size, (50, 100))
        with patch('app.detect_objects', return_value=self.proposed) as detect:
            self.app.auto_label()
            self.wait_for_detection()
        self.assertEqual(detect.call_args.args[0].size, (50, 100))
        saved = json.loads((folder / 'local-label.annotations.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['images']['rotated.jpg']['boxes'], self.proposed)
        with patch('app.filedialog.askdirectory', return_value=str(folder)):
            self.app.open_folder()
        self.assertEqual(self.app.boxes(), self.proposed)

    def test_controls_fit_minimum_window_size(self):
        self.app.auto_status.set('2 running · 100 queued\nrunning: red rectangle (12 boxes)')
        self.root.geometry('800x720')
        self.root.deiconify()
        self.root.update()
        sidebar = self.app.box_list.master
        for child in sidebar.winfo_children():
            self.assertTrue(child.winfo_ismapped(), str(child))
            self.assertLessEqual(child.winfo_y() + child.winfo_height(), sidebar.winfo_height())


if __name__ == '__main__':
    unittest.main()
