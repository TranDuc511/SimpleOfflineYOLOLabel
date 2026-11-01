# Local Label

A small Windows desktop app for drawing bounding boxes on images. Manual labeling works offline; optional auto-labeling uses Codex online.

Double-click **Start Labeler.bat**. The launcher uses `%USERPROFILE%\anaconda3\python.exe` when available, otherwise `python` from PATH. Requires Python with Tkinter and Pillow. On another computer, install Pillow into the Python used by the launcher: `python -m pip install -r requirements.txt` (use the full path to Anaconda Python if applicable).

1. Open a folder containing JPG, PNG, BMP, WebP or TIFF images.
2. Type or select a class name, then drag a box around an object.
3. Use Previous / Next or the arrow keys to move through images.
4. Select a box in the sidebar and choose Delete box to remove it. Ctrl+Z undoes box changes on the current image, including after navigating away and back during the same app session. Escape cancels drawing.

Labels save automatically in `local-label.annotations.json` inside the image folder. Reopen that folder to resume. Keep this file with your images. Originals are never edited. The folder must be writable. Images are shown fitted to the window; enlarge the window for finer boxes. To change a box, delete it and draw it again.

To edit an existing YOLO bounding-box dataset, click **Open YOLO dataset** and choose its root folder containing `images/` and `labels/`. Nested folders such as `images/train/` map to `labels/train/`. Put `classes.txt` in the dataset root (one class name per line, in class ID order), or select a class-name text file when prompted. For datasets with names in `data.yaml`, first put those names into `classes.txt` in the same order; YAML is not imported.

Only the image you select is opened, and its existing boxes appear immediately. Delete and redraw boxes as needed; changes and Undo save directly to that image's YOLO `.txt` file. Existing class IDs stay in order, and new classes append to the class-name file. Unedited label files and all image files stay untouched, so you do not need to visit thousands of images or export the dataset again. Missing label files show no boxes and are created only when edited. In this mode images use their stored pixel orientation to match YOLO coordinates. **Export YOLO** simply confirms that edits are already saved in place. Keep a backup if you need the original annotations, since this mode updates labels directly.

**Export YOLO** creates a new export subfolder with `classes.txt`, `images/`, and `labels/`. It includes only images you have viewed, including empty label files for viewed images without objects. PNG copies have EXIF orientation applied so exported coordinates match the exported images. Each label line is `class_id center_x center_y width height`, normalized to 0–1. Class IDs follow the order in `classes.txt`. Training configuration and train/validation splitting are not included.

A progress window shows the number of exported images and percentage complete. The interface stays responsive during export; wait for completion before continuing to label or closing the app.

Manual labeling works offline without an account or image upload.

## Auto-label with Codex

Install the official Codex CLI (`npm install -g @openai/codex` with Node.js installed), run `codex login`, and restart the labeler so it can find Codex on PATH. An existing working CLI sign-in is reused; no API key is stored by this app. Every request explicitly uses **`gpt-6-luna` with `low` reasoning (Luna Light)**.

1. Open an image and type the object name in **CLASS NAME**, for example `helmet`, `cat`, or `chai nước`.
2. Click **Auto-label** to queue that image and object name. You can move to another image immediately and queue it too. Two images run at a time; additional images wait in order. Moving between images does **not** cancel detection. Each queued job keeps the class name entered when it was queued.
3. Valid results **automatically append and save** to the correct image, even when you are viewing a different image. There is no Apply step. Existing boxes remain; class IDs keep their order. Requests send images online and use your account's available Codex usage, isolated from custom CLI configuration and connectors.
4. Watch the image list for `[queued]`, `[running]`, `[done]`, `[empty]`, `[error]`, or `[save_error]`. The status below Auto-label shows running/waiting counts and the current image's latest job. Completed results remain in the image's annotations. Repeated clicks for the same image/class reuse its job in this session; failed or cancelled jobs can be retried.
5. Return to any processed image to inspect the boxes. **Ctrl+Z** undoes an automatically applied batch and preserves edits made before that batch. Undo histories survive image navigation in the same app session. To adjust an individual box, delete and redraw it.

**Cancel queue** cancels running requests and clears waiting jobs; completed labels stay saved. Closing the app does the same and waits for the request processes to stop. Completed labels persist across app restarts; waiting/running jobs and Undo histories do not. Finish or cancel the queue before opening a different folder/dataset or exporting. Each running request times out after three minutes. Cancellation may still consume usage for work already sent to the service.

Failed requests and empty detections leave annotations unchanged. An image modified on disk after queuing is rejected rather than labeled using stale coordinates. If saving fails, results remain in memory, the row shows `[save_error]`, and navigation/closing is blocked by the save failure until you retry with **Ctrl+S**. A successful retry saves the retained result without another model request.

The app sends a temporary PNG in the exact orientation shown on screen, without the original filename or EXIF metadata. Images larger than 2048 pixels on either side are proportionally resized for the request; returned coordinates map back to the full image. Temporary request files are deleted afterwards. Codex runs in a read-only sandbox with shell tools disabled; the app validates coordinates and writes accepted annotations itself.

Auto-label handles one object class and one image per request. Queued images are decoded only when a worker is available. Vision models may miss small objects, produce loose boxes, or identify the wrong object. Results now save automatically: inspect the saved boxes and evaluate accuracy on your own sample images before using the labels for training.

If detection fails, check `codex login status`, your internet connection, account limits, and CLI version. No labels are changed by a failed detection. Run the checks with `python -m unittest discover -v` using the same Python as the launcher.
