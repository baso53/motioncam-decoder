# MotionCam MCRAW mounter with a FUSE Filesystem

A FUSE-based virtual filesystem that mounts all `.mcraw` files in the current directory, exposing each frame as a DNG and the audio as `audio.wav`.

---

## Demo

[![Watch the video](https://img.youtube.com/vi/1JCld9Bp7mw/0.jpg)](https://www.youtube.com/watch?v=1JCld9Bp7mw)

---

## Downloads

[Releases page](https://github.com/baso53/mcraw-mounter/releases)  

---

## Prerequisites to run

- Dependencies:
  - **macFUSE** (macOS)  
    - macOS: Download and install from [macFUSE](https://osxfuse.github.io/)  

---

## Prerequisites for building

- A C++17-capable compiler (e.g. `g++`, `clang++`)  
- CMake ≥ 3.10
- Dependencies:
  - **macFUSE** (macOS)  
    - macOS: Download and install from [macFUSE](https://osxfuse.github.io/)  
  - [nlohmann/json](https://github.com/nlohmann/json)  
  - [tinydngwriter](https://github.com/syoyo/tinydng)  
  - [audiofile](https://github.com/adamstark/AudioFile) (for WAV output)  

---

## Building

1. Clone the repo:
   ```bash
   git clone https://github.com/baso53/mcraw-mounter
   cd mcraw-mounter
   ```
2. Create and enter a build directory:
   ```bash
   mkdir build && cd build
   ```
3. Configure and build:
   ```bash
   cmake ..
   make
   ```
   This produces one binary:
   - `mcraw-mounter-fuse`

---

## FUSE-Based Virtual Filesystem

Mounts a directory `mcraws` in your working folder. After mounting you’ll see:

```
mcraws/
├── 007-VIDEO_24mm-240328_141729.0/
│   ├── frame_000000.dng
│   ├── frame_000001.dng
│   └── audio.wav
└── another-recording/ …
```

### How It Works

1. On startup, `mcraw-mounter-fuse` scans the folder where the application is run from for all `.mcraw` files.  
2. For each file it creates an in-memory context:
   - Loads container metadata (black/white levels, CFA pattern, matrices, orientation)  
   - Builds a list of frame timestamps  
   - Decodes & caches the frames for reading by the kernel
   - Extracts and converts all audio chunks into an in-memory WAV buffer  
3. `ls mcraws/<basename>/` lists `frame_*.dng` and `audio.wav`.  
4. Opening a frame-file decodes (or retrieves from cache) the RAW frame as a valid DNG. 
5. Reading `audio.wav` serves the constructed WAV data.

### Usage

```bash
./mcraw-mounter-fuse
# DEBUG: [007-VIDEO_...0.mcraw] found N frames
# Mounts at ./mcraws
```

or

- Copy the `mcraw-mounter` into the folder with your `.mcraw` files and run it from there.

When done:
```bash
umount ./mcraws
```

or

- unmount from Finder.


---

## Tips

Don't open the mounter folders in Finder if you don't have to, since MacOS will start thumbnail generation process, which will take a lot of RAM.

---

## Sample File

Download a sample `.mcraw`:

[https://storage.googleapis.com/motioncamapp.com/samples/007-VIDEO_24mm-240328_141729.0.mcraw](https://storage.googleapis.com/motioncamapp.com/samples/007-VIDEO_24mm-240328_141729.0.mcraw)

---

## MotionCam Pro

MotionCam Pro is an Android app for capturing RAW video.  
Get it on the [Play Store](https://play.google.com/store/apps/details?id=com.motioncam.pro&hl=en&gl=US).
