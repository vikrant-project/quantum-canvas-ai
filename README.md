# 🎨 Quantum Canvas V1 Ultimate — AI Image Generator

> A powerful, browser-based AI Image Generation Studio built with Python & Flask.  
> Supports Text-to-Image, Image-to-Image, Background Removal, Upscaling, Denoising, and more — all running locally on CPU.

---

## 🖼️ Project Preview

```
[ Text2Img ] [ Img2Img ] [ Upscale ] [ Remove BG ] [ Denoise ] [ Restore ]
      ↓
  Write Prompt → Pick AI Model → Adjust Steps → Generate → Download
```

---

## ✨ Features

- 🤖 **50+ AI Models** — Speed, Realistic, Anime, Artistic, Sci-Fi & more
- 🖼️ **Text to Image** — Generate images from text prompts
- 🔄 **Image to Image** — Transform existing images with AI
- 🪄 **Background Removal** — One-click BG remover using `rembg`
- 📐 **Super Resolution** — Upscale images 2x, 4x, 8x
- 🔇 **Denoising** — Remove noise from photos
- 🕰️ **Image Restoration** — Restore old or damaged images
- ⚡ **Speed Presets** — Lightning Fast → Ultra Quality
- 🌗 **Dark / Light Theme** — Toggle with `Ctrl+T`
- 📜 **Generation History** — Persistent history with preview
- ⭐ **Favorites** — Star your favourite AI models
- 🔍 **Model Search** — Search 50+ models instantly
- ⌨️ **Keyboard Shortcuts** — Full keyboard control (press `?` to see)
- 📱 **Mobile Responsive** — Works on phone & tablet too
- 🧠 **CPU Optimized** — No GPU required, runs on any machine

---

## 🗂️ Project Structure

```
quantum-canvas/
│
├── app.py              ← Main application (Flask server + AI logic)
├── requirements.txt    ← Python dependencies
├── README.md           ← You are here
├── .gitignore          ← Git ignore rules
│
├── history.json        ← Auto-created: generation history
└── favorites.json      ← Auto-created: saved favourite models
```

---

## ⚙️ Requirements

| Tool | Version |
|------|---------|
| Python | 3.9 or higher |
| pip | Latest |
| RAM | Minimum 4GB (8GB recommended) |
| Storage | 3–5 GB free (for AI model downloads) |
| GPU | ❌ Not required — CPU works fine |

---

## 🚀 How to Run (Step by Step)

### Step 1 — Clone the Repository

```bash
# Using Git
git clone https://soulcrack-spoofs-admin@bitbucket.org/soulcrack-spoofs/quantum-canvas-ai.git

cd canvas-ai
```

---

### Step 2 — Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⏳ First time setup may take 5–10 minutes depending on your internet speed.  
> The app will also auto-install any missing packages when you first run it.

---

### Step 4 — Run the App

```bash
python app.py
```

---

### Step 5 — Open in Browser

```
http://localhost:1600
```

That's it! 🎉 The app will automatically download the default AI model (`tiny-sd`) on first launch.

---

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl + Enter` | Generate image |
| `Ctrl + D` | Download image |
| `Ctrl + N` | New / Reset canvas |
| `Ctrl + F` | Search models |
| `Ctrl + T` | Toggle Dark/Light theme |
| `1` – `6` | Switch between modules |
| `Alt + ↑ / ↓` | Adjust steps by 5 |
| `Esc` | Reset canvas |
| `?` | Show all shortcuts |

---

## 🧠 AI Models Included

| Category | Examples |
|----------|---------|
| ⚡ Speed | Tiny SD, SD Turbo, LCM SD |
| 🎨 General | Stable Diffusion v1.5, v2.1, DreamShaper |
| 📷 Realistic | Realistic Vision, Dreamlike Photoreal |
| 🌸 Anime | Anything V5, Counterfeit |
| 🖌️ Artistic | OpenJourney, Van Gogh Diffusion |
| 🎭 Special | Ghibli Diffusion, Sci-Fi Diffusion, Modern Disney |

---

## ❓ Common Issues

**Q: App is slow on first run?**  
A: It downloads the AI model on first use (~300MB–5GB). Wait for it to complete.

**Q: `ModuleNotFoundError`?**  
A: Run `pip install -r requirements.txt` again inside your virtual environment.

**Q: Port 1600 already in use?**  
A: Change the last line in `app.py`: `app.run(port=5000)` to any free port.

**Q: Image generation is taking too long?**  
A: Use the ⚡ **Lightning Fast** preset or select the **Tiny SD** model.

---

## 👨‍💻 About the Developer

**Name:** [soulcracks_owner


This project was built to demonstrate practical skills in:
- Python web development (Flask)
- AI/ML model integration (HuggingFace Diffusers)
- Computer Vision (OpenCV, rembg, scikit-image)
- Modern UI/UX design (Glassmorphism, Dark theme)
- REST API design

---

## 📄 License

This project is open-source and free to use for educational purposes.

---

> ⭐ If you liked this project, please give it a star on Bitbucket!
