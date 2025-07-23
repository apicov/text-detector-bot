# text-detector-bot

A Telegram bot and Flask API for detecting and recognizing text in images using PaddleOCR and TrOCR.

## Features

- Detects text regions in images using PaddleOCR.
- Recognizes text in detected regions using TrOCR (transformer-based OCR).
- REST API for programmatic access.
- Telegram bot for easy image-to-text interaction.

---

## Requirements

- Python 3.7+
- pip

### Python Dependencies

Install all required packages:

```bash
pip install paddleocr opencv-python-headless numpy pillow matplotlib flask python-telegram-bot transformers torch
```

> **Note:**  
> - `paddleocr` may require additional system dependencies (see [PaddleOCR installation guide](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/installation_en.md)).
> - `torch` version should match your CUDA version if using GPU.

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd text-detector-bot
```

### 2. Configure Telegram Bot

Create a file named `private_data.json` in the project root with the following content:

```json
{
  "telegram_token": "YOUR_TELEGRAM_BOT_TOKEN"
}
```

Get your bot token from [BotFather](https://core.telegram.org/bots#botfather) on Telegram.

---

## Usage

### 1. Start the Flask API Server

```bash
python flask_server.py
```

- The server will run on `http://127.0.0.1:5000`.
- Main endpoint: `POST /api/process_image/`
    - Request JSON: `{ "image": "<base64-encoded-image>" }`
    - Response JSON: `{ "bboxes": [...], "binary_image": "<base64>", "text": ["..."] }`

### 2. Start the Telegram Bot

```bash
python telegram_bot.py
```

- The bot will listen for image messages.
- When you send an image, it will reply with:
    - The image with detected text regions highlighted.
    - The binarized (preprocessed) image.
    - The recognized text.

---

## How it Works

- **Detection:** PaddleOCR detects text regions in the image.
- **Recognition:** Each detected region is cropped and passed to TrOCR for text recognition.
- **API:** The Flask server exposes an endpoint for image processing.
- **Telegram Bot:** Forwards images to the API and returns results to the user.

---

## File Structure

- `text_detector.py` — PaddleOCR-based text detection.
- `trocr.py` — TrOCR-based text recognition.
- `flask_server.py` — REST API server.
- `telegram_bot.py` — Telegram bot logic.
- `utils.py` — Image conversion utilities.

---

## Notes

- The Telegram bot expects the Flask server to be running locally at `http://127.0.0.1:5000`.
- For production, consider deploying the Flask server and bot separately, and update the API URL in `telegram_bot.py` if needed.
- For GPU acceleration, ensure CUDA is installed and `torch` is installed with CUDA support.

---

## Acknowledgements

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [TrOCR (Microsoft)](https://huggingface.co/microsoft/trocr-base-handwritten)
- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
