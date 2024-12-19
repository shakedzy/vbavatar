# Vision-Based Agent

![robot](robot.png)

> [!IMPORTANT]  
> **This repository is meant to be used for educational purposes only!**

This repository contains a demo of an automatic LLM/VLM-based agent which visits Google News (specifically, the Technology > AI page),
scrolls through it and look for interesting articles. It then clicks on the links it selected, and extracts the full article as plain text from
the opened page. It then returns to Google News, scrolls down and continues this routine.

The agent uses only small local models for this:
* Quantized `llama-3.2-vision` (11b, via Ollama)
* Quantized `llama-3.1` (3b, via Ollama)
* `Florence-2-base`


## Installation
1. Install [Ollama](https://ollama.com/) and pull the required models:
```bash
ollama pull llama-3.1
ollama pull llama-3.2-vision
```
2. Create a new virtual environment (_recommended_) and clone this repo to it.
3. From the root repo of this repo, install using:
```bash
pip install -e .
playwright install
```

## Running
Runt he agent using the `vba` command:
```
usage: vba [-h] [-s SCROLLS] [-o OUTPUT_FILE] [--debug]

options:
  -h, --help            show this help message and exit
  -s SCROLLS, --scrolls SCROLLS
                        Number of mouse-scrolls to perform (non-negative integer, default = 1)
  -o OUTPUT_FILE, --output-file OUTPUT_FILE
                        Output filename (defaults to `output_[run-time].json`)
  --debug               Turn on debug mode
```