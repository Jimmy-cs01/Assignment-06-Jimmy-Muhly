"""
AI Personal Assistant - Terminal Version
Uses Google Gemini API directly via HTTP requests (no SDK needed)

Requirements:
    pip install requests pillow

Usage:
    python assistant.py

Commands inside the app:
    image <path>   - Attach an image for the assistant to analyze
    generate       - Ask the assistant to generate an image (say "generate an image of...")
    exit / quit    - Exit the program
"""

import os
import sys
import json
import base64
import re
import requests
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = "gemini-2.5-flash"
BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}"
IMAGEN_URL = "https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict"

# ── Tool Definitions (5 functions) ────────────────────────────────────────────
TOOLS = [
    {
        "functionDeclarations": [
            {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name, e.g. Athens, OH"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"},
                    },
                    "required": ["city"],
                },
            },
            {
                "name": "schedule_meeting",
                "description": "Schedule a meeting with attendees on a date and time.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "attendees": {"type": "array", "items": {"type": "string"}, "description": "List of attendee names"},
                        "date": {"type": "string", "description": "Date, e.g. 2025-04-01"},
                        "time": {"type": "string", "description": "Time, e.g. 10:00 AM"},
                        "topic": {"type": "string", "description": "Meeting topic"},
                    },
                    "required": ["attendees", "date", "time", "topic"],
                },
            },
            {
                "name": "set_reminder",
                "description": "Set a reminder for the user at a specific date/time.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Reminder message"},
                        "datetime": {"type": "string", "description": "ISO datetime, e.g. 2025-04-01T09:00"},
                    },
                    "required": ["message", "datetime"],
                },
            },
            {
                "name": "calculate",
                "description": "Evaluate a math expression and return the result.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression, e.g. (12 * 4) / 3 + 7"},
                    },
                    "required": ["expression"],
                },
            },
            {
                "name": "search_web",
                "description": "Search the web for a query and return a summary of results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        ]
    }
]

SYSTEM_PROMPT = """You are a helpful personal assistant running in the terminal.
You have access to tools: weather lookup, meeting scheduling, reminders, calculator, and web search.
When the user asks you to generate/create/draw an image, include exactly [IMAGE_GEN: <prompt>] in your response.
Be concise and friendly."""

# ── Mock Tool Executors ───────────────────────────────────────────────────────
import random

def execute_tool(name, args):
    if name == "get_weather":
        temp = random.randint(55, 90)
        condition = random.choice(["Sunny", "Partly Cloudy", "Overcast", "Light Rain"])
        unit = args.get("unit", "fahrenheit")
        return {
            "city": args["city"],
            "temperature": temp,
            "unit": unit,
            "condition": condition,
            "humidity": f"{random.randint(40, 80)}%",
        }

    elif name == "schedule_meeting":
        return {
            "status": "scheduled",
            "meeting_id": f"MTG-{random.randint(1000,9999)}",
            "attendees": args["attendees"],
            "date": args["date"],
            "time": args["time"],
            "topic": args["topic"],
        }

    elif name == "set_reminder":
        return {
            "status": "set",
            "reminder_id": f"REM-{random.randint(1000,9999)}",
            "message": args["message"],
            "datetime": args["datetime"],
        }

    elif name == "calculate":
        try:
            result = eval(args["expression"], {"__builtins__": {}}, {})
            return {"expression": args["expression"], "result": result}
        except Exception as e:
            return {"error": str(e)}

    elif name == "search_web":
        return {
            "query": args["query"],
            "results": [
                {"title": f"Top result for '{args['query']}'", "snippet": "Simulated result with relevant info.", "url": "https://example.com/1"},
                {"title": f"More about '{args['query']}'", "snippet": "Another simulated result with additional details.", "url": "https://example.com/2"},
            ],
        }

    return {"error": f"Unknown tool: {name}"}


# ── Gemini API Call ───────────────────────────────────────────────────────────
def call_gemini(contents, tools=None):
    body = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2048},
    }
    if tools:
        body["tools"] = tools

    resp = requests.post(
        f"{BASE_URL}:generateContent?key={API_KEY}",
        headers={"Content-Type": "application/json"},
        json=body,
        timeout=30,
    )
    if not resp.ok:
        err = resp.json()
        raise RuntimeError(err.get("error", {}).get("message", "API error"))
    return resp.json()


# ── Image Generation ──────────────────────────────────────────────────────────
def generate_image(prompt, output_path="generated_image.png"):
    print(f"\n  Generating image: '{prompt}' ...")
    resp = requests.post(
        f"{IMAGEN_URL}?key={API_KEY}",
        headers={"Content-Type": "application/json"},
        json={"instances": [{"prompt": prompt}], "parameters": {"sampleCount": 1}},
        timeout=60,
    )
    if not resp.ok:
        err = resp.json()
        raise RuntimeError(err.get("error", {}).get("message", "Image generation failed"))

    data = resp.json()
    b64 = data.get("predictions", [{}])[0].get("bytesBase64Encoded")
    if not b64:
        raise RuntimeError("No image data returned")

    img_bytes = base64.b64decode(b64)
    with open(output_path, "wb") as f:
        f.write(img_bytes)
    return output_path


# ── Agent Loop ────────────────────────────────────────────────────────────────
def agent_loop(contents):
    for _ in range(5):
        response = call_gemini(contents, tools=TOOLS)
        candidate = response["candidates"][0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        # Check for function calls
        fn_calls = [p for p in parts if "functionCall" in p]
        if fn_calls:
            contents.append({"role": "model", "parts": parts})
            response_parts = []
            for p in fn_calls:
                name = p["functionCall"]["name"]
                args = p["functionCall"]["args"]
                result = execute_tool(name, args)
                response_parts.append({
                    "functionResponse": {"name": name, "response": {"result": result}}
                })
            contents.append({"role": "user", "parts": response_parts})
            continue

        # Text response
        text = "".join(p.get("text", "") for p in parts)

        # Check for image generation trigger
        img_match = re.search(r"\[IMAGE_GEN:\s*(.+?)\]", text)
        if img_match:
            img_prompt = img_match.group(1).strip()
            text = re.sub(r"\[IMAGE_GEN:\s*.+?\]", "", text).strip()
            try:
                path = generate_image(img_prompt)
                text += f"\n\n  Image saved to: {os.path.abspath(path)}"
            except Exception as e:
                text += f"\n\n  Image generation failed: {e}"

        return text

    return "Sorry, I couldn't complete that request."


# ── Image Loading Helper ───────────────────────────────────────────────────────
def load_image_part(path):
    ext = path.rsplit(".", 1)[-1].lower()
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif", "webp": "image/webp"}
    mime = mime_map.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return {"inlineData": {"mimeType": mime, "data": b64}}


# ── Main REPL ─────────────────────────────────────────────────────────────────
def main():
    global API_KEY

    print("=" * 55)
    print("  AI Personal Assistant  (Gemini-powered)")
    print("=" * 55)
    print("  Commands:")
    print("    image <path>  — attach an image to your next message")
    print("    exit / quit   — exit")
    print("-" * 55)

    # Get API key
    if not API_KEY:
        API_KEY = input("\n  Enter your Gemini API key: ").strip()
        if not API_KEY:
            print("  No API key provided. Exiting.")
            sys.exit(1)

    print("\n  Ready! Ask me anything.\n")

    conversation = []  # full history
    pending_image = None  # image to attach on next message

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("\n  Goodbye!")
            break

        # image attach command
        if user_input.lower().startswith("image "):
            path = user_input[6:].strip()
            if not os.path.exists(path):
                print(f"  File not found: {path}\n")
                continue
            pending_image = path
            print(f"  Image attached: {path}")
            print("  Now type your question about the image.\n")
            continue

        # Build message parts
        parts = []
        if pending_image:
            try:
                parts.append(load_image_part(pending_image))
                print(f"  Sending image: {pending_image}")
            except Exception as e:
                print(f"  Could not load image: {e}\n")
            pending_image = None
        parts.append({"text": user_input})

        conversation.append({"role": "user", "parts": parts})

        print("\nAssistant: ", end="", flush=True)
        try:
            reply = agent_loop(list(conversation))
            print(reply)
            conversation.append({"role": "model", "parts": [{"text": reply}]})
        except Exception as e:
            print(f"\n  Error: {e}")

        print()


if __name__ == "__main__":
    main()