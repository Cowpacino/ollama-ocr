import requests
import base64
import sys
import os


def extract_text_from_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return None

    # Read and encode image
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

    # Prepare request to Ollama
    url = "http://localhost:11434/api/generate"
    prompt = """
                You are a highly accurate OCR assistant. Your goal is to extract **all** visible text from the supplied image—printed, handwritten, or stylized—without adding any commentary, formatting, or markup.  
                • Output only the raw text, preserving line breaks and paragraph breaks as they appear in the image.  
                • Do not include any bounding box data or JSON—just plaintext.  
                • Do not label or annotate anything.  
                • If you encounter illegible regions, skip them silently.  
                Begin now:
"""

    payload = {
        "model": "qwen2.5vl:3b",
        "prompt": f"{prompt}",
        "images": [image_data],
        "stream": False,
        "options": {"temperature": 0.1, "top_p": 0.1},
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()

        result = response.json()
        return result.get("response", "")

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama: {e}")
        return None


def save_to_file(text, filename="output.md"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Text extracted and saved to {filename}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    extracted_text = extract_text_from_image(image_path)

    if extracted_text:
        save_to_file(extracted_text)
    else:
        print("Failed to extract text from image.")
        sys.exit(1)


if __name__ == "__main__":
    main()
