from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import numpy as np
from PIL import Image
import tensorflow as tf
import time
import os
import re
import json
import uuid
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# Gemini setup
genai.configure(api_key="")  # Replace with your actual API key
gemini_model = genai.GenerativeModel(model_name="gemini-2.0-flash")
















# Helper function to extract JSON values from Gemini's response text
def extract_json_values(response_text):
    try:
        json_matches = re.findall(r'```json\n({.*?})\n```', response_text, re.DOTALL)
        if not json_matches:
            return []

        json_data = json.loads(json_matches[0])
        values = list(json_data.values())
        return values[:9]

    except json.JSONDecodeError:
        return []



















from tensorflow.keras.models import load_model
import os

def analyze_with_gemini(image_path):
    try:
        with Image.open(image_path).convert("RGB") as img:

            prompt1 = """Analyze the given handwritten text in MALAY Language based on the following features. For each feature, return 1 if the issue is present and 0 if not. Provide explanations where necessary.

    Text Input (OCR Output): [Provide extracted text].   Display the ocr text too.

    Format should be exactly like this -
    OCR Output:Text

    Based on this above text, evaluate the below features.

    1. **Atypical Margin Usage:** Detect if words extend significantly beyond the expected right margin. Minimal extensions should be ignored. Return 1 if major extensions are present with a short reason; otherwise, return 0.
    2. **Letter Inversions:** Identify letters written as their horizontally flipped counterparts (e.g., 'b' as 'd'). If inversions lead to incorrect words, return 1 with the affected letter or word; otherwise, return 0. Ignore other mistakes.
    3. **Abandoned Words:** Identify words that were started but not completed. If incomplete words exist, return 1; otherwise, return 0.
    4. **Letter Transpositions:** Detect cases where letters within a word are swapped, leading to incorrect or unintended words. If a misspelled word can be rearranged to form a correct word, it indicates a transposition. Return 1 with a brief explanation if transpositions are found; otherwise, return 0.
    5. **Poor Legibility:** Detect illegible handwriting, fragmented words, or recognition errors. Provide OCR confidence scores with extracted text. If illegibility is significant, return 1; otherwise, return 0.
    6. **Spelling Errors:** Identify misspelled words, phonetic errors, or missing letters. If mistakes are found, return 1 with a brief reason; otherwise, return 0.  if there is spelling errors, automatically check for transpositions too, (refer ocr text also)

    IMPORTANT : If the reason for spelling errors is due to incomplete words, then put abandoned words as 1, and spelling errors as 0


    Return format:
    {
      "Atypical_Margin_Usage": 0 or 1,
      "Letter_Inversions": 0 or 1,
      "Letter_Transpositions": 0 or 1,
      "Spelling_Errors": 0 or 1,
      "Poor_Legibility": 0 or 1,
      "Abandoned_Words": 0 or 1,
    }

    provide reasons too FOR EACH FEATURES
""" 
            prompt2 = """Analyze the given handwritten text in MALAY Language based on the following features. For each feature, return 1 if the issue is present and 0 if not. Provide explanations where necessary.


    Based on the provided text, evaluate the below features.


    IMPORTANT: Refer the provided text only and evaluate the following


    1. **Letter Reversals:** Identify letters that appear as vertically mirrored versions of their correct form. If reversals change word meanings or deviate from typical handwriting conventions, return 1 with the corrected word and its meaning; otherwise, return 0.
    2. **Incorrect Capitalization:** Detect capitalization mistakes based on the following rules:
    - First word of a sentence SHOULD BE CAPITALIZED
    - Proper nouns (people, places, brands, etc.) SHOULD BE CAPITALIZED
    - Days, months, and holidays SHOULD BE CAPITALIZED
    - The pronoun "I" SHOULD BE CAPITALIZED
    - Titles of books, movies, and songs (Title Case) SHOULD BE CAPITALIZED
    - Acronyms and initialisms SHOULD BE CAPITALIZED
    - First word in a quotation (if it’s a full sentence) SHOULD BE CAPITALIZED
    - Words like 'emak' should be treated as common nouns
    If incorrect capitalizations are found, return 1 with a short reason; otherwise, return 0.    Unique styles or writers styles are not tolerated. Any adjective or adverb or verb shouldnt be capitalized


    Refer the provided text very clearly and evaluate whether there is absence of space between words---


    3. **Letter or Word Crowding:** Detect cases where:
        - Spaces between words are missing, causing multiple words to merge into one.
        If crowded words without spaces are detected, return 1 with a brief reason; otherwise, return 0, And explain meaning of each word too.
    Malay words are typically separate unless they form an accepted compound word.
    ✅ "Baju itu" (Correct)
    ❌ "Bajuitu" (Incorrect, space missing)
    Some words in Malay are naturally combined, but "Baju itu" is not one of them.
    ✅ "Kerjasama" (Correct, valid compound word)
    ❌ "Bajuitu" (Incorrect, not a valid word)
    In Malay, 'Bajuitu' is incorrect because 'baju' (shirt) and 'itu' (that) must be written separately. Unlike languages that frequently combine words, Malay typically maintains clear word boundaries. The correct form is 'Baju itu' with a space.

    Return format:
    {

      "Letter_Reversals": 0 or 1,
      "Incorrect_Capitalization": 0 or 1,
      "Letter_or_Word_Crowding": 0 or 1
    }

    IMPORTANT: DO NOT INCLUDE OCR OUTPUT TEXT IN THE JSON RETURN FORMAT. ONLY INCLUDE THOSE 3 FEATURES.

    provide reasons too FOR EACH FEATURES""" 
            
            response1 = gemini_model.generate_content([img, prompt1])
            print("Response 1 (Prompt 1) Output:\n", response1.text)  # Debug print

            match = re.search(r"OCR Output:\s*(.+)", response1.text)
            ocr_text = match.group(1) if match else ""

            response2 = gemini_model.generate_content([ocr_text, prompt2])
            print("Response 2 (Prompt 2) Output:\n", response2.text)  # Debug print

            combined_prompt = f"""
    Below are two responses from a handwriting analysis system. Merge them into a single clear, detailed, and human-friendly explanation, AVOID THE USE OF BOLD or "**". Avoid JSON or code format. Just give a clean summary.
    
    Response 1:
    {response1.text}


    Response 2:
    {response2.text}.
    """
            final_response = gemini_model.generate_content(combined_prompt)
            print("Final Response Output:\n", final_response.text)  # Debug print


            numbers1 = extract_json_values(response1.text)
            numbers2 = extract_json_values(response2.text)

            if not numbers1 or not numbers2:
                return None, final_response.text

            all_data = numbers1 + numbers2
            all_data = [all_data]

           

            base_dir = os.path.dirname(__file__)
            model_path = os.path.join(base_dir, "ann.h5")
            model = load_model(model_path)


            X = np.array(all_data)
            predictions = model.predict(X)

            
            integer_prediction = predictions[0][0]
        

  



            return all_data, final_response.text, integer_prediction
        
    except Exception as e:
        print("Gemini Error:", e)
        return None, "", ""



















import tempfile

def save_base64_image(image_data):
    image_bytes = base64.b64decode(image_data)
    temp_dir = tempfile.gettempdir()
    filename = os.path.join(temp_dir, f"{uuid.uuid4().hex}.jpg")
    with open(filename, "wb") as f:
        f.write(image_bytes)
    print(f"Image saved at: {filename}")  # Debug log
    return filename







def convert_to_python_types(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.int32, np.int64)):
        return int(data)
    elif isinstance(data, list):
        return [convert_to_python_types(item) for item in data]
    else:
        return data









# Prediction + Gemini API
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = data.get('image')

        image_path = save_base64_image(image_data)
        time.sleep(1)

        gemini_features, finalresponse_text, prediction_final = analyze_with_gemini(image_path)
        os.remove(image_path)

   


        if gemini_features is None:
            return jsonify({'error': 'Gemini analysis failed', 
                            'response': finalresponse_text}), 500

        prediction_final_py = convert_to_python_types(prediction_final)



        feature_labels = [
            "Atypical_Margin_Usage", "Letter_Inversions", "Letter_Transpositions",
            "Spelling_Errors", "Poor_Legibility", "Abandoned_Words",
            "Letter_Reversals", "Incorrect_Capitalization", "Letter_or_Word_Crowding"
        ]
        result = dict(zip(feature_labels, gemini_features))





        return jsonify({
                        'all_data': gemini_features,
                        "explanation": finalresponse_text,
                        "predictions": prediction_final_py})
        

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
