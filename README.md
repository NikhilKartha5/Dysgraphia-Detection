
# Dysgraphia Detection Web Platform
# Reference


## Publication

This project is published as:
- [Dysgraphia Detection Using Deep Learning Techniques](https://ieeexplore.ieee.org/document/11089394) (IEEE Xplore)

# Project Summary

**Dysgraphia Detection Web App | React, Flask, TensorFlow, PyTorch, Gemini**  
*Jan 2025– Mar 2025*

- Built a full-stack web app allowing users to upload handwriting samples for dysgraphia diagnosis and receive clear, structured reports; model trained on 249 images and tested on 50+ real entries.
- Developed a CNN-BiLSTM model with 91% accuracy on handwriting classification, expanded dataset with 900+ GAN samples, and integrated LLM to detect dysgraphia symptoms with 30% improved interpretability.

An advanced web application for automated dysgraphia screening using deep learning and generative AI. Built with Next.js, TypeScript, Tailwind CSS, and a Python Flask API.

## Features
- Upload handwriting samples and receive instant dysgraphia risk analysis
- ML-powered backend (Keras/TensorFlow) for classification
- Google Gemini AI for feature extraction and natural language feedback
- Secure authentication and user management
- Modern, responsive UI

## Screenshots

### Home Page
![Home Page](pics/HOME.png)

### Testing for Potential Dysgraphia
![Testing for Potential Dysgraphia](pics/TEST%20POTENTIAL.png)

### Testing for Low Potential Dysgraphia
![Testing for Low Potential Dysgraphia](pics/TEST%20LOW%20POTENTIAL.png)

### Results of Various Models Trained to Classify
![Results of Models](pics/RESULTS.png)

### Proposed Architecture of the Model
![Proposed Architecture](pics/PROPOSED%20ARCHITECURE.png)

## Getting Started

Clone the repo and install dependencies:
```bash
git clone https://github.com/NikhilKartha5/Dysgraphia-Detection.git
cd Dysgraphia-Detection
npm install
```

Start the development server:
```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## API
The Flask backend exposes `/api/predict` for ML-powered handwriting analysis. See `api/app.py` for details.

## Tech Stack
- Next.js, React, TypeScript, Tailwind CSS
- Python, Flask, TensorFlow, Google Gemini AI

## License
Released under the MIT License. See LICENSE for details.

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.



