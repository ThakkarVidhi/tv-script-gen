# <h1 align="center">SceneScribe: AI-Powered TV Script Generation</h1>

**A deep learning-based script generation tool trained on Seinfeld dialogues to create realistic TV show scripts.**

![Build Status](https://img.shields.io/badge/build-passing-brightgreen) ![License](https://img.shields.io/badge/license-MIT-blue)

---

## Project Description

**SceneScribe** is an AI-driven script generation project that utilizes **Recurrent Neural Networks (RNNs)** trained on Seinfeld scripts. The model learns the structure, style, and character interactions from real episodes and generates original dialogues that maintain the comedic essence of the show. The project is designed for AI enthusiasts, scriptwriters, and researchers interested in natural language generation (NLG).

---

## Table of Contents
- [Project Title and Overview](#project-title-and-overview)
- [Project Description](#project-description)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Demo and Preview](#demo-and-preview)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Testing](#testing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [FAQs](#faqs)
- [Contact Information](#contact-information)

---

## Features

- **Seinfeld Script Generation** – Trained on real Seinfeld scripts to generate dialogue that mimics the show's unique humor and style.
- **Deep Learning-Powered** – Utilizes a Recurrent Neural Network (RNN) built with PyTorch for text generation.
- **Customizable Output** – Users can specify parameters like text length and temperature to control randomness and creativity.
- **Pretrained and Fine-tuned Models** – Offers pretrained models for quick usage and supports fine-tuning on custom datasets.
- **Tokenization and Preprocessing** – Automatically cleans and processes script data for better training and performance.
- **Interactive Generation** – Users can generate new script lines and iterate over outputs to refine results.

---

## Technology Stack

**Machine Learning Framework**:
- PyTorch (for deep learning and RNN implementation)
- CUDA (for GPU-accelerated training)

**Data Processing**:
- NumPy, Pandas (for dataset preprocessing)
- NLTK (for text tokenization)

**Model Training & Evaluation**:
- RNN, LSTM architectures (for text generation)
- Google Colab (for training execution)
- CUDA-enabled GPUs for faster training

**Other Tools**:
- Jupyter Notebook (for experimentation and model development)

---

## Demo and Preview

Here’s a preview of a sample AI-generated Seinfeld dialogue:

```text
Jerry: What's the deal with airplane food?
George: You know, I asked that once... and I still don’t know!
Elaine: Maybe it’s the tiny forks. They make everything taste worse.
Kramer: You should try eating peanuts with chopsticks, Jerry. It’s a game-changer!
```

For a more detailed output, check the `generated_script.txt` file, which contains a full AI-generated scene. 

---

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/ThakkarVidhi/tv_script_gen.git
    cd tv_script_gen
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Ensure you have CUDA installed for GPU acceleration:
    ```bash
    nvcc --version  # Check CUDA installation
    ```

5. Run the Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

---

## Usage

All script generation processes are handled within the provided **Jupyter Notebook** (`tv_script_generator.ipynb`). Follow these steps to generate TV scripts:  

1. **Open the Notebook**  
    Launch Jupyter Notebook and open `tv_script_generator.ipynb`:

2. **Train the Model (Optional)**  
    If you want to train the model from scratch instead of using pre-trained weights, run the training cells in the notebook. This will process the Seinfeld dataset and train an RNN model using PyTorch **with CUDA acceleration** if available.

3. **Generate a Seinfeld Dialogue**  
    Once the model is trained or pre-trained weights are loaded, execute the generation cells to create new TV script dialogues. Modify the prompt to start the conversation.
    Example usage:
    ```bash
    generated_script = generate_script("Jerry: So I was thinking...", length=100, temperature=0.8)
    print(generated_script)
    ```

4. **Customize Script Generation**  
    You can fine-tune parameters to control output style:
    
    - **Prompt**: Defines how the dialogue begins.
    - **Length**: Number of words in the generated script.
    - **Temperature**: Controls randomness (higher = more creative, lower = more structured).
    
    Example usage:
    ```bash
    generate_script("Elaine: I need coffee.", length=150, temperature=0.7)
    ```

5. **View and Save Generated Scripts**  
    All generated scripts can be saved to `generated_script.txt` for easy reference.

---

## Configuration

There are no special configuration settings for this project.

---

## Testing

To test the model and generation pipeline, you can try it out by generating your own TV scripts.

---

## License

This project is licensed under the [MIT License](LICENSE).  
![License](https://img.shields.io/badge/license-MIT-blue)

---

## Acknowledgments

Special thanks to:
- The creators of Seinfeld for inspiring this project.
- OpenAI’s community for discussions on text generation.
- PyTorch for making deep learning accessible.

---

## FAQs

**Q: What type of model does SceneScribe use?**  
**A**: SceneScribe uses a Recurrent Neural Network (RNN) trained on Seinfeld scripts to generate new TV scripts in a similar style.

**Q: Can I train this model on different TV scripts?**  
**A:** Yes! Replace `seinfeld_scripts.txt` with your own dataset and retrain the model.

**Q: How long does training take?**  
**A:** Depending on hardware, training may take an hour or so on a GPU. Using CUDA can significantly speed up the process.

---

## Contact Information

You can reach me at [vidhithakkar.ca@gmail.com](mailto:vidhithakkar.ca@gmail.com) or via [LinkedIn](https://www.linkedin.com/in/vidhi-thakkar-0b509724a/).