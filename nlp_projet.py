# -*- coding: utf-8 -*-



import os 
os.environ["HF_HOME"] = "D:/huggingface_cache"

import gradio as gr
import nltk
from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util

# Tab 1 : Tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Tab 2 : Sentiment Analysis
sentiment_pipeline = pipeline(
    "text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    top_k=None
)

# Tab 3 : Semantic Similarity
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# Tab 4 : Zero-Shot Classification
# Note : Ce modèle fait environ 1.6 Go, le premier téléchargement peut prendre quelques minutes.
zero_shot_pipeline = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# Tab 5 : NER (Named Entity Recognition)
# L'aggregation_strategy="simple" regroupe les sous-mots pour reformer les mots complets
ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

# FONCTIONS DE TRAITEMENT
def tokenize_text(text):
    nltk_tokens = nltk.word_tokenize(text)
    hf_tokens = hf_tokenizer.tokenize(text)
    return ", ".join(nltk_tokens), len(nltk_tokens), ", ".join(hf_tokens), len(hf_tokens)

def analyze_sentiment(text):
    results = sentiment_pipeline(text)[0]
    return {res['label']: res['score'] for res in results}

def calculate_similarity(text1, text2):
    embedding1 = similarity_model.encode(text1)
    embedding2 = similarity_model.encode(text2)
    cosine_score = util.cos_sim(embedding1, embedding2)[0][0].item()
    final_score = max(0.0, cosine_score) * 100
    return f"{final_score:.2f}%"

def zero_shot_classify(text, labels_string):
    """Sépare la chaîne de labels par les virgules et classe le texte."""
    # Nettoyage de la liste fournie par l'utilisateur (ex: "Tech, Sports" -> ["Tech", "Sports"])
    candidate_labels = [label.strip() for label in labels_string.split(",") if label.strip()]

    if not candidate_labels:
        return {"Please enter valid labels": 1.0}

    result = zero_shot_pipeline(text, candidate_labels)
    # Reformatage pour le composant gr.Label de Gradio
    return {label: score for label, score in zip(result['labels'], result['scores'])}

def extract_entities(text):
    """Extrait les entités et formate la sortie pour gr.HighlightedText."""
    entities = ner_pipeline(text)
    highlighted_output = []
    last_end = 0

    for ent in entities:
        start, end = ent['start'], ent['end']

        # Ajouter le texte normal avant l'entité
        if start > last_end:
            highlighted_output.append((text[last_end:start], None))

        # Ajouter l'entité avec son label (ex: PER pour Personne, LOC pour Lieu)
        highlighted_output.append((text[start:end], ent['entity_group']))
        last_end = end

    # Ajouter le reste du texte après la dernière entité
    if last_end < len(text):
        highlighted_output.append((text[last_end:], None))

    return highlighted_output



# INTERFACE GRADIO
mon_theme_perso = gr.Theme.from_hub("freddyaboulton/dracula_revamped")
mon_theme_perso = mon_theme_perso.set(
    button_primary_background_fill_dark="#ff5555",       # Rouge vif pour le bouton
    button_primary_background_fill_hover_dark="#ff7777", # Rouge clair au survol
    button_primary_text_color_dark="#ffffff",            # Texte en blanc
    block_border_color_dark="#ff5555"                    # (Optionnel) Bordures des blocs en rouge
)
with gr.Blocks(title="NLP Toolkit", theme=mon_theme_perso) as demo:

    gr.Markdown("# Massyl's NLP Toolkit")
    gr.Markdown("A multi-tabbed web application demonstrating core NLP text processing pipelines.")

    with gr.Tabs():

        # TAB 1
        with gr.TabItem("Tokenization"):
            gr.Markdown("### Visualize how different models 'see' text")
            text_input_t1 = gr.Textbox(label="Input Text", value="Gradio makes building NLP web apps incredibly easy!", lines=2)
            tokenize_btn = gr.Button("Tokenize Text", variant="primary")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("####  NLTK (Word-level)")
                    nltk_output_tokens = gr.Textbox(label="Tokens", interactive=False)
                    nltk_output_count = gr.Number(label="Total Token Count", interactive=False)
                with gr.Column():
                    gr.Markdown("####  Hugging Face - BERT (Subword-level)")
                    hf_output_tokens = gr.Textbox(label="Tokens", interactive=False)
                    hf_output_count = gr.Number(label="Total Token Count", interactive=False)
            tokenize_btn.click(fn=tokenize_text, inputs=text_input_t1, outputs=[nltk_output_tokens, nltk_output_count, hf_output_tokens, hf_output_count])

        # TAB 2
        with gr.TabItem(" Sentiment Analysis"):
            gr.Markdown("### Detect the underlying tone of the input")
            text_input_t2 = gr.Textbox(label="Input Text", value="I was initially skeptical, but this toolkit is brilliant!", lines=2)
            sentiment_btn = gr.Button("Analyze Sentiment", variant="primary")
            sentiment_output = gr.Label(num_top_classes=2, label="Confidence Scores Breakdown")
            sentiment_btn.click(fn=analyze_sentiment, inputs=text_input_t2, outputs=sentiment_output)

        #  TAB 3
        with gr.TabItem(" Semantic Similarity"):
            gr.Markdown("### Measure the 'meaning distance' between two texts")
            with gr.Row():
                text_input_a = gr.Textbox(label="Text A", value="The cat is resting on the mat.", lines=2)
                text_input_b = gr.Textbox(label="Text B", value="A feline is sleeping quietly on the rug.", lines=2)
            similarity_btn = gr.Button("Calculate Similarity", variant="primary")
            similarity_output = gr.Textbox(label="Cosine Similarity Score", interactive=False)
            similarity_btn.click(fn=calculate_similarity, inputs=[text_input_a, text_input_b], outputs=similarity_output)

        #  TAB 4
        with gr.TabItem("Zero-Shot Classification"):
            gr.Markdown("### Categorize text into arbitrary labels without specific training")
            with gr.Row():
                text_input_t4 = gr.Textbox(
                    label="Input Text",
                    value="Apple just announced a new VR headset that costs $3499. It features high-end displays and eye-tracking.",
                    lines=3
                )
                labels_input = gr.Textbox(
                    label="Categories (comma-separated)",
                    value="Technology, Politics, Sports, Finance, Cooking",
                    lines=3
                )
            zeroshot_btn = gr.Button("Classify Topic", variant="primary")
            zeroshot_output = gr.Label(label="Category Probabilities")

            zeroshot_btn.click(
                fn=zero_shot_classify,
                inputs=[text_input_t4, labels_input],
                outputs=zeroshot_output
            )

        #  TAB 5
        with gr.TabItem(" Named Entity Recognition (NER)"):
            gr.Markdown("### Extract and highlight people, places, and organizations")
            text_input_t5 = gr.Textbox(
                label="Input Text",
                value="Tim Cook, the CEO of Apple, announced that the new headquarters in Cupertino, California, will open in September.",
                lines=3
            )
            ner_btn = gr.Button("Extract Entities", variant="primary")
            # Composant spécifique pour afficher du texte avec des surlignages colorés
            ner_output = gr.HighlightedText(
                label="Identified Entities",
                color_map={"PER": "blue", "ORG": "green", "LOC": "red", "MISC": "orange"}
            )

            ner_btn.click(
                fn=extract_entities,
                inputs=text_input_t5,
                outputs=ner_output
            )

if __name__ == "__main__":
    demo.launch()
