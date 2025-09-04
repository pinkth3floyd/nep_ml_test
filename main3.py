import gradio as gr
from pipeline import NepaliGrammarCorrector

# Load models from Hugging Face Hub
corrector = NepaliGrammarCorrector(
    ged_path="sparshrestha/muril-sentence-ged-100k",
    token_path="sparshrestha/muril-token-ged-100k",
    mlm_path="sparshrestha/muril-mlm-50k"
)

def process(sentence):
    result = corrector.correct(sentence)
    if result["status"] == "Correct":
        return "✅ Correct", "-", "-", result["original"]

    # Format top-5 suggestions
    suggestion_lines = []
    for pos, suggs in result["suggestions"].items():
        token = result["incorrect_tokens"][result["incorrect_positions"].index(pos)]
        suggestion_lines.append(f"{token} → {', '.join(suggs)}")

    return (
        "❌ Incorrect",
        ", ".join(result["incorrect_tokens"]),
        "\n".join(suggestion_lines),
        result["corrected_sentence"]
    )

demo = gr.Interface(
    fn=process,
    inputs=gr.Textbox(label="Enter Nepali Sentence"),
    outputs=[
        gr.Textbox(label="Sentence Status"),
        gr.Textbox(label="Incorrect Tokens"),
        gr.Textbox(label="Top-5 Suggestions"),
        gr.Textbox(label="Corrected Sentence")
    ],
    title="Nepali Grammar Correction",
    description="A MuRIL-finetuned tool for GED + GEC."
)

if __name__ == "__main__":
    demo.launch()