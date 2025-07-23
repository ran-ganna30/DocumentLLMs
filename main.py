import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline

nltk.download('punkt')

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''.join(page.extract_text() for page in pdf.pages if page.extract_text())
    return text


def summarize_text(text, chunk_size=1000):
    summarizer = pipeline("summarization", model="t5-small")
    summaries = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    return '\n'.join(summaries)

#Split Text into Passages (~200 words)
def split_into_passages(text, word_limit=200):
    sentences = sent_tokenize(text)
    passages, current_passage = [], ""

    for sentence in sentences:
        if len((current_passage + sentence).split()) < word_limit:
            current_passage += " " + sentence
        else:
            passages.append(current_passage.strip())
            current_passage = sentence

    if current_passage:
        passages.append(current_passage.strip())

    return passages

# Generate Questions from Text using valhalla/t5-base-qg-hl
def generate_questions(passage, qg_pipeline, min_questions=3):
    input_text = f"generate questions: {passage}"
    results = qg_pipeline(input_text)
    questions = results[0]['generated_text'].split('<sep>')
    questions = [q.strip() for q in questions if q.strip()]

    # Regenerate from sub-sentences if fewer than min_questions
    if len(questions) < min_questions:
        sentences = sent_tokenize(passage)
        for i in range(len(sentences)):
            if len(questions) >= min_questions:
                break
            chunk = ' '.join(sentences[i:i+2])
            chunk_input = f"generate questions: {chunk}"
            additional = qg_pipeline(chunk_input)[0]['generated_text'].split('<sep>')
            questions += [q.strip() for q in additional if q.strip()]

    return questions[:min_questions]

# Answer Questions Using QA Model (Roberta)
def answer_questions(passages, qg_pipeline, qa_pipeline):
    answered = set()

    for idx, passage in enumerate(passages):
        print(f"\n📝 Passage {idx+1}: {passage[:250]}...\n")
        questions = generate_questions(passage, qg_pipeline)

        for q in questions:
            if q not in answered:
                answer = qa_pipeline({'question': q, 'context': passage})
                print(f"❓ Q: {q}")
                print(f"✅ A: {answer['answer']}\n")
                answered.add(q)
        print("=" * 60)

if __name__ == "__main__":
    # Path to your PDF file
    pdf_path = "google_terms_of_service_en_in.pdf"

    # Step 1
    raw_text = extract_text_from_pdf(pdf_path)
    print("[✔] PDF text extracted.")

    # Step 2
    summary = summarize_text(raw_text)
    print("\n📌 Summary:\n", summary)

    # Step 3
    passages = split_into_passages(raw_text)
    print(f"\n[✔] Document split into {len(passages)} passages.")

    # Step 4 & 5
    qg_pipeline = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    answer_questions(passages, qg_pipeline, qa_pipeline)
