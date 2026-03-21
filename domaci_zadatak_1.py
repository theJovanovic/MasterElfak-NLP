from collections import Counter
import queue
import threading
import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import spacy

# nlp = spacy.load("en_core_web_sm")

nlp = spacy.load("xx_ent_wiki_sm")
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

result_queue = queue.Queue()
bar_canvas = None
pie_canvas = None
bar_figure = None
pie_figure = None


def set_status(message: str) -> None:
    status_var.set(message)


def clear_charts() -> None:
    global bar_canvas, pie_canvas, bar_figure, pie_figure

    if bar_canvas is not None:
        bar_canvas.get_tk_widget().destroy()
        bar_canvas = None
    if pie_canvas is not None:
        pie_canvas.get_tk_widget().destroy()
        pie_canvas = None

    if bar_figure is not None:
        plt.close(bar_figure)
        bar_figure = None
    if pie_figure is not None:
        plt.close(pie_figure)
        pie_figure = None


def analysis_worker(text: str) -> None:
    try:
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]
        entities = [ent.text for ent in doc.ents]
        entity_labels = [ent.label_ for ent in doc.ents]

        entity_freq = Counter(entities)
        label_freq = Counter(entity_labels)

        top_entities = [ent for ent, _ in entity_freq.most_common(5)]
        summary_sentences = []

        for sent in sentences:
            for ent in top_entities:
                if ent in sent:
                    summary_sentences.append(sent)
                    break

        summary = " ".join(summary_sentences).strip()

        result_queue.put(
            {
                "ok": True,
                "entity_freq": entity_freq,
                "label_freq": label_freq,
                "summary": summary,
            }
        )
    except Exception as exc:
        result_queue.put({"ok": False, "error": str(exc)})


def analyze_text() -> None:
    text = input_text.get("1.0", tk.END)

    stats_text.configure(state=tk.NORMAL)
    summary_text.configure(state=tk.NORMAL)
    stats_text.delete("1.0", tk.END)
    summary_text.delete("1.0", tk.END)

    if not text.strip():
        stats_text.insert(tk.END, "Please enter text to analyze.")
        summary_text.insert(tk.END, "")
        stats_text.configure(state=tk.DISABLED)
        summary_text.configure(state=tk.DISABLED)
        clear_charts()
        set_status("Ready")
        return

    analyze_button.configure(state=tk.DISABLED)
    set_status("Analyzing...")

    worker = threading.Thread(target=analysis_worker, args=(text,), daemon=True)
    worker.start()


def poll_analysis_result() -> None:
    try:
        while True:
            result = result_queue.get_nowait()
            handle_result(result)
    except queue.Empty:
        pass
    root.after(100, poll_analysis_result)


def handle_result(result: dict) -> None:
    analyze_button.configure(state=tk.NORMAL)

    stats_text.configure(state=tk.NORMAL)
    summary_text.configure(state=tk.NORMAL)
    stats_text.delete("1.0", tk.END)
    summary_text.delete("1.0", tk.END)

    if not result.get("ok"):
        error_message = result.get("error", "Unknown error")
        set_status(f"Error: {error_message}")
        stats_text.insert(tk.END, f"Analysis failed:\n{error_message}")
        clear_charts()
        stats_text.configure(state=tk.DISABLED)
        summary_text.configure(state=tk.DISABLED)
        return

    entity_freq = result["entity_freq"]
    label_freq = result["label_freq"]
    summary = result["summary"]

    if entity_freq:
        stats_text.insert(tk.END, "Entity Frequencies\n\n")
        for ent, freq in entity_freq.most_common():
            stats_text.insert(tk.END, f"{ent} : {freq}\n")
    else:
        stats_text.insert(tk.END, "No named entities found in the provided text.")

    if summary:
        summary_text.insert(tk.END, summary)
    else:
        summary_text.insert(tk.END, "No summary could be generated from entities.")

    clear_charts()
    if entity_freq:
        draw_bar_chart(entity_freq)
    if label_freq:
        draw_pie_chart(label_freq)

    set_status("Done")
    stats_text.configure(state=tk.DISABLED)
    summary_text.configure(state=tk.DISABLED)


def draw_bar_chart(entity_freq: Counter) -> None:
    global bar_canvas, bar_figure

    entities = [e for e, _ in entity_freq.most_common(10)]
    freqs = [f for _, f in entity_freq.most_common(10)]

    bar_figure = plt.Figure(figsize=(5, 4), dpi=100)
    ax = bar_figure.add_subplot(111)
    ax.bar(entities, freqs)
    ax.set_title("Top Named Entities")
    ax.set_xlabel("Entity")
    ax.set_ylabel("Frequency")
    ax.tick_params(axis="x", rotation=45)
    bar_figure.tight_layout()

    ax.yaxis.get_major_locator().set_params(integer=True)
    bar_canvas = FigureCanvasTkAgg(bar_figure, master=bar_frame)
    bar_canvas.draw()
    bar_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def draw_pie_chart(label_freq: Counter) -> None:
    global pie_canvas, pie_figure

    labels = list(label_freq.keys())
    sizes = list(label_freq.values())

    pie_figure = plt.Figure(figsize=(5, 4), dpi=100)
    ax = pie_figure.add_subplot(111)
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Entity Type Distribution")
    pie_figure.tight_layout()

    pie_canvas = FigureCanvasTkAgg(pie_figure, master=pie_frame)
    pie_canvas.draw()
    pie_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def create_scrolled_text(parent, height=8, width=40):
    container = ttk.Frame(parent)
    container.grid_rowconfigure(0, weight=1)
    container.grid_columnconfigure(0, weight=1)

    text_widget = tk.Text(container, wrap=tk.WORD, height=height, width=width)
    scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=text_widget.yview)
    text_widget.configure(yscrollcommand=scrollbar.set)

    text_widget.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")
    return container, text_widget


if __name__ == "__main__":

    root = tk.Tk()
    root.title("NLP Text Summarization and Entity Analysis")
    root.geometry("1200x800")
    root.minsize(980, 680)

    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")

    style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"))
    style.configure("Card.TLabelframe", padding=10)
    style.configure("Card.TLabelframe.Label", font=("Segoe UI", 11, "bold"))
    style.configure("TButton", padding=6)

    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(3, weight=1)
    root.grid_rowconfigure(4, weight=1)

    header = ttk.Frame(root)
    header.grid(row=0, column=0, sticky="ew", padx=20, pady=(14, 8))
    header.grid_columnconfigure(0, weight=1)

    title = ttk.Label(header, text="Named Entity Based Text Summarizer", style="Title.TLabel")
    title.grid(row=0, column=0, sticky="w")

    status_var = tk.StringVar(value="Ready")
    status_label = ttk.Label(header, textvariable=status_var)
    status_label.grid(row=0, column=1, sticky="e")

    input_frame = ttk.LabelFrame(root, text="Input", style="Card.TLabelframe")
    input_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 10))
    input_frame.grid_columnconfigure(0, weight=1)
    input_frame.grid_rowconfigure(0, weight=1)

    input_container, input_text = create_scrolled_text(input_frame, height=8, width=120)
    input_container.grid(row=0, column=0, sticky="nsew")

    analyze_button = ttk.Button(root, text="Analyze Text", command=analyze_text)
    analyze_button.grid(row=2, column=0, sticky="w", padx=20, pady=(0, 10))

    middle_frame = ttk.Frame(root)
    middle_frame.grid(row=3, column=0, sticky="nsew", padx=20, pady=(0, 10))
    middle_frame.grid_columnconfigure(0, weight=1)
    middle_frame.grid_columnconfigure(1, weight=1)
    middle_frame.grid_rowconfigure(0, weight=1)

    stats_frame = ttk.LabelFrame(middle_frame, text="Entity Statistics", style="Card.TLabelframe")
    stats_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
    stats_frame.grid_columnconfigure(0, weight=1)
    stats_frame.grid_rowconfigure(0, weight=1)

    stats_container, stats_text = create_scrolled_text(stats_frame, height=15, width=45)
    stats_container.grid(row=0, column=0, sticky="nsew")
    stats_text.configure(state=tk.DISABLED)

    summary_frame = ttk.LabelFrame(middle_frame, text="Generated Summary", style="Card.TLabelframe")
    summary_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
    summary_frame.grid_columnconfigure(0, weight=1)
    summary_frame.grid_rowconfigure(0, weight=1)

    summary_container, summary_text = create_scrolled_text(summary_frame, height=15, width=65)
    summary_container.grid(row=0, column=0, sticky="nsew")
    summary_text.configure(state=tk.DISABLED)

    charts_container = ttk.LabelFrame(root, text="Visualizations", style="Card.TLabelframe")
    charts_container.grid(row=4, column=0, sticky="nsew", padx=20, pady=(0, 14))
    charts_container.grid_columnconfigure(0, weight=1)
    charts_container.grid_columnconfigure(1, weight=1)
    charts_container.grid_rowconfigure(0, weight=1)

    bar_frame = ttk.Frame(charts_container)
    bar_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

    pie_frame = ttk.Frame(charts_container)
    pie_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

    root.after(100, poll_analysis_result)
    root.mainloop()