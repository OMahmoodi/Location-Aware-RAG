import argparse
from src.build_rag_agent import build_rag_agent


def run_interactive(folder_path: str):
    print("[✓] Loading RAG agent...")
    qa_chain = build_rag_agent(folder_path)
    print("[✓] Agent ready. Type your questions (type 'exit' to quit):\n")

    while True:
        query = input("Q: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            result = qa_chain.invoke(query)
            answer = result['result'].split('Answer:\n')[-1].strip()
            print(f"A: {answer}\n")
        except Exception as e:
            print(f"[!] Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Run RAG agent interactively")
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default="pdfreports",
        help="Directory containing scanned geoscience report PDFs",
    )
    args = parser.parse_args()
    run_interactive(args.pdf_dir)


if __name__ == "__main__":
    main()
