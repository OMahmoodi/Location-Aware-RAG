import json
from sentence_transformers import SentenceTransformer, util


def compute_similarity(reference, response, model):
    emb1 = model.encode(reference, convert_to_tensor=True)
    emb2 = model.encode(response, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2)
    return float(score[0][0])


def evaluate_agent(qa_chain, query_data_path):
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    with open(query_data_path, "r") as f:
        query_data = json.load(f)

    results = []
    for pair_dict in query_data:
        query = pair_dict["query"]
        expected_answer = pair_dict["groundtruth"]

        result = qa_chain.invoke(query)["result"]
        generated_answer = result.split("Answer:\n")[-1].strip() if "Answer:" in result else result

        cosine_similarity = compute_similarity(expected_answer, generated_answer, model)

        results.append({
            "query": query,
            "generated_answer": generated_answer,
            "expected_answer": expected_answer,
            "cosine_similarity": cosine_similarity
        })

    return results


if __name__ == "__main__":
    from src.build_rag_agent import build_rag_agent

    qa_chain = build_rag_agent("pdfreports")
    eval_results = evaluate_agent(qa_chain, "query_examples.json")

    os.makedirs("results", exist_ok=True)
    with open("results/eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    print("[✓] Evaluation completed. Results saved to results/eval_results.json")
