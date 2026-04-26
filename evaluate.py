from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

def run_ragas_eval(questions, answers, contexts, ground_truths):
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,       # list of lists of retrieved chunks
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy,
                 context_recall, context_precision],
    )
    print(result)
    return result

# Key metrics explained:
# faithfulness      — is the answer grounded in the retrieved context?
# answer_relevancy  — does the answer actually address the question?
# context_recall    — did retrieval find the right chunks?
# context_precision — are retrieved chunks precise (no noise)?

