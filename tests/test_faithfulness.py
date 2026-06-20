import importlib.util
import os

# Load eval/faithfulness.py as a module (the eval/ dir is not a package).
_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "faithfulness.py")
_spec = importlib.util.spec_from_file_location("faithfulness", _path)
faithfulness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(faithfulness)


def test_parse_judge_scores_extracts_floats():
    text = '{"groundedness": 0.9, "relevance": 1.0, "correctness": 0.8}'
    out = faithfulness.parse_judge_scores(text)
    assert out == {"groundedness": 0.9, "relevance": 1.0, "correctness": 0.8}


def test_parse_judge_scores_tolerates_surrounding_text():
    text = 'Here is my rating:\n{"groundedness": 0.5, "relevance": 0.6, "correctness": 0.7}\nThanks.'
    out = faithfulness.parse_judge_scores(text)
    assert out["groundedness"] == 0.5
    assert out["correctness"] == 0.7
