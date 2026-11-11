import unittest

from eval_agents import (
    CASES,
    evaluate_outcome_and_trajectory,
    evaluate_tool_arguments,
    is_t4_best_experiment_answer,
    is_t5_mean_arguments,
)
from mcp_server import calculate


T5_CASE = next(case for case in CASES if case.name == "T5_composed_task")


class EvaluatorTests(unittest.TestCase):
    def test_t4_accepts_equivalent_experiment_c_names(self) -> None:
        for answer in (
            "experiment_c 的得分是 0.91",
            "experiment c 的得分是 0.91",
            "实验 C，得分为 0.91",
            "实验C的得分为0.91",
        ):
            with self.subTest(answer=answer):
                self.assertTrue(is_t4_best_experiment_answer(answer))

    def test_t4_rejects_wrong_experiment(self) -> None:
        self.assertFalse(is_t4_best_experiment_answer("实验 B，得分为 0.91"))

    def test_structured_calculate_mean(self) -> None:
        self.assertAlmostEqual(calculate("mean", [0.82, 0.76, 0.91]), 0.83)

    def test_structured_calculate_subtract(self) -> None:
        self.assertAlmostEqual(calculate("subtract", [0.91, 0.76]), 0.15)

    def test_structured_calculate_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "mean requires"):
            calculate("mean", [])
        with self.assertRaisesRegex(ValueError, "subtract requires"):
            calculate("subtract", [0.91])

    def test_structured_calculate_rejects_invalid_operation(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported operation"):
            calculate("multiply", [0.82, 0.76])  # type: ignore[arg-type]

    def test_t5_accepts_mean_with_score_order_changes(self) -> None:
        self.assertTrue(
            is_t5_mean_arguments(
                {"operation": "mean", "values": [0.82, 0.76, 0.91]}
            )
        )
        self.assertTrue(
            is_t5_mean_arguments(
                {"operation": "mean", "values": [0.91, 0.82, 0.76]}
            )
        )

    def test_t5_rejects_wrong_structured_arguments(self) -> None:
        self.assertFalse(
            is_t5_mean_arguments(
                {"operation": "mean", "values": [0.82, 0.76]}
            )
        )
        self.assertFalse(
            is_t5_mean_arguments({"operation": "mean"})
        )
        self.assertFalse(
            is_t5_mean_arguments(
                {"operation": "subtract", "values": [0.91, 0.76]}
            )
        )

    def test_t5_tool_argument_validation(self) -> None:
        correct_calls = [
            {"tool_name": "get_fkg_experiments", "args": {}},
            {
                "tool_name": "calculate",
                "args": '{"operation":"mean","values":[0.91,0.82,0.76]}',
            },
        ]
        incorrect_calls = [
            {"tool_name": "get_fkg_experiments", "args": {}},
            {
                "tool_name": "calculate",
                "args": '{"operation":"mean","values":[85,90,95]}',
            },
        ]

        self.assertTrue(evaluate_tool_arguments(T5_CASE, correct_calls))
        self.assertFalse(evaluate_tool_arguments(T5_CASE, incorrect_calls))

    def test_self_correction_keeps_outcome_but_marks_trajectory(self) -> None:
        success, trajectory_correct = evaluate_outcome_and_trajectory(
            answer_correct=True,
            tool_selection_correct=True,
            tool_argument_correct=False,
            extra_tool_calls=0,
        )

        self.assertTrue(success)
        self.assertFalse(trajectory_correct)


if __name__ == "__main__":
    unittest.main()
