from veripatch.studio_intent import classify_intent, contradictory_verification_request


def test_literal_test_comment_is_not_a_request_to_run_tests():
    message = "在文件最开头加一行注释：# flash-test。不修改其他文件，仍然不要运行命令。"
    assert not contradictory_verification_request(message)
    assert classify_intent(message).intent == "change"
    assert not classify_intent(message).verification_requested


def test_past_test_question_does_not_authorize_execution():
    policy = classify_intent("刚才具体改了什么？测试运行了吗？")
    assert not policy.verification_requested
    assert not policy.mutation_requested
    assert policy.intent == "answer"


def test_actual_conflicting_test_instruction_is_preserved():
    assert contradictory_verification_request("修改 hello.py，完成后运行测试，但不要运行任何命令。")
