ASSESSMENT_PROMPTS = {

    "Vocabulary Awareness": {
        "description": """
                            Each challenge presents a triplet of words (e.g. "dog-cat-bone") and the student is asked to 
                            choose 2 words that go together and justify their choice. For each triplet, the student must 
                            provide 2 pairs with justifications for each.

                            Examples:
                            dog-cat-bone: (dog, cat), because they are both animals
                                        (dog, bone), because dogs like bones

                            light-sun-feather: (light, sun), because the sun produces light
                                            (light, feather), because a feather is light / not heavy
                        """,
        

        "extraction": """
                        You are given student responses to Vocabulary Awareness (VA) challenges.

                        Your task is to extract the pairs and their justifications from the given student response.

                        Remember that a student can make atmost 2 pairs. Only extract the pairs present in the student response. Do not add or repeat pairs.

                        Example:
                        - Student Response: "I think its dog and cat because they are animals and dog and bone because dogs like bones."
                        - Output:  (dog, cat): "they are animals"
                                (dog, bone): "dogs like bones"

                    """,


        "evaluation": """
                        You are responsible for evaluating student responses to Vocabulary Awareness (VA) challenges.

                        Your task is to verify whether the student's selected word pair and their justification for each pair is valid, given the expected response. 
                        If BOTH word pair and justification are correct, score 1, else score 0. Then, compute the total score as the sum of scores for each evaluated pair. The maximum total score is 2.

                        Remember that a student can make atmost 2 pairs. Only evaluate the pairs present in the student response. Do not add or repeat evaluations.

                        Example:
                        - Triplet Pair: (light, sun, feather)

                        - Student Response:
                            (light, sun): light comes from sun
                            (sun, feather): both are light

                        - Expected Response:
                            (light, sun): because sun gives light / both are bright
                            (light, feather): because feather is light / not heavy

                        - Output:
                            (light, sun): True
                            "light comes from sun": True
                            Score: 1

                            (sun, feather): False
                            "both are light": False
                            Score: 0

                            Total Score = 1 + 0 = 2
                    """
    },
}