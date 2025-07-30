ASSESSMENT_PROMPTS = {

    "Vocabulary Awareness": {
        "description": """
                        Each challenge presents a triplet of words (e.g. "dog-cat-bone") and the student is asked to 
                        choose 2 words that go together and justify their choice. For each triplet, the student must 
                        provide 2 pairs with justifications for each.

                        ### Examples:
                        dog-cat-bone: (dog, cat), because they are both animals
                                    (dog, bone), because dogs like bones

                        light-sun-feather: (light, sun), because the sun produces light
                                        (light, feather), because a feather is light / not heavy
                    """,
        

        "extraction": """
                        You are given student responses to Vocabulary Awareness (VA) challenges.

                        Your job is to extract word pairs and justifications exactly as the student said them — even if they are unusual or incorrect.

                        - Do not reinterpret or "fix" student responses.
                        - Do not infer better or more plausible pairings.
                        - Only extract what the student actually wrote, exactly as they wrote it.
                        - If the student says "cat and bone", extract ("cat", "bone") — even if you think "dog and bone" would make more sense.

                        Important: if a pair is not explicitly stated, do not include it.


                        ---

                        Example:

                        Student: "cat and bone because cats chew bones, and cat and dog because they are animals."

                        Correct Extraction:
                        (cat, bone): "cats chew bones"
                        (cat, dog): "they are animals"

                        Incorrect Extraction:
                        (dog, bone): "dogs like bones" ← This was not said and must not be added.

                    """,


        "evaluation": """
                        You are an expert educational evaluator assessing a child's responses to Vocabulary Awareness (VA) challenges.

                        Each challenge presents a triplet of words (e.g., "dog, cat, bone"). The student is asked to choose two word pairs and provide a justification for each.

                        Your task is to:
                        1. Evaluate the correctness of each pair and its justification.
                        2. Score each pair: 1 if both the word pair and justification are valid; 0 otherwise.
                        3. For any incorrect response, classify the type of error.
                        4. Compute the total score

                        ### Error categories:
                        - semantic_mismatch: the selected word pair does not share a valid or meaningful relationship.
                        - justification_vague: the explanation is unclear.
                        - off_topic: the response doesn’t relate to the triplet or task at all.
                        - incomplete: the student provided fewer than two pairs or left parts blank.
                        - other: use only if none of the above apply.

                        Be strict. Only mark the pair and justification as valid if:
                        - The two words have a strong, commonly understood semantic relationship.
                        - The justification is clear, precise, and correct.
                        Do not accept guesses, unusual logic, or surface-level associations.

                        If the justification seems plausible but the word pair is unrelated or based on a pun or homonym, mark it as invalid.
                        Do not let surface-level similarity override actual meaning.

                        ---

                        ### Example:

                        Triplet: (dog, cat, bone)
                        Expected Pairs: dog, cat; dog, bone 

                        # Student Pair 1:
                        Words: dog, cat  
                        Justification: both are animals
                        Pair valid: yes  
                        Justification valid: yes  
                        Score: 1
                        Error Category: None  
                        Error Reasoning: The pair and justification are both valid

                        ---

                        # Student Pair 2:
                        Words: cat, bone  
                        Justification: cats chew bones 
                        Pair valid: no  
                        Justification valid: no  
                        Score: 0
                        Error Category: semantic_mismatch  
                        Error Reasoning: The words cat and bone are not semantically related in the intended way

                    """,
    },
}