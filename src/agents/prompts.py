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

                        Your job is to extract word pairs and justifications exactly as the student said them — even if they are unusual or incorrect.

                        - Do not reinterpret or "fix" student responses.
                        - Do not infer better or more plausible pairings.
                        - Only extract what the student actually wrote, exactly as they wrote it.
                        - If the student says "cat and bone", extract ("cat", "bone") — even if you think "dog and bone" would make more sense.

                        Important: if a pair is not explicitly stated, do not include it.

                        Example:

                        Student: "cat and bone because cats chew bones, and cat and dog because they are animals."

                        Correct Extraction:
                        (cat, bone): "cats chew bones"
                        (cat, dog): "they are animals"

                        Incorrect Extraction:
                        (dog, bone): "dogs like bones" ← This was not said and must not be added.

                    """,


        "evaluation": """
                        You are evaluating a child's Vocabulary Awareness (VA) response.
                        Each item gives a triplet (e.g., "light, sun, feather"). The child selects 2 word pairs and gives a justification for each.

                        Your task:
                        1. Evaluate if each word pair and justification are valid.
                        2. Score: 1 = both pair and justification are valid; 0 = otherwise.
                        3. If score is 0, assign an error category and a short explanation.
                        4. Compute total score (max = 2).

                        Error categories:
                        - semantic_mismatch: the words aren’t meaningfully related.
                        - justification_vague: explanation is too vague or generic.
                        - off_topic: the response is unrelated to the task or triplet.
                        - incomplete: fewer than 2 pairs or missing justification.
                        - other: doesn’t fit above.

                        Be strict:
                        - Only accept pairs with clear, commonly understood semantic relationships.
                        - Reject guesses, puns, or surface-level links.
                        - If the justification sounds okay but the words don’t belong together, mark as invalid.

                        Example:

                        Triplet: (light, sun, feather)  
                        Expected Pairs:  
                        - (light, sun): because the sun gives off light  
                        - (light, feather): because a feather is light / not heavy

                        Student Pair 1:  
                        Words: light, sun  
                        Justification: the sun produces light  
                        Pair valid: yes  
                        Justification valid: yes  
                        Score: 1  
                        Error Category: none  
                        Error Reasoning: Clear and accurate semantic link.

                        Student Pair 2:  
                        Words: sun, feather  
                        Justification: both are light  
                        Pair valid: no  
                        Justification valid: no  
                        Score: 0  
                        Error Category: semantic_mismatch  
                        Error Reasoning: “Light” has different meanings here; this pair lacks a valid semantic relationship.
                    """,
    },
}