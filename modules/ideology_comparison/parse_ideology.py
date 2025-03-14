import re
import json

def parse_markdown(markdown_text):
    # Split the markdown text into two parts using the divider '---'
    parts = markdown_text.strip().split('---')
    if len(parts) < 2:
        raise ValueError("Markdown format is missing the divider '---'")
    
    pre_section = parts[0].strip()
    post_section = parts[1].strip()

    # Parse the potential bias block from the pre-divider section.
    # This captures everything between "- **Potential Bias**:" and "- **Overall Assessment**:"
    potential_bias_match = re.search(
        r"- \*\*Potential Bias\*\*:\s*\n(.*?)\n- \*\*Overall Assessment\*\*:",
        pre_section,
        re.DOTALL
    )
    potential_bias_list = []
    if potential_bias_match:
        potential_bias_text = potential_bias_match.group(1).strip()
        # Define a pattern to capture each bias block.
        bias_block_pattern = re.compile((
                r"- \*\*Bias\*\*:\s*(.*?)\n\s*"
                r"- \*\*Bias Type\*\*:\s*(.*?)\n\s*"
                r"- \*\*News Event\*\*:\s*(.*?)\n\s*"
                r"- \*\*Left-Wing Perspective\*\*:\s*(.*?)\n\s*"
                r"- \*\*Right-Wing Perspective\*\*:\s*(.*?)\n\s*"
                r"- \*\*Bias Analysis\*\*:\s*(.*?)\n\s*"
                r"- \*\*Bias Favoring\*\*:\s*(.*?)"
                r"(?=\n\s*- \*\*Potential Bias\*\*:|\Z)"
            ),
            re.DOTALL
        )
        for match in bias_block_pattern.finditer(potential_bias_text):
            bias_dict = {
                "bias": match.group(1).strip(),
                "bias_type": match.group(2).strip(),
                "news_event": match.group(3).strip(),
                "left_wing_perspective": match.group(4).strip(),
                "right_wing_perspective": match.group(5).strip(),
                "bias_analysis": match.group(6).strip(),
                "bias_favoring": match.group(7).strip()
            }
            potential_bias_list.append(bias_dict)

    # Parse the overall assessment from the pre-divider section.
    overall_assessment_match = re.search(
        r"- \*\*Overall Assessment\*\*: (.*)",
        pre_section
    )
    overall_assessment = overall_assessment_match.group(1).strip() if overall_assessment_match else ""

    # Parse the logit scores, softmax probabilities, and categorical label from the post-divider section.
    logit_pattern = (
        r"- \*\*Logit Scores\*\*:\s*\n"
        r"\s*- \*\*Left\*\*: ([\d\.\-]+)\s*\n"
        r"\s*- \*\*Left-Center\*\*: ([\d\.\-]+)\s*\n"
        r"\s*- \*\*Center\*\*: ([\d\.\-]+)\s*\n"
        r"\s*- \*\*Center-Right\*\*: ([\d\.\-]+)\s*\n"
        r"\s*- \*\*Right\*\*: ([\d\.\-]+)\s*\n"
        r"\s*- \*\*Unbiased\*\*: ([\d\.\-]+)\s*\n"
        r"- \*\*Softmax Probabilities\*\*:\s*\n"
        r"\s*- \*\*Left\*\*: ([\d\.\-]+)%\s*\n"
        r"\s*- \*\*Left-Center\*\*: ([\d\.\-]+)%\s*\n"
        r"\s*- \*\*Center\*\*: ([\d\.\-]+)%\s*\n"
        r"\s*- \*\*Center-Right\*\*: ([\d\.\-]+)%\s*\n"
        r"\s*- \*\*Right\*\*: ([\d\.\-]+)%\s*\n"
        r"\s*- \*\*Unbiased\*\*: ([\d\.\-]+)%\s*\n"
        r"- \*\*Categorical Label\*\*: ([\w-]+)"
    )
    logit_match = re.search(logit_pattern, post_section, re.DOTALL)
    if logit_match:
        logit_scores = {
            "Left": float(logit_match.group(1)),
            "Left-Center": float(logit_match.group(2)),
            "Center": float(logit_match.group(3)),
            "Center-Right": float(logit_match.group(4)),
            "Right": float(logit_match.group(5)),
            "Unbiased": float(logit_match.group(6))
        }
        softmax_probabilities = {
            "Left": float(logit_match.group(7)) / 100.0,
            "Left-Center": float(logit_match.group(8)) / 100.0,
            "Center": float(logit_match.group(9)) / 100.0,
            "Center-Right": float(logit_match.group(10)) / 100.0,
            "Right": float(logit_match.group(11)) / 100.0,
            "Unbiased": float(logit_match.group(12)) / 100.0
        }
        categorical_label = logit_match.group(13)
    else:
        raise ValueError(f"Post-divider markdown format is incorrect: {markdown_text}")

    return {
        "potential_bias": potential_bias_list,
        "overall_assessment": overall_assessment,
        "logit_scores": logit_scores,
        "softmax_probabilities": softmax_probabilities,
        "label": categorical_label
    }

# Example usage
if __name__ == "__main__":
    markdown_text = """
```markdown
- **Potential Bias**:
  - **Bias**: Emphasis on sensational headlines.
  - **Bias Type**: Sensationalism Bias, Clickbait
  - **News Event**: Celebrity scandal
  - **Left-Wing Perspective**: Critique of mainstream media narrative.
  - **Right-Wing Perspective**: Defense of traditional values in journalism.
  - **Bias Analysis**: The article uses hyperbolic language to attract attention, which is common in sensationalist reporting.
  - **Bias Favoring**: left
- **Potential Bias**:
  - **Bias**: Emphasis on sensational headlines.
  - **Bias Type**: Sensationalism Bias, Clickbait
  - **News Event**: Celebrity scandal
  - **Left-Wing Perspective**: Critique of mainstream media narrative.
  - **Right-Wing Perspective**: Defense of traditional values in journalism.
  - **Bias Analysis**: The article uses hyperbolic language to attract attention, which is common in sensationalist reporting.
  - **Bias Favoring**: right
- **Overall Assessment**: The article shows a moderate level of sensationalism with some political undertones.

---

- **Logit Scores**:
  - **Left**: -1.2
  - **Left-Center**: -0.5
  - **Center**: 2.8
  - **Center-Right**: -0.6
  - **Right**: -1.3
    - **Unbiased**: 0.0

- **Softmax Probabilities**:
  - **Left**: 3.47%
  - **Left-Center**: 10.21%
  - **Center**: 76.32%
  - **Center-Right**: 9.56%
  - **Right**: 0.44%
    - **Unbiased**: 0.0%

- **Categorical Label**: center
```
"""
    result = parse_markdown(markdown_text)
    print(json.dumps(result, indent=2))
