import torch
from urllib.parse import urlparse


def domain_of_url(url: str) -> str:
    return urlparse(url).netloc


def dedent(text: str) -> str:
    def count_leading_whitespaces(line: str) -> tuple[int, str]:
        stripped = line.lstrip()
        return len(line) - len(stripped), stripped
    
    leading_whitespace_count = -1
    lines = text.splitlines()
    output: list[str] = []
    for line in lines:
        whitespaces_count, stripped = count_leading_whitespaces(line)
        if stripped:  # Check if the line is non-empty (i.e., it contains non-whitespace characters)
            if leading_whitespace_count < 0:
                leading_whitespace_count = whitespaces_count
                output.append(stripped)
            else:
                if whitespaces_count <= leading_whitespace_count:
                    output.append(stripped)
                else:
                    output.append(line[leading_whitespace_count:])
        else:
            output.append("")
    return "\n".join(output)


def torch_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"  # Apple silicon
    elif torch.cuda.is_available():
        device = "cuda"
    else: 
        device = "cpu"
    return device
